from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from harm_detection.config import GOLD_DIR, MODELS_DIR, REPORTS_DIR
from harm_detection.utils.io import ensure_dir, write_json
from harm_detection.utils.text import build_model_text, risk_band, token_count


AGREEMENT_METRICS_PATH = REPORTS_DIR / "agreement_metrics.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = REPORTS_DIR / "final_test_predictions.csv"
WEAK_LABELS_PATH = REPORTS_DIR / "accepted_weak_labels.csv"
PSEUDO_LABELS_PATH = REPORTS_DIR / "accepted_pseudo_labels.csv"
THRESHOLD_SWEEP_PATH = REPORTS_DIR / "validation_threshold_sweep.csv"
TRAINING_SUMMARY_PATH = MODELS_DIR / "training_summary.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"


@dataclass
class ModelBundle:
    name: str
    model: Any
    metadata: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_gold_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_parquet(GOLD_DIR / "wide_labels.parquet")
    splits = pd.read_parquet(GOLD_DIR / "splits.parquet")
    return wide, splits


def compute_agreement_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("domain_expert", "de_binary", "gpt4_turbo", "gpt_binary"),
        ("domain_expert", "de_binary", "crowdworker", "cw_binary"),
        ("gpt4_turbo", "gpt_binary", "crowdworker", "cw_binary"),
    ]
    rows: list[dict[str, Any]] = []
    for left_name, left_col, right_name, right_col in pairs:
        overlap = wide.loc[wide[left_col].notna() & wide[right_col].notna(), [left_col, right_col]]
        if overlap.empty:
            rows.append(
                {
                    "left_source": left_name,
                    "right_source": right_name,
                    "overlap_count": 0,
                    "percent_agreement": None,
                    "cohen_kappa": None,
                }
            )
            continue
        agreement = float((overlap[left_col].astype(int) == overlap[right_col].astype(int)).mean())
        kappa = float(cohen_kappa_score(overlap[left_col].astype(int), overlap[right_col].astype(int)))
        rows.append(
            {
                "left_source": left_name,
                "right_source": right_name,
                "overlap_count": int(len(overlap)),
                "percent_agreement": agreement,
                "cohen_kappa": kappa,
            }
        )
    agreement_df = pd.DataFrame(rows)
    ensure_dir(REPORTS_DIR)
    agreement_df.to_csv(AGREEMENT_METRICS_PATH, index=False)
    return agreement_df


def _prepare_model_frame(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["model_text"] = prepared.apply(
        lambda row: build_model_text(row.get("title"), row.get("description"), row.get("transcript")),
        axis=1,
    )
    prepared["token_count"] = prepared["model_text"].map(token_count)
    prepared[label_column] = prepared[label_column].astype(int)
    return prepared


def _fit_calibrated_model(
    train_text: pd.Series,
    train_y: pd.Series,
    validation_text: pd.Series,
    validation_y: pd.Series,
    sample_weight: pd.Series | None = None,
) -> Any:
    estimator = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150_000)),
            ("classifier", LogisticRegression(max_iter=2_000, solver="liblinear")),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight.to_numpy()
    estimator.fit(train_text, train_y, **fit_kwargs)
    calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv="prefit")
    calibrated.fit(validation_text, validation_y)
    return calibrated


def _safe_auc(metric_fn, y_true: pd.Series, probs: np.ndarray) -> float | None:
    if len(set(y_true.astype(int))) < 2:
        return None
    return float(metric_fn(y_true, probs))


def compute_metrics_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "auroc": _safe_auc(roc_auc_score, y_true, probabilities),
        "pr_auc": _safe_auc(average_precision_score, y_true, probabilities),
        "confusion_matrix": matrix.tolist(),
    }


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    return compute_metrics_at_threshold(y_true, probabilities, threshold=0.5)


def _predict_frame(bundle: ModelBundle, frame: pd.DataFrame, label_column: str) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = bundle.model.predict_proba(frame["model_text"])[:, 1]
    metrics = compute_metrics(frame[label_column].astype(int), probabilities)
    predictions = frame[["video_id", "split", "consensus_code", label_column, "title", "description", "transcript"]].copy()
    predictions["model_name"] = bundle.name
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= 0.5).astype(int)
    predictions["risk_band"] = predictions["probability"].map(risk_band)
    return metrics, predictions


def _build_majority_predictions(train_y: pd.Series, target_y: pd.Series) -> np.ndarray:
    prevalence = float(train_y.mean())
    return np.repeat(prevalence, len(target_y))


def estimate_weak_source_reliabilities(train_frame: pd.DataFrame) -> dict[str, float]:
    reliabilities: dict[str, float] = {}
    for source_key, column in [("gpt4_turbo", "gpt_binary"), ("crowdworker", "cw_binary")]:
        overlap = train_frame.loc[train_frame[column].notna(), ["de_binary", column]]
        if overlap.empty:
            reliabilities[source_key] = 0.5
            continue
        reliability = float((overlap["de_binary"].astype(int) == overlap[column].astype(int)).mean())
        reliabilities[source_key] = reliability

    total = sum(reliabilities.values())
    if total == 0:
        return {"gpt4_turbo": 0.5, "crowdworker": 0.5}
    return {key: value / total for key, value in reliabilities.items()}


def build_weak_labels(wide: pd.DataFrame, holdout_video_ids: set[str], weights: dict[str, float]) -> pd.DataFrame:
    candidate = wide.loc[
        wide["de_binary"].isna()
        & wide["text_present"]
        & ((wide["gpt_binary"].notna()) | (wide["cw_binary"].notna()))
        & ~wide["video_id"].isin(holdout_video_ids)
    ].copy()
    candidate["model_text"] = candidate.apply(
        lambda row: build_model_text(row.get("title"), row.get("description"), row.get("transcript")),
        axis=1,
    )
    accepted_rows: list[dict[str, Any]] = []

    for row in candidate.to_dict(orient="records"):
        components: list[tuple[float, int]] = []
        if pd.notna(row.get("gpt_binary")):
            components.append((weights["gpt4_turbo"], int(row["gpt_binary"])))
        if pd.notna(row.get("cw_binary")):
            components.append((weights["crowdworker"], int(row["cw_binary"])))
        if not components:
            continue
        weight_sum = sum(weight for weight, _ in components)
        weak_p = sum(weight * label for weight, label in components) / weight_sum
        if weak_p >= 0.75:
            weak_label = 1
        elif weak_p <= 0.25:
            weak_label = 0
        else:
            continue
        accepted_rows.append(
            {
                "video_id": row["video_id"],
                "model_text": row["model_text"],
                "weak_label": weak_label,
                "weak_p": weak_p,
                "sample_weight": 0.5 + abs(weak_p - 0.5),
                "source_count": len(components),
            }
        )

    return pd.DataFrame(accepted_rows)


def build_pseudo_labels(
    wide: pd.DataFrame,
    holdout_video_ids: set[str],
    weak_bundle: ModelBundle,
) -> tuple[pd.DataFrame, dict[str, int]]:
    with_any_text = wide.loc[
        wide["de_binary"].isna()
        & wide["gpt_binary"].isna()
        & wide["cw_binary"].isna()
        & wide["text_present"]
    ].copy()
    remaining_after_holdout = with_any_text.loc[
        ~with_any_text["video_id"].isin(holdout_video_ids)
    ].copy()
    remaining_after_holdout["model_text"] = remaining_after_holdout.apply(
        lambda row: build_model_text(row.get("title"), row.get("description"), row.get("transcript")),
        axis=1,
    )
    remaining_after_holdout["token_count"] = remaining_after_holdout["model_text"].map(token_count)
    candidate = remaining_after_holdout.loc[remaining_after_holdout["token_count"] >= 30].copy()
    summary = {
        "with_any_text_rows": int(len(with_any_text)),
        "after_holdout_rows": int(len(remaining_after_holdout)),
        "min_token_rows": int(len(candidate)),
        "accepted_rows": 0,
        "accepted_harmful_rows": 0,
        "accepted_harmless_rows": 0,
    }
    if candidate.empty:
        return (
            pd.DataFrame(
                columns=["video_id", "model_text", "pseudo_label", "pseudo_probability", "sample_weight"]
            ),
            summary,
        )

    probabilities = weak_bundle.model.predict_proba(candidate["model_text"])[:, 1]
    candidate["pseudo_probability"] = probabilities
    accepted = candidate.loc[(candidate["pseudo_probability"] >= 0.95) | (candidate["pseudo_probability"] <= 0.05)].copy()
    accepted["pseudo_label"] = (accepted["pseudo_probability"] >= 0.5).astype(int)
    accepted["sample_weight"] = 0.5
    summary.update(
        {
            "accepted_rows": int(len(accepted)),
            "accepted_harmful_rows": int((accepted["pseudo_label"] == 1).sum()),
            "accepted_harmless_rows": int((accepted["pseudo_label"] == 0).sum()),
        }
    )
    return (
        accepted[["video_id", "model_text", "pseudo_label", "pseudo_probability", "sample_weight"]],
        summary,
    )


def _bundle_from_train(
    name: str,
    train_rows: pd.DataFrame,
    validation_frame: pd.DataFrame,
    label_column: str,
    sample_weight_column: str | None = None,
) -> ModelBundle:
    sample_weight = train_rows[sample_weight_column] if sample_weight_column else None
    model = _fit_calibrated_model(
        train_rows["model_text"],
        train_rows[label_column],
        validation_frame["model_text"],
        validation_frame["de_binary"],
        sample_weight=sample_weight,
    )
    metadata = {
        "name": name,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_frame)),
    }
    return ModelBundle(name=name, model=model, metadata=metadata)


def _merge_predictions_for_report(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = predictions.copy()
    enriched["error_group"] = np.where(
        enriched["prediction"] == enriched["de_binary"].astype(int),
        "correct",
        np.where(enriched["prediction"] == 1, "false_positive", "false_negative"),
    )
    enriched["missing_transcript"] = enriched["transcript"].fillna("").str.strip().eq("")
    return enriched


def _build_threshold_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true = predictions["de_binary"].astype(int)
    probabilities = predictions["probability"].to_numpy()
    for threshold in [step / 100 for step in range(5, 100, 5)]:
        metrics = compute_metrics_at_threshold(y_true, probabilities, threshold=threshold)
        rows.append(
            {
                "model_name": str(predictions["model_name"].iloc[0]),
                "threshold": threshold,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auroc": metrics["auroc"],
                "pr_auc": metrics["pr_auc"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )
    return pd.DataFrame(rows)


def train_models() -> dict[str, Any]:
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    wide, splits = load_gold_frames()
    agreement_df = compute_agreement_metrics(wide)

    expert = wide.loc[wide["de_binary"].notna()].copy()
    expert["de_binary"] = expert["de_binary"].astype(int)
    expert = expert.merge(splits[["video_id", "split"]], on="video_id", how="inner")
    expert = _prepare_model_frame(expert, "de_binary")
    train_frame = expert.loc[expert["split"] == "train"].copy()
    validation_frame = expert.loc[expert["split"] == "validation"].copy()
    test_frame = expert.loc[expert["split"] == "test"].copy()

    holdout_video_ids = set(validation_frame["video_id"]) | set(test_frame["video_id"])

    majority_rows: list[dict[str, Any]] = []
    for split_name, frame in [("validation", validation_frame), ("test", test_frame)]:
        probs = _build_majority_predictions(train_frame["de_binary"], frame["de_binary"])
        metrics = compute_metrics(frame["de_binary"], probs)
        majority_rows.append({"model_name": "majority_baseline", "split": split_name, **metrics})

    baseline_bundle = _bundle_from_train("expert_only", train_frame, validation_frame, "de_binary")
    baseline_val_metrics, baseline_val_predictions = _predict_frame(baseline_bundle, validation_frame, "de_binary")
    baseline_test_metrics, baseline_test_predictions = _predict_frame(baseline_bundle, test_frame, "de_binary")

    reliabilities = estimate_weak_source_reliabilities(train_frame)
    weak_labels = build_weak_labels(wide, holdout_video_ids, reliabilities)

    weak_train = train_frame[["video_id", "model_text", "de_binary"]].copy()
    weak_train["sample_weight"] = 1.0
    weak_train = weak_train.rename(columns={"de_binary": "label"})

    if not weak_labels.empty:
        weak_label_rows = weak_labels.rename(columns={"weak_label": "label"})
        weak_train = pd.concat(
            [weak_train, weak_label_rows[["video_id", "model_text", "label", "sample_weight"]]],
            ignore_index=True,
        )

    weak_bundle = _bundle_from_train("expert_plus_weak", weak_train, validation_frame, "label", "sample_weight")
    weak_val_metrics, weak_val_predictions = _predict_frame(weak_bundle, validation_frame, "de_binary")
    weak_test_metrics, weak_test_predictions = _predict_frame(weak_bundle, test_frame, "de_binary")

    pseudo_labels, pseudo_label_funnel = build_pseudo_labels(wide, holdout_video_ids, weak_bundle)
    pseudo_train = weak_train.copy()
    if not pseudo_labels.empty:
        pseudo_train = pd.concat(
            [
                pseudo_train,
                pseudo_labels.rename(columns={"pseudo_label": "label"})[
                    ["video_id", "model_text", "label", "sample_weight"]
                ],
            ],
            ignore_index=True,
        )

    pseudo_bundle = _bundle_from_train("expert_plus_weak_plus_pseudo", pseudo_train, validation_frame, "label", "sample_weight")
    pseudo_val_metrics, pseudo_val_predictions = _predict_frame(pseudo_bundle, validation_frame, "de_binary")
    pseudo_test_metrics, pseudo_test_predictions = _predict_frame(pseudo_bundle, test_frame, "de_binary")

    selected_augmented = weak_bundle
    selected_augmented_name = "expert_plus_weak"
    pseudo_kept = False
    if not pseudo_labels.empty and (
        pseudo_val_metrics["macro_f1"] >= weak_val_metrics["macro_f1"]
        and pseudo_val_metrics["recall"] >= weak_val_metrics["recall"]
    ):
        selected_augmented = pseudo_bundle
        selected_augmented_name = "expert_plus_weak_plus_pseudo"
        pseudo_kept = True

    final_bundle = baseline_bundle
    final_name = "expert_only"
    if (
        (selected_augmented is weak_bundle and weak_val_metrics["macro_f1"] > baseline_val_metrics["macro_f1"])
        or (selected_augmented is pseudo_bundle and pseudo_val_metrics["macro_f1"] > baseline_val_metrics["macro_f1"])
        or (
            (
                selected_augmented is weak_bundle
                and weak_val_metrics["macro_f1"] == baseline_val_metrics["macro_f1"]
                and weak_val_metrics["recall"] >= baseline_val_metrics["recall"]
            )
            or (
                selected_augmented is pseudo_bundle
                and pseudo_val_metrics["macro_f1"] == baseline_val_metrics["macro_f1"]
                and pseudo_val_metrics["recall"] >= baseline_val_metrics["recall"]
            )
        )
    ):
        final_bundle = selected_augmented
        final_name = selected_augmented_name

    final_predictions = {
        "expert_only": baseline_test_predictions,
        "expert_plus_weak": weak_test_predictions,
        "expert_plus_weak_plus_pseudo": pseudo_test_predictions,
    }[final_name]
    final_predictions = _merge_predictions_for_report(final_predictions)
    final_predictions.to_csv(PREDICTIONS_PATH, index=False)
    weak_labels.to_csv(WEAK_LABELS_PATH, index=False)
    pseudo_labels.to_csv(PSEUDO_LABELS_PATH, index=False)

    threshold_sweep_df = pd.concat(
        [
            _build_threshold_sweep(baseline_val_predictions),
            _build_threshold_sweep(weak_val_predictions),
            _build_threshold_sweep(pseudo_val_predictions),
        ],
        ignore_index=True,
    )
    threshold_sweep_df.to_csv(THRESHOLD_SWEEP_PATH, index=False)

    consensus_test = test_frame.loc[test_frame["consensus_code"].isin(["HHH", "NNN"])].copy()
    consensus_metrics: dict[str, Any] = {}
    if not consensus_test.empty:
        for bundle in [baseline_bundle, weak_bundle, pseudo_bundle]:
            metrics, _ = _predict_frame(bundle, consensus_test, "de_binary")
            consensus_metrics[bundle.name] = metrics

    model_rows = [
        {"model_name": "majority_baseline", "split": row["split"], **row}
        for row in majority_rows
    ]
    model_rows.extend(
        [
            {"model_name": "expert_only", "split": "validation", **baseline_val_metrics},
            {"model_name": "expert_only", "split": "test", **baseline_test_metrics},
            {"model_name": "expert_plus_weak", "split": "validation", **weak_val_metrics},
            {"model_name": "expert_plus_weak", "split": "test", **weak_test_metrics},
            {"model_name": "expert_plus_weak_plus_pseudo", "split": "validation", **pseudo_val_metrics},
            {"model_name": "expert_plus_weak_plus_pseudo", "split": "test", **pseudo_test_metrics},
        ]
    )
    comparison_df = pd.DataFrame(model_rows)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    weak_label_summary = {
        "candidate_rows": int(
            wide.loc[
                wide["de_binary"].isna()
                & wide["text_present"]
                & ((wide["gpt_binary"].notna()) | (wide["cw_binary"].notna()))
                & ~wide["video_id"].isin(holdout_video_ids)
            ].shape[0]
        ),
        "accepted_rows": int(len(weak_labels)),
        "accepted_harmful_rows": int((weak_labels["weak_label"] == 1).sum()) if not weak_labels.empty else 0,
        "accepted_harmless_rows": int((weak_labels["weak_label"] == 0).sum()) if not weak_labels.empty else 0,
        "accepted_labels_path": str(WEAK_LABELS_PATH),
    }
    pseudo_label_summary = {
        "with_any_text_rows": pseudo_label_funnel["with_any_text_rows"],
        "after_holdout_rows": pseudo_label_funnel["after_holdout_rows"],
        "candidate_rows": pseudo_label_funnel["min_token_rows"],
        "accepted_rows": pseudo_label_funnel["accepted_rows"],
        "accepted_harmful_rows": pseudo_label_funnel["accepted_harmful_rows"],
        "accepted_harmless_rows": pseudo_label_funnel["accepted_harmless_rows"],
        "kept_for_final_model": pseudo_kept,
        "accepted_labels_path": str(PSEUDO_LABELS_PATH),
    }

    summary = {
        "run_at": _now_iso(),
        "agreement_metrics_path": str(AGREEMENT_METRICS_PATH),
        "model_comparison_path": str(MODEL_COMPARISON_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
        "validation_threshold_sweep_path": str(THRESHOLD_SWEEP_PATH),
        "reliabilities": reliabilities,
        "weak_label_summary": weak_label_summary,
        "pseudo_label_summary": pseudo_label_summary,
        "selected_augmented_model": selected_augmented_name,
        "final_model_name": final_name,
        "consensus_metrics": consensus_metrics,
        "validation_metrics": {
            "expert_only": baseline_val_metrics,
            "expert_plus_weak": weak_val_metrics,
            "expert_plus_weak_plus_pseudo": pseudo_val_metrics,
        },
        "test_metrics": {
            "expert_only": baseline_test_metrics,
            "expert_plus_weak": weak_test_metrics,
            "expert_plus_weak_plus_pseudo": pseudo_test_metrics,
        },
    }

    model_lookup = {
        "expert_only": baseline_bundle,
        "expert_plus_weak": weak_bundle,
        "expert_plus_weak_plus_pseudo": pseudo_bundle,
    }
    selected_bundle = model_lookup[final_name]
    selected_bundle.metadata.update(
        {
            "model_version": final_name,
            "run_at": summary["run_at"],
        }
    )
    joblib.dump(
        {
            "model": selected_bundle.model,
            "metadata": _json_safe(summary | selected_bundle.metadata),
        },
        FINAL_MODEL_PATH,
    )
    write_json(TRAINING_SUMMARY_PATH, _json_safe(summary))
    return _json_safe(summary)
