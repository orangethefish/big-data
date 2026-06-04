from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import ARTIFACTS_READY, LAKE_DIR, MODELS_DIR
from harm_detection.config import REQUIRED_SILVER_COLUMNS


pytestmark = pytest.mark.artifacts


def _expected_final_model(summary: dict[str, object]) -> tuple[str, str, bool]:
    validation_metrics = summary["validation_metrics"]
    baseline_val = validation_metrics["expert_only"]
    weak_val = validation_metrics["expert_plus_weak"]
    pseudo_val = validation_metrics["expert_plus_weak_plus_pseudo"]
    pseudo_summary = summary["pseudo_label_summary"]

    selected_augmented_name = "expert_plus_weak"
    pseudo_kept = False
    if pseudo_summary["accepted_rows"] > 0 and (
        pseudo_val["macro_f1"] >= weak_val["macro_f1"] and pseudo_val["recall"] >= weak_val["recall"]
    ):
        selected_augmented_name = "expert_plus_weak_plus_pseudo"
        pseudo_kept = True

    selected_val = weak_val if selected_augmented_name == "expert_plus_weak" else pseudo_val
    final_name = "expert_only"
    if (
        selected_val["macro_f1"] > baseline_val["macro_f1"]
        or (
            selected_val["macro_f1"] == baseline_val["macro_f1"]
            and selected_val["recall"] >= baseline_val["recall"]
        )
    ):
        final_name = selected_augmented_name
    return final_name, selected_augmented_name, pseudo_kept


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_bronze_manifest_matches_local_source_profile() -> None:
    manifest = json.loads((LAKE_DIR / "bronze" / "_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "agreement_full_harmful": {"row_count": 5901, "blank_video_rows": 0},
        "agreement_subset_harmful": {"row_count": 14019, "blank_video_rows": 39},
        "domain_expert_harmful": {"row_count": 15114, "blank_video_rows": 57},
        "domain_expert_harmless": {"row_count": 3302, "blank_video_rows": 8},
        "gpt4_turbo_harmful": {"row_count": 10494, "blank_video_rows": 0},
        "gpt4_turbo_harmless": {"row_count": 7818, "blank_video_rows": 24},
        "crowdworker_harmful": {"row_count": 12668, "blank_video_rows": 46},
        "crowdworker_harmless": {"row_count": 4390, "blank_video_rows": 16},
        "crowdworker_harmless_duplicate": {"row_count": 4390, "blank_video_rows": 16},
        "unlabeled_large_pool": {"row_count": 60905, "blank_video_rows": 1},
    }
    observed = {item["source_name"]: item for item in manifest["sources"]}
    for source_name, counts in expected.items():
        assert observed[source_name]["row_count"] == counts["row_count"]
        assert observed[source_name]["blank_video_rows"] == counts["blank_video_rows"]

    unlabeled = observed["unlabeled_large_pool"]
    assert unlabeled["nul_bytes_removed"] == 816
    assert unlabeled["malformed_lines"] == 1783


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_silver_contract_and_unlabeled_dedup_audit() -> None:
    silver = pd.read_parquet(LAKE_DIR / "silver" / "records.parquet")
    assert set(REQUIRED_SILVER_COLUMNS).issubset(silver.columns)
    assert silver["video_id"].isna().sum() == 0
    assert "crowdworker_harmless_duplicate" not in set(silver["source_name"])
    unlabeled = silver.loc[silver["source_name"] == "unlabeled_large_pool"]
    assert int(unlabeled["text_present"].sum()) > 0

    bronze_unlabeled = pd.read_parquet(LAKE_DIR / "bronze" / "unlabeled_large_pool.parquet", columns=["video_id"])
    unique_unlabeled = (
        bronze_unlabeled["video_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
    assert unique_unlabeled == 59671


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_gold_splits_are_disjoint_and_fixed() -> None:
    splits = pd.read_parquet(LAKE_DIR / "gold" / "splits.parquet")
    train_ids = set(splits.loc[splits["split"] == "train", "video_id"])
    validation_ids = set(splits.loc[splits["split"] == "validation", "video_id"])
    test_ids = set(splits.loc[splits["split"] == "test", "video_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert len(splits) == 18349
    assert splits["split"].value_counts().to_dict() == {
        "train": 12844,
        "test": 2753,
        "validation": 2752,
    }


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_training_summary_follows_selection_rules_and_records_pseudo_funnel() -> None:
    summary = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    expected_final, expected_augmented, expected_pseudo_kept = _expected_final_model(summary)
    assert summary["final_model_name"] == expected_final
    assert summary["selected_augmented_model"] == expected_augmented
    assert summary["pseudo_label_summary"]["kept_for_final_model"] is expected_pseudo_kept
    assert summary["weak_label_summary"]["accepted_rows"] > 0

    pseudo = summary["pseudo_label_summary"]
    assert pseudo["with_any_text_rows"] >= pseudo["after_holdout_rows"] >= pseudo["candidate_rows"] >= pseudo["accepted_rows"]
    if pseudo["kept_for_final_model"]:
        assert pseudo["accepted_rows"] > 0
    else:
        assert pseudo["accepted_rows"] >= 0


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_accepted_weak_and_pseudo_labels_do_not_leak_into_holdouts() -> None:
    splits = pd.read_parquet(LAKE_DIR / "gold" / "splits.parquet")
    holdout_ids = set(splits.loc[splits["split"].isin(["validation", "test"]), "video_id"])
    summary = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))

    weak_labels = pd.read_csv(summary["weak_label_summary"]["accepted_labels_path"])
    assert set(weak_labels["video_id"]).isdisjoint(holdout_ids)

    pseudo_labels = pd.read_csv(summary["pseudo_label_summary"]["accepted_labels_path"])
    if not pseudo_labels.empty:
        assert set(pseudo_labels["video_id"]).isdisjoint(holdout_ids)


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_data_quality_summary_includes_source_text_coverage() -> None:
    summary = json.loads((LAKE_DIR / "gold" / "data_quality_summary.json").read_text(encoding="utf-8"))
    unlabeled_coverage = summary["source_text_coverage"]["unlabeled_large_pool"]
    assert unlabeled_coverage["text_present_count"] > 0
    assert unlabeled_coverage["title_present_count"] > 0
    assert unlabeled_coverage["description_present_count"] > 0
    assert unlabeled_coverage["transcript_present_count"] > 0
    assert summary["wide_text_coverage"]["text_present_count"] > 0


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_validation_threshold_sweep_artifact_is_present_and_complete() -> None:
    summary = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    threshold_sweep = pd.read_csv(summary["validation_threshold_sweep_path"])
    assert set(threshold_sweep["model_name"]) == {
        "expert_only",
        "expert_plus_weak",
        "expert_plus_weak_plus_pseudo",
    }
    assert set(threshold_sweep["threshold"]) == {step / 100 for step in range(5, 100, 5)}
    assert threshold_sweep["macro_f1"].between(0.0, 1.0).all()
    assert threshold_sweep["precision"].between(0.0, 1.0).all()
    assert threshold_sweep["recall"].between(0.0, 1.0).all()


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_canonical_merge_audit_records_selected_sources() -> None:
    audit = pd.read_parquet(LAKE_DIR / "gold" / "audit_canonical_merges.parquet")
    assert not audit.empty
    expected_columns = {
        "video_id",
        "canonical_title_source",
        "canonical_description_source",
        "canonical_transcript_source",
        "canonical_published_date_source",
    }
    assert expected_columns.issubset(set(audit.columns))
    assert audit["record_count"].ge(2).all()
