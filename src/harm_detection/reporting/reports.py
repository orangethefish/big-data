from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from harm_detection.config import BRONZE_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR
from harm_detection.utils.io import ensure_dir, read_json


def _write_markdown(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_block(frame: pd.DataFrame) -> str:
    return "\n".join(["```text", frame.to_string(index=False), "```"])


def generate_reports() -> dict[str, str]:
    ensure_dir(REPORTS_DIR)
    training_summary = read_json(MODELS_DIR / "training_summary.json")
    data_quality = read_json(GOLD_DIR / "data_quality_summary.json")
    bronze_manifest = read_json(BRONZE_DIR / "_manifest.json")
    agreement_df = pd.read_csv(REPORTS_DIR / "agreement_metrics.csv")
    comparison_df = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    predictions_df = pd.read_csv(REPORTS_DIR / "final_test_predictions.csv")
    unlabeled_unique = (
        pd.read_parquet(BRONZE_DIR / "unlabeled_large_pool.parquet", columns=["video_id"])["video_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
    reference_counts = [
        ("full_agreement_unique_ids", 5901, data_quality["consensus_hhh_count"]),
        ("subset_agreement_unique_ids", 13981, data_quality["source_counts"]["agreement_subset_harmful"]),
        ("domain_expert_harmful_unique_ids", 15058, data_quality["source_counts"]["domain_expert_harmful"]),
        ("domain_expert_harmless_unique_ids", 3296, data_quality["source_counts"]["domain_expert_harmless"]),
        ("gpt_harmful_unique_ids", 10494, data_quality["source_counts"]["gpt4_turbo_harmful"]),
        ("gpt_harmless_unique_ids", 7796, data_quality["source_counts"]["gpt4_turbo_harmless"]),
        ("crowdworker_harmful_unique_ids", 12623, data_quality["source_counts"]["crowdworker_harmful"]),
        ("crowdworker_harmless_unique_ids", 4376, data_quality["source_counts"]["crowdworker_harmless"]),
        ("unlabeled_unique_ids_after_dedup", 59925, int(unlabeled_unique)),
    ]
    reference_df = pd.DataFrame(reference_counts, columns=["reference_metric", "plan_value", "local_value"])
    reference_df["delta"] = reference_df["local_value"] - reference_df["plan_value"]

    sns.set_theme(style="whitegrid")

    agreement_plot_path = REPORTS_DIR / "agreement_kappa.png"
    agreement_df = agreement_df.copy()
    agreement_df["pair"] = agreement_df["left_source"] + " vs " + agreement_df["right_source"]
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=agreement_df, x="cohen_kappa", y="pair", hue="pair", palette="crest", legend=False)
    plt.xlim(0, 1)
    plt.title("Pairwise Cohen's Kappa")
    plt.xlabel("Kappa")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(agreement_plot_path, dpi=160)
    plt.close()

    metric_plot_path = REPORTS_DIR / "model_macro_f1.png"
    macro_df = comparison_df.loc[comparison_df["model_name"] != "majority_baseline", ["model_name", "split", "macro_f1"]]
    plt.figure(figsize=(8.5, 4.5))
    sns.barplot(data=macro_df, x="macro_f1", y="model_name", hue="split", palette="mako")
    plt.title("Macro-F1 by Model")
    plt.xlabel("Macro-F1")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=160)
    plt.close()

    examples: list[str] = []
    correct = predictions_df.loc[(predictions_df["error_group"] == "correct") & ((predictions_df["probability"] >= 0.9) | (predictions_df["probability"] <= 0.1))]
    false_positive = predictions_df.loc[predictions_df["error_group"] == "false_positive"]
    false_negative = predictions_df.loc[predictions_df["error_group"] == "false_negative"]
    missing_transcript = predictions_df.loc[predictions_df["missing_transcript"]]

    for label, frame in [
        ("Correct High-Confidence Prediction", correct),
        ("False Positive", false_positive),
        ("False Negative", false_negative),
        ("Missing-Transcript Case", missing_transcript),
    ]:
        if frame.empty:
            examples.append(f"### {label}\nNo example was available in the final test predictions.\n")
            continue
        row = frame.sort_values("probability", ascending=False).iloc[0]
        examples.append(
            "\n".join(
                [
                    f"### {label}",
                    f"- `video_id`: `{row['video_id']}`",
                    f"- probability: `{row['probability']:.4f}`",
                    f"- predicted label: `{int(row['prediction'])}`",
                    f"- true label: `{int(row['de_binary'])}`",
                    f"- title: {row['title']}",
                ]
            )
        )

    agreement_lines = [
        "# Annotator Agreement Report",
        "",
        "All agreement metrics are computed on overlapping rows in the canonical wide table.",
        "",
        _frame_block(agreement_df.drop(columns=["pair"])),
        "",
        f"![Agreement Kappa]({agreement_plot_path.as_posix()})",
    ]
    _write_markdown(REPORTS_DIR / "annotator_agreement_report.md", agreement_lines)

    data_quality_lines = [
        "# Data Quality Report",
        "",
        f"- Wide table rows: `{data_quality['wide_row_count']}`",
        f"- Expert labeled rows with text: `{data_quality['expert_labeled_row_count']}`",
        f"- Conflict rows excluded: `{data_quality['conflict_count']}`",
        f"- Duplicate video IDs collapsed: `{data_quality['duplicate_video_id_count']}`",
        f"- Strict HHH rows: `{data_quality['consensus_hhh_count']}`",
        f"- Strict NNN rows: `{data_quality['consensus_nnn_count']}`",
        f"- Bronze sources ingested: `{len(bronze_manifest['sources'])}`",
        "",
        "Source counts after silver normalization:",
        "",
        _frame_block(
            pd.DataFrame(
                sorted(data_quality["source_counts"].items()),
                columns=["source_name", "row_count"],
            )
        ),
        "",
        "Reference-count drift between the plan and the local authoritative files:",
        "",
        _frame_block(reference_df),
    ]
    _write_markdown(REPORTS_DIR / "data_quality_report.md", data_quality_lines)

    model_lines = [
        "# Model Comparison Report",
        "",
        f"- Final model: `{training_summary['final_model_name']}`",
        f"- Selected augmented model: `{training_summary['selected_augmented_model']}`",
        "",
        _frame_block(comparison_df),
        "",
        f"![Macro F1]({metric_plot_path.as_posix()})",
    ]
    _write_markdown(REPORTS_DIR / "model_comparison_report.md", model_lines)

    pseudo = training_summary["pseudo_label_summary"]
    pseudo_lines = [
        "# Pseudo-Label Acceptance Summary",
        "",
        f"- Candidate rows: `{pseudo['candidate_rows']}`",
        f"- Accepted rows: `{pseudo['accepted_rows']}`",
        f"- Harmful pseudo-labels: `{pseudo['accepted_harmful_rows']}`",
        f"- Harmless pseudo-labels: `{pseudo['accepted_harmless_rows']}`",
        f"- Kept in final model: `{pseudo['kept_for_final_model']}`",
    ]
    _write_markdown(REPORTS_DIR / "pseudo_label_summary.md", pseudo_lines)

    error_lines = [
        "# Error Analysis",
        "",
        "The examples below are pulled from the final test predictions table.",
        "",
        *examples,
    ]
    _write_markdown(REPORTS_DIR / "error_analysis.md", error_lines)

    return {
        "agreement_report": str(REPORTS_DIR / "annotator_agreement_report.md"),
        "data_quality_report": str(REPORTS_DIR / "data_quality_report.md"),
        "model_report": str(REPORTS_DIR / "model_comparison_report.md"),
        "pseudo_label_report": str(REPORTS_DIR / "pseudo_label_summary.md"),
        "error_analysis_report": str(REPORTS_DIR / "error_analysis.md"),
    }
