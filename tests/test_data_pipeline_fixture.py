from __future__ import annotations

from pathlib import Path

import pandas as pd

from harm_detection.config import DatasetSource
from harm_detection.modeling import training
from harm_detection.pipeline import data_pipeline


def _write_excel(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, index=False)


def _long_text(prefix: str, tokens: int = 40) -> str:
    return " ".join(f"{prefix}_{index}" for index in range(tokens))


def _patch_runtime_paths(monkeypatch, artifacts_dir: Path) -> None:
    lake_dir = artifacts_dir / "lake"
    bronze_dir = lake_dir / "bronze"
    silver_dir = lake_dir / "silver"
    gold_dir = lake_dir / "gold"
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"

    monkeypatch.setattr(data_pipeline, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(data_pipeline, "SILVER_DIR", silver_dir)
    monkeypatch.setattr(data_pipeline, "GOLD_DIR", gold_dir)
    monkeypatch.setattr(data_pipeline, "BRONZE_MANIFEST_PATH", bronze_dir / "_manifest.json")
    monkeypatch.setattr(data_pipeline, "DATA_QUALITY_REPORT_PATH", gold_dir / "data_quality_summary.json")

    monkeypatch.setattr(training, "GOLD_DIR", gold_dir)
    monkeypatch.setattr(training, "MODELS_DIR", models_dir)
    monkeypatch.setattr(training, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(training, "AGREEMENT_METRICS_PATH", reports_dir / "agreement_metrics.csv")
    monkeypatch.setattr(training, "MODEL_COMPARISON_PATH", reports_dir / "model_comparison.csv")
    monkeypatch.setattr(training, "PREDICTIONS_PATH", reports_dir / "final_test_predictions.csv")
    monkeypatch.setattr(training, "WEAK_LABELS_PATH", reports_dir / "accepted_weak_labels.csv")
    monkeypatch.setattr(training, "PSEUDO_LABELS_PATH", reports_dir / "accepted_pseudo_labels.csv")
    monkeypatch.setattr(training, "THRESHOLD_SWEEP_PATH", reports_dir / "validation_threshold_sweep.csv")
    monkeypatch.setattr(training, "TRAINING_SUMMARY_PATH", models_dir / "training_summary.json")
    monkeypatch.setattr(training, "FINAL_MODEL_PATH", models_dir / "final_model.joblib")


def _build_fixture_sources(dataset_dir: Path) -> list[DatasetSource]:
    return [
        DatasetSource(
            name="domain_expert_harmful",
            path=dataset_dir / "Domain Experts" / "Harmful.xlsx",
            file_type="excel",
            source_group="annotated",
            annotator_source="domain_expert",
            binary_label=1,
            harm_label_column="harm_cat",
        ),
        DatasetSource(
            name="domain_expert_harmless",
            path=dataset_dir / "Domain Experts" / "Harmless.xlsx",
            file_type="excel",
            source_group="annotated",
            annotator_source="domain_expert",
            binary_label=0,
            harm_label_column="harm_cat",
        ),
        DatasetSource(
            name="gpt4_turbo_harmful",
            path=dataset_dir / "GPT-4-Turbo" / "Harmful.xlsx",
            file_type="excel",
            source_group="annotated",
            annotator_source="gpt4_turbo",
            binary_label=1,
            harm_label_column="harm_cat",
        ),
        DatasetSource(
            name="gpt4_turbo_harmless",
            path=dataset_dir / "GPT-4-Turbo" / "Harmless.xlsx",
            file_type="excel",
            source_group="annotated",
            annotator_source="gpt4_turbo",
            binary_label=0,
            harm_label_column="harm_cat",
        ),
        DatasetSource(
            name="crowdworker_harmful",
            path=dataset_dir / "Crowdworker" / "Harmful.xlsx",
            file_type="excel",
            source_group="annotated",
            annotator_source="crowdworker",
            binary_label=1,
            harm_label_column="harm_cat",
        ),
        DatasetSource(
            name="crowdworker_harmless",
            path=dataset_dir / "Crowdworker" / "Harmless.xlsx",
            file_type="excel",
            source_group="annotated",
            annotator_source="crowdworker",
            binary_label=0,
            harm_label_column="harm_cat",
        ),
        DatasetSource(
            name="unlabeled_large_pool",
            path=dataset_dir / "Unannotated_large_pool.csv",
            file_type="csv",
            source_group="unlabeled_pool",
            annotator_source="unlabeled_pool",
            binary_label=None,
            harm_label_column=None,
        ),
    ]


def test_normalize_row_uses_uppercase_aliases_and_typo_cleanup() -> None:
    source = DatasetSource(
        name="unlabeled_large_pool",
        path=Path("Unannotated_large_pool.csv"),
        file_type="csv",
        source_group="unlabeled_pool",
        annotator_source="unlabeled_pool",
        binary_label=None,
        harm_label_column=None,
    )
    row = {
        "raw_record_id": "unlabeled_large_pool:0000001",
        "source_file": "Unannotated_large_pool.csv",
        "video_id": "abc123",
        "Title": "  Alias Title  ",
        "deacription": " typo description ",
        "Transcript": " transcript words ",
        "Date": "2024-01-01",
        "link": "https://example.com/watch?v=abc123",
    }

    normalized = data_pipeline._normalize_row(source, row)

    assert normalized["title"] == "Alias Title"
    assert normalized["description"] == "typo description"
    assert normalized["transcript"] == "transcript words"
    assert normalized["published_date"] == "2024-01-01"
    assert normalized["text_present"] is True
    assert "title_column_alias:Title" in normalized["ingest_issue_flags"]
    assert "description_column_typo" in normalized["ingest_issue_flags"]
    assert "transcript_column_alias:Transcript" in normalized["ingest_issue_flags"]
    assert "published_date_column_alias:Date" in normalized["ingest_issue_flags"]


def test_fixture_pipeline_preserves_unlabeled_text_and_tracks_augmentation(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    artifacts_dir = tmp_path / "artifacts"

    harmful_rows = [
        {
            "video_id": f"H{index}",
            "title": f"harmful title {index}",
            "description": f"harmful description {index}",
            "transcript": "short harmful transcript" if index == 0 else _long_text(f"harmful_{index}", 35),
            "date": f"2024-01-{index + 1:02d}",
            "harm_cat": "HH",
        }
        for index in range(10)
    ]
    harmless_rows = [
        {
            "video_id": f"N{index}",
            "title": f"harmless title {index}",
            "description": f"harmless description {index}",
            "transcript": _long_text(f"harmless_{index}", 35),
            "date": f"2024-02-{index + 1:02d}",
            "harm_cat": "0",
        }
        for index in range(10)
    ]
    weak_harmful_row = {
        "video_id": "W1",
        "title": "weak harmful title",
        "description": "weak harmful description",
        "transcript": _long_text("weak_harmful", 35),
        "date": "2024-03-01",
        "harm_cat": "HH",
    }

    _write_excel(dataset_dir / "Domain Experts" / "Harmful.xlsx", harmful_rows)
    _write_excel(dataset_dir / "Domain Experts" / "Harmless.xlsx", harmless_rows)
    _write_excel(dataset_dir / "GPT-4-Turbo" / "Harmful.xlsx", harmful_rows + [weak_harmful_row])
    _write_excel(dataset_dir / "GPT-4-Turbo" / "Harmless.xlsx", harmless_rows)
    _write_excel(dataset_dir / "Crowdworker" / "Harmful.xlsx", harmful_rows + [weak_harmful_row])
    _write_excel(dataset_dir / "Crowdworker" / "Harmless.xlsx", harmless_rows)

    unlabeled_rows = [
        {
            "link": "https://example.com/watch?v=H0",
            "video_id": "H0",
            "Channel Name": "channel-h0",
            "Title": "harmful title 0",
            "Description": "harmful description 0",
            "Transcript": _long_text("canonical_unlabeled_h0", 50),
            "Date": "2024-01-01",
            "Duration": "",
            "Views": "",
        },
        {
            "link": "https://example.com/watch?v=U0",
            "video_id": "U0",
            "Channel Name": "channel-u0",
            "Title": "unlabeled title 0",
            "Description": "unlabeled description 0",
            "Transcript": _long_text("pseudo_candidate_u0", 45),
            "Date": "2024-04-01",
            "Duration": "",
            "Views": "",
        },
        {
            "link": "https://example.com/watch?v=U1",
            "video_id": "U1",
            "Channel Name": "channel-u1",
            "Title": "unlabeled title 1",
            "Description": "unlabeled description 1",
            "Transcript": _long_text("pseudo_candidate_u1", 45),
            "Date": "2024-04-02",
            "Duration": "",
            "Views": "",
        },
    ]
    pd.DataFrame(unlabeled_rows).to_csv(dataset_dir / "Unannotated_large_pool.csv", index=False)

    _patch_runtime_paths(monkeypatch, artifacts_dir)
    monkeypatch.setattr(data_pipeline, "DATASET_SOURCES", _build_fixture_sources(dataset_dir))

    lake_summary = data_pipeline.build_data_lake()
    training_summary = training.train_models()

    silver = pd.read_parquet(artifacts_dir / "lake" / "silver" / "records.parquet")
    unlabeled_silver = silver.loc[silver["source_name"] == "unlabeled_large_pool"]
    assert unlabeled_silver["text_present"].all()

    wide = pd.read_parquet(artifacts_dir / "lake" / "gold" / "wide_labels.parquet").set_index("video_id")
    assert wide.loc["H0", "transcript"] == _long_text("canonical_unlabeled_h0", 50)

    canonical_audit = pd.read_parquet(artifacts_dir / "lake" / "gold" / "audit_canonical_merges.parquet")
    h0_audit = canonical_audit.loc[canonical_audit["video_id"] == "H0"].iloc[0]
    assert h0_audit["canonical_transcript_source"] == "unlabeled_large_pool"

    weak_labels = pd.read_csv(artifacts_dir / "reports" / "accepted_weak_labels.csv")
    assert "W1" in set(weak_labels["video_id"])

    pseudo_summary = training_summary["pseudo_label_summary"]
    assert pseudo_summary["with_any_text_rows"] >= 2
    assert pseudo_summary["candidate_rows"] >= 2

    assert lake_summary["gold"]["source_text_coverage"]["unlabeled_large_pool"]["text_present_count"] == len(
        unlabeled_rows
    )
