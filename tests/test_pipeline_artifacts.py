from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import ARTIFACTS_READY, LAKE_DIR, MODELS_DIR
from harm_detection.config import REQUIRED_SILVER_COLUMNS


pytestmark = pytest.mark.artifacts


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

    unlabeled = pd.read_parquet(LAKE_DIR / "bronze" / "unlabeled_large_pool.parquet", columns=["video_id"])
    unique_unlabeled = (
        unlabeled["video_id"]
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
def test_training_summary_selects_weak_model_and_documents_no_pseudo_rows() -> None:
    summary = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["final_model_name"] == "expert_plus_weak"
    assert summary["selected_augmented_model"] == "expert_plus_weak"
    assert summary["weak_label_summary"]["accepted_rows"] == 648
    assert summary["pseudo_label_summary"]["accepted_rows"] == 0
    assert summary["pseudo_label_summary"]["kept_for_final_model"] is False
    assert (
        summary["test_metrics"]["expert_plus_weak"]["macro_f1"]
        > summary["test_metrics"]["expert_only"]["macro_f1"]
    )
