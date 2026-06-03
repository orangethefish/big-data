from __future__ import annotations

from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LAKE_DIR = ARTIFACTS_DIR / "lake"
MODELS_DIR = ARTIFACTS_DIR / "models"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "artifacts: marks tests that require the generated lake/model artifacts to exist",
    )


ARTIFACTS_READY = (
    (LAKE_DIR / "bronze" / "_manifest.json").exists()
    and (LAKE_DIR / "silver" / "records.parquet").exists()
    and (LAKE_DIR / "gold" / "wide_labels.parquet").exists()
    and (MODELS_DIR / "training_summary.json").exists()
    and (MODELS_DIR / "final_model.joblib").exists()
)
