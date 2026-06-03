from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LAKE_DIR = ARTIFACTS_DIR / "lake"
BRONZE_DIR = LAKE_DIR / "bronze"
SILVER_DIR = LAKE_DIR / "silver"
GOLD_DIR = LAKE_DIR / "gold"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

PLATFORM = "youtube"
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RISK_BANDS = {
    "low": (None, 0.35),
    "medium": (0.35, 0.65),
    "high": (0.65, None),
}

INVALID_CATEGORY_LABELS = {
    "",
    "0",
    "NO AGREEMENT",
    "NO MAJORITY",
    "UNAVAILABLE",
    "NOT APPLICABLE",
}

VALID_CATEGORY_CODES = {"IH", "HH", "CB", "ADD", "SXL", "PH"}


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: Path
    file_type: str
    source_group: str
    annotator_source: str
    binary_label: int | None
    harm_label_column: str | None
    duplicate_of: str | None = None


DATASET_SOURCES = [
    DatasetSource(
        name="agreement_full_harmful",
        path=DATASET_DIR / "Harmful _full_agreement.xlsx",
        file_type="excel",
        source_group="agreement",
        annotator_source="agreement_full",
        binary_label=1,
        harm_label_column="maj_harmcat",
    ),
    DatasetSource(
        name="agreement_subset_harmful",
        path=DATASET_DIR / "Harmful _subset_agreement.xlsx",
        file_type="excel",
        source_group="agreement",
        annotator_source="agreement_subset",
        binary_label=1,
        harm_label_column="maj_harmcat",
    ),
    DatasetSource(
        name="domain_expert_harmful",
        path=DATASET_DIR / "Domain Experts" / "Harmful.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="domain_expert",
        binary_label=1,
        harm_label_column="harm_cat",
    ),
    DatasetSource(
        name="domain_expert_harmless",
        path=DATASET_DIR / "Domain Experts" / "Harmless.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="domain_expert",
        binary_label=0,
        harm_label_column="harm_cat",
    ),
    DatasetSource(
        name="gpt4_turbo_harmful",
        path=DATASET_DIR / "GPT-4-Turbo" / "Harmful.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="gpt4_turbo",
        binary_label=1,
        harm_label_column="harm_cat",
    ),
    DatasetSource(
        name="gpt4_turbo_harmless",
        path=DATASET_DIR / "GPT-4-Turbo" / "Harmless.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="gpt4_turbo",
        binary_label=0,
        harm_label_column="harm_cat",
    ),
    DatasetSource(
        name="crowdworker_harmful",
        path=DATASET_DIR / "Crowdworker" / "Harmful.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="crowdworker",
        binary_label=1,
        harm_label_column="harm_cat",
    ),
    DatasetSource(
        name="crowdworker_harmless",
        path=DATASET_DIR / "Crowdworker" / "Harmless.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="crowdworker",
        binary_label=0,
        harm_label_column="harm_cat",
    ),
    DatasetSource(
        name="crowdworker_harmless_duplicate",
        path=DATASET_DIR / "Crowdworker" / "HarmlessM.xlsx",
        file_type="excel",
        source_group="annotated",
        annotator_source="crowdworker",
        binary_label=0,
        harm_label_column="harm_cat",
        duplicate_of="crowdworker_harmless",
    ),
    DatasetSource(
        name="unlabeled_large_pool",
        path=DATASET_DIR / "Unannotated_large_pool.csv",
        file_type="csv",
        source_group="unlabeled_pool",
        annotator_source="unlabeled_pool",
        binary_label=None,
        harm_label_column=None,
    ),
]


REQUIRED_SILVER_COLUMNS = [
    "video_id",
    "platform",
    "source_group",
    "annotator_source",
    "binary_label",
    "harm_labels_raw",
    "harm_labels_array",
    "title",
    "description",
    "transcript",
    "published_date",
    "text_present",
    "ingest_issue_flags",
]

