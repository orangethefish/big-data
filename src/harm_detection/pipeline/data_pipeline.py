from __future__ import annotations

import csv
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import Row
from pyspark.sql import types as T

from harm_detection.config import (
    BRONZE_DIR,
    DATASET_SOURCES,
    GOLD_DIR,
    PLATFORM,
    REQUIRED_SILVER_COLUMNS,
    SILVER_DIR,
    DatasetSource,
)
from harm_detection.pipeline.spark import get_spark
from harm_detection.utils.io import ensure_dir, write_json
from harm_detection.utils.text import normalize_text, parse_harm_labels, select_canonical_text, text_present


BRONZE_MANIFEST_PATH = BRONZE_DIR / "_manifest.json"
DATA_QUALITY_REPORT_PATH = GOLD_DIR / "data_quality_summary.json"

_NORMALIZED_SCHEMA = T.StructType(
    [
        T.StructField("raw_record_id", T.StringType(), False),
        T.StructField("source_name", T.StringType(), False),
        T.StructField("source_file", T.StringType(), False),
        T.StructField("video_id", T.StringType(), True),
        T.StructField("platform", T.StringType(), False),
        T.StructField("source_group", T.StringType(), False),
        T.StructField("annotator_source", T.StringType(), False),
        T.StructField("binary_label", T.IntegerType(), True),
        T.StructField("harm_labels_raw", T.StringType(), True),
        T.StructField("harm_labels_array", T.ArrayType(T.StringType()), False),
        T.StructField("title", T.StringType(), True),
        T.StructField("description", T.StringType(), True),
        T.StructField("transcript", T.StringType(), True),
        T.StructField("published_date", T.StringType(), True),
        T.StructField("text_present", T.BooleanType(), False),
        T.StructField("ingest_issue_flags", T.ArrayType(T.StringType()), False),
    ]
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=object)


def _read_csv_with_cleanup(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    nul_bytes_removed = 0
    malformed_lines = 0
    rows: list[dict[str, str]] = []
    with path.open("rb") as binary_handle:
        for chunk in iter(lambda: binary_handle.read(1024 * 1024), b""):
            nul_bytes_removed += chunk.count(b"\x00")

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        cleaned_lines = (line.replace("\x00", "") for line in handle)
        reader = csv.reader(cleaned_lines)
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                malformed_lines += 1
                continue
            rows.append(dict(zip(header, row)))

    frame = pd.DataFrame(rows)
    return frame, {
        "nul_bytes_removed": nul_bytes_removed,
        "malformed_lines": malformed_lines,
    }


def _sanitize_pandas_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(column) for column in cleaned.columns]
    return cleaned.where(pd.notnull(cleaned), None)


def _spark_string_schema(columns: list[str]) -> T.StructType:
    return T.StructType([T.StructField(column, T.StringType(), True) for column in columns])


def _parquet_path(directory: Path, name: str) -> Path:
    ensure_dir(directory)
    return directory / f"{name}.parquet"


def _spark_validate_records(
    spark,
    rows: list[dict[str, Any]],
    schema: T.StructType,
    sample_size: int = 1_000,
) -> None:
    if not rows:
        return
    sample_rows = [Row(**row) for row in rows[:sample_size]]
    spark.createDataFrame(sample_rows, schema=schema).limit(1).collect()


def _as_string_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return [str(value)]


def _bronze_rows_from_frame(source: DatasetSource, frame: pd.DataFrame, ingested_at: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_columns = list(frame.columns)
    rows: list[dict[str, Any]] = []
    blank_video_rows = 0
    for index, payload in enumerate(frame.to_dict(orient="records"), start=1):
        raw_record_id = f"{source.name}:{index:07d}"
        raw_payload = {str(key): (None if value is None else str(value)) for key, value in payload.items()}
        video_id = normalize_text(payload.get("video_id"), zero_is_empty=False)
        if not video_id:
            blank_video_rows += 1
        row = {
            "raw_record_id": raw_record_id,
            "source_name": source.name,
            "source_file": source.path.name,
            "source_path": str(source.path),
            "source_group": source.source_group,
            "annotator_source": source.annotator_source,
            "platform": PLATFORM,
            "ingested_at": ingested_at,
            "original_columns_json": str(original_columns),
            "source_duplicate_of": source.duplicate_of,
            **raw_payload,
        }
        rows.append(row)
    manifest = {
        "source_name": source.name,
        "source_file": source.path.name,
        "row_count": len(rows),
        "blank_video_rows": blank_video_rows,
        "duplicate_of": source.duplicate_of,
        "original_columns": original_columns,
    }
    return rows, manifest


def build_bronze_layer() -> dict[str, Any]:
    ensure_dir(BRONZE_DIR)
    spark = get_spark("harm-detect-bronze")
    manifest: dict[str, Any] = {"ingested_at": _now_iso(), "sources": []}
    try:
        for source in DATASET_SOURCES:
            if source.file_type == "excel":
                frame = _read_excel(source.path)
                source_notes: dict[str, Any] = {}
            else:
                frame, source_notes = _read_csv_with_cleanup(source.path)
            frame = _sanitize_pandas_frame(frame)
            rows, summary = _bronze_rows_from_frame(source, frame, manifest["ingested_at"])
            summary.update(source_notes)
            _spark_validate_records(
                spark,
                rows,
                _spark_string_schema(list(rows[0].keys())),
            )
            bronze_path = _parquet_path(BRONZE_DIR, source.name)
            pd.DataFrame(rows).to_parquet(bronze_path, index=False)
            manifest["sources"].append(summary)
    finally:
        spark.stop()
    write_json(BRONZE_MANIFEST_PATH, manifest)
    return manifest


def _normalize_row(source: DatasetSource, row: dict[str, Any]) -> dict[str, Any] | None:
    video_id = normalize_text(row.get("video_id"), zero_is_empty=False)
    title = normalize_text(row.get("title"))
    description = normalize_text(row.get("description") or row.get("deacription"))
    transcript = normalize_text(row.get("transcript"))
    published_date = normalize_text(row.get("date"), zero_is_empty=False)
    harm_raw = normalize_text(row.get(source.harm_label_column), zero_is_empty=False) if source.harm_label_column else ""

    issue_flags: list[str] = []
    link_column = "links" if "links" in row else "link" if "link" in row else None
    if link_column == "link":
        issue_flags.append("link_column_variant")
    if "deacription" in row:
        issue_flags.append("description_column_typo")
    if source.duplicate_of:
        issue_flags.append("duplicate_source_file")
    if not video_id:
        issue_flags.append("missing_video_id")
    if not text_present(title, description, transcript):
        issue_flags.append("missing_all_text")

    normalized = {
        "raw_record_id": str(row["raw_record_id"]),
        "source_name": source.name,
        "source_file": row["source_file"],
        "video_id": video_id or None,
        "platform": PLATFORM,
        "source_group": source.source_group,
        "annotator_source": source.annotator_source,
        "binary_label": source.binary_label,
        "harm_labels_raw": harm_raw or None,
        "harm_labels_array": parse_harm_labels(harm_raw),
        "title": title or None,
        "description": description or None,
        "transcript": transcript or None,
        "published_date": published_date or None,
        "text_present": text_present(title, description, transcript),
        "ingest_issue_flags": issue_flags,
    }
    return normalized


def build_silver_layer() -> dict[str, Any]:
    ensure_dir(SILVER_DIR)
    spark = get_spark("harm-detect-silver")
    silver_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    source_level_audits: list[dict[str, Any]] = []
    try:
        duplicate_hashes: dict[str, str] = {}
        for source in DATASET_SOURCES:
            bronze_path = _parquet_path(BRONZE_DIR, source.name)
            bronze_pdf = pd.read_parquet(bronze_path).where(pd.notnull, None)
            fingerprint = hashlib.md5(
                bronze_pdf.drop(columns=[column for column in ["raw_record_id", "source_name", "source_file", "source_path", "source_group", "annotator_source", "platform", "ingested_at", "original_columns_json", "source_duplicate_of"] if column in bronze_pdf.columns])
                .to_json(orient="records", force_ascii=False)
                .encode("utf-8")
            ).hexdigest()

            if source.duplicate_of:
                if duplicate_hashes.get(source.duplicate_of) == fingerprint:
                    source_level_audits.append(
                        {
                            "source_name": source.name,
                            "event": "duplicate_source_excluded",
                            "duplicate_of": source.duplicate_of,
                            "row_count": len(bronze_pdf),
                        }
                    )
                    continue
            duplicate_hashes[source.name] = fingerprint

            for payload in bronze_pdf.to_dict(orient="records"):
                normalized = _normalize_row(source, payload)
                if normalized is None:
                    continue
                if not normalized["video_id"]:
                    dropped_rows.append(
                        {
                            "raw_record_id": normalized["raw_record_id"],
                            "source_name": normalized["source_name"],
                            "source_file": normalized["source_file"],
                            "drop_reason": "missing_video_id",
                            "ingest_issue_flags": normalized["ingest_issue_flags"],
                        }
                    )
                    continue
                silver_rows.append(normalized)
    finally:
        spark.stop()

    spark = get_spark("harm-detect-silver-validate")
    try:
        _spark_validate_records(spark, silver_rows, _NORMALIZED_SCHEMA)
    finally:
        spark.stop()

    pd.DataFrame(silver_rows).to_parquet(_parquet_path(SILVER_DIR, "records"), index=False)
    pd.DataFrame(dropped_rows).to_parquet(_parquet_path(GOLD_DIR, "audit_dropped_rows"), index=False)
    pd.DataFrame(source_level_audits).to_parquet(_parquet_path(GOLD_DIR, "audit_source_events"), index=False)

    silver_summary = {
        "silver_row_count": len(silver_rows),
        "dropped_row_count": len(dropped_rows),
        "source_events": source_level_audits,
        "required_columns": REQUIRED_SILVER_COLUMNS,
    }
    return silver_summary


def _build_canonical_text_map(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record["video_id"]), []).append(record)

    canonical: dict[str, dict[str, Any]] = {}
    duplicates_audit: list[dict[str, Any]] = []
    for video_id, rows in buckets.items():
        titles = [row.get("title") for row in rows]
        descriptions = [row.get("description") for row in rows]
        transcripts = [row.get("transcript") for row in rows]
        dates = [row.get("published_date") for row in rows]
        canonical[video_id] = {
            "platform": PLATFORM,
            "title": select_canonical_text(titles) or None,
            "description": select_canonical_text(descriptions) or None,
            "transcript": select_canonical_text(transcripts) or None,
            "published_date": select_canonical_text(dates) or None,
        }
        if len(rows) > 1:
            duplicates_audit.append(
                {
                    "video_id": video_id,
                    "record_count": len(rows),
                    "distinct_title_count": len({normalize_text(value) for value in titles if normalize_text(value)}),
                    "distinct_description_count": len({normalize_text(value) for value in descriptions if normalize_text(value)}),
                    "distinct_transcript_count": len({normalize_text(value) for value in transcripts if normalize_text(value)}),
                    "source_names": sorted({row["source_name"] for row in rows}),
                    "annotator_sources": sorted({row["annotator_source"] for row in rows}),
                }
            )
    return canonical, duplicates_audit


def _compute_conflicts(records: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    label_buckets: dict[tuple[str, str], set[int]] = {}
    for record in records:
        if record["binary_label"] is None or pd.isna(record["binary_label"]):
            continue
        key = (str(record["video_id"]), str(record["annotator_source"]))
        label_buckets.setdefault(key, set()).add(int(record["binary_label"]))
    conflicts = {key for key, labels in label_buckets.items() if len(labels) > 1}
    audit = [
        {
            "video_id": video_id,
            "annotator_source": annotator_source,
            "labels_seen": sorted(label_buckets[(video_id, annotator_source)]),
        }
        for video_id, annotator_source in sorted(conflicts)
    ]
    return conflicts, audit


def _stratified_split(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    train_ids, temp_ids = train_test_split(
        frame["video_id"],
        test_size=1 - 0.70,
        random_state=42,
        stratify=frame["de_binary"],
    )
    temp = frame.set_index("video_id").loc[temp_ids].reset_index()
    val_ids, test_ids = train_test_split(
        temp["video_id"],
        test_size=0.5,
        random_state=42,
        stratify=temp["de_binary"],
    )

    split_map = {video_id: "train" for video_id in train_ids}
    split_map.update({video_id: "validation" for video_id in val_ids})
    split_map.update({video_id: "test" for video_id in test_ids})
    result = frame.copy()
    result["split"] = result["video_id"].map(split_map)
    return result


def build_gold_layer() -> dict[str, Any]:
    ensure_dir(GOLD_DIR)
    silver_pdf = pd.read_parquet(_parquet_path(SILVER_DIR, "records")).where(pd.notnull, None)

    silver_records = silver_pdf.to_dict(orient="records")
    conflicts, conflict_audit = _compute_conflicts(silver_records)
    filtered_records = [
        record
        for record in silver_records
        if (str(record["video_id"]), str(record["annotator_source"])) not in conflicts
    ]

    canonical_text_map, duplicates_audit = _build_canonical_text_map(filtered_records)

    label_table: dict[str, dict[str, Any]] = {}
    for video_id, canonical in canonical_text_map.items():
        label_table[video_id] = {
            "video_id": video_id,
            "platform": canonical["platform"],
            "title": canonical["title"],
            "description": canonical["description"],
            "transcript": canonical["transcript"],
            "published_date": canonical["published_date"],
            "de_binary": None,
            "gpt_binary": None,
            "cw_binary": None,
            "de_harm_labels": None,
            "gpt_harm_labels": None,
            "cw_harm_labels": None,
        }

    source_name_map = {
        "domain_expert": ("de_binary", "de_harm_labels"),
        "gpt4_turbo": ("gpt_binary", "gpt_harm_labels"),
        "crowdworker": ("cw_binary", "cw_harm_labels"),
    }
    for record in filtered_records:
        mapping = source_name_map.get(str(record["annotator_source"]))
        if not mapping:
            continue
        binary_col, harm_col = mapping
        bucket = label_table[str(record["video_id"])]
        bucket[binary_col] = record["binary_label"]
        harm_labels = _as_string_list(record.get("harm_labels_array"))
        bucket[harm_col] = ",".join(harm_labels) if harm_labels else None

    wide_pdf = pd.DataFrame(label_table.values()).sort_values("video_id").reset_index(drop=True)
    wide_pdf["text_present"] = wide_pdf.apply(
        lambda row: text_present(row["title"], row["description"], row["transcript"]),
        axis=1,
    )
    wide_pdf["consensus_code"] = wide_pdf.apply(
        lambda row: (
            "HHH"
            if row["de_binary"] == 1 and row["gpt_binary"] == 1 and row["cw_binary"] == 1
            else "NNN"
            if row["de_binary"] == 0 and row["gpt_binary"] == 0 and row["cw_binary"] == 0
            else None
        ),
        axis=1,
    )

    expert_frame = wide_pdf.loc[wide_pdf["de_binary"].notna() & wide_pdf["text_present"]].copy()
    expert_frame["de_binary"] = expert_frame["de_binary"].astype(int)
    split_pdf = _stratified_split(expert_frame[["video_id", "de_binary"]])

    data_quality_summary = {
        "wide_row_count": int(len(wide_pdf)),
        "expert_labeled_row_count": int(len(expert_frame)),
        "conflict_count": int(len(conflict_audit)),
        "duplicate_video_id_count": int(len(duplicates_audit)),
        "consensus_hhh_count": int((wide_pdf["consensus_code"] == "HHH").sum()),
        "consensus_nnn_count": int((wide_pdf["consensus_code"] == "NNN").sum()),
        "source_counts": dict(Counter(silver_pdf["source_name"])),
    }

    spark = get_spark("harm-detect-gold-validate")
    try:
        if not wide_pdf.empty:
            spark.createDataFrame(wide_pdf.head(1_000)).limit(1).collect()
        if not split_pdf.empty:
            spark.createDataFrame(split_pdf.head(1_000)).limit(1).collect()
    finally:
        spark.stop()

    wide_pdf.to_parquet(_parquet_path(GOLD_DIR, "wide_labels"), index=False)
    split_pdf.to_parquet(_parquet_path(GOLD_DIR, "splits"), index=False)
    pd.DataFrame(conflict_audit).to_parquet(_parquet_path(GOLD_DIR, "audit_conflicts"), index=False)
    pd.DataFrame(duplicates_audit).to_parquet(_parquet_path(GOLD_DIR, "audit_duplicates"), index=False)

    write_json(DATA_QUALITY_REPORT_PATH, data_quality_summary)
    return data_quality_summary


def build_data_lake() -> dict[str, Any]:
    bronze = build_bronze_layer()
    silver = build_silver_layer()
    gold = build_gold_layer()
    return {"bronze": bronze, "silver": silver, "gold": gold}
