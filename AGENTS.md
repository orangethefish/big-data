# Generalizable Harmful Video Detection for Streaming Platforms Using Weak Supervision and Pseudo-Labeling

## Summary
- Rename the project to **Generalizable Harmful Video Detection for Streaming Platforms Using Weak Supervision and Pseudo-Labeling**.
- Frame it carefully: the system targets streaming platforms in general, but all experiments are evaluated on **YouTube-derived MetaHarm data**, so claims are about **generalizable pipeline design**, not proven cross-platform performance.
- Build an **engineering-heavy, local single-node Spark pipeline** with a **batch training/scoring workflow** and a **small demo API**.
- Make **binary harmful vs harmless detection** the core deliverable. Keep **6-category harm classification** as a phase-2 extension after the binary system is stable.
- Keep version 1 **text-first**. Exclude thumbnails and image frames from the core scope because they are external, increase setup risk, and are not needed for a strong Big Data project.

## Key Changes, Interfaces, and Implementation
- **Technology stack**: Python 3.10+, PySpark in local mode, pandas/openpyxl for Excel ingestion, standard CSV readers with NUL-byte cleanup, Parquet for storage, scikit-learn for modeling, FastAPI for the demo service, and Matplotlib/Seaborn for reporting visuals.
- **Storage pattern**: use a 3-layer `bronze/silver/gold` layout in Parquet.
- **Bronze layer**: ingest each raw spreadsheet/CSV as-is, preserve source file name, actor name, original column names, ingestion timestamp, and a `raw_record_id`.
- **Silver layer**: normalize all sources into one schema: `video_id`, `platform`, `source_group`, `annotator_source`, `binary_label`, `harm_labels_raw`, `harm_labels_array`, `title`, `description`, `transcript`, `published_date`, `text_present`, `ingest_issue_flags`.
- **Wide label table**: build one row per `video_id` with `de_binary`, `gpt_binary`, `cw_binary`, `de_harm_labels`, `gpt_harm_labels`, `cw_harm_labels`, plus canonical text fields.
- **Canonical text rule**: for duplicate rows of the same `video_id`, keep the longest non-empty value per field after trimming whitespace; preserve an audit table of duplicates and conflicts.
- **Conflict rule**: if the same actor marks the same `video_id` in both harmful and harmless files, exclude it from training and keep it in an `audit_conflicts` table.
- **Source-specific cleanup**: fix `link` vs `links`, `description` vs `deacription`, padded empty Excel rows, embedded NUL bytes in the unlabeled CSV, and duplicate crowd harmless files.
- **Model input contract**: the core model uses only `title`, `description`, and `transcript`. Do not use `channel`, `views`, or `duration` in the core classifier because they either create creator leakage or do not exist consistently in labeled training data.
- **Training split policy**: split only on unique `video_id`, stratified by binary label, with `70/15/15` train/validation/test using **domain expert labels** as the primary ground truth.
- **Leakage rule**: no weak-labeled or pseudo-labeled record whose `video_id` is in the expert validation or test split may be added back into training.
- **Primary binary target**: use the domain expert files as the main supervised label source, with harmful = 1 and harmless = 0.
- **Consensus benchmark**: also create a strict benchmark using `HHH` as harmful and `NNN` as harmless for a secondary robustness evaluation.
- **Weak supervision design**: treat GPT-4-Turbo and crowdworker labels as weak sources. Convert their binary decisions to a weighted weak label only for records outside the expert validation/test folds.
- **Weak-label formula**: estimate weak-source reliability on the expert training overlap, normalize those reliabilities into `w_gpt` and `w_crowd`, compute `weak_p = w_gpt * gpt_label + w_crowd * crowd_label` over available sources, assign harmful if `weak_p >= 0.75`, harmless if `weak_p <= 0.25`, otherwise abstain.
- **Weak-sample weight**: set training sample weight to `0.5 + abs(weak_p - 0.5)` for accepted weak labels.
- **Baseline model**: TF-IDF over concatenated text plus a logistic regression classifier trained on expert labels only.
- **Model comparison set**: train and report three versions: expert-only baseline, expert + weak supervision, and expert + weak supervision + pseudo-labeling.
- **Pseudo-labeling design**: score the cleaned unlabeled pool once using the best weak-supervision model, then keep only high-confidence predictions with calibrated harmful probability `>= 0.95` or `<= 0.05`, with at least 30 tokens of total text and at least one non-empty text field.
- **Pseudo-label weight**: assign all accepted pseudo-labels a fixed sample weight of `0.5`, retrain once, and stop after one self-training round.
- **Fallback rule**: if pseudo-labeling reduces validation macro-F1 or harmful-class recall, keep the weak-supervision model as the final binary model and report pseudo-labeling as a negative result.
- **Phase-2 extension**: train six one-vs-rest category models for `IH`, `HH`, `CB`, `ADD`, `SXL`, and `PH` using expert harmful rows with valid category labels; exclude rows whose harm category is `No agreement`, `No majority`, `Unavailable`, `Not applicable`, or `0`.
- **Demo API**: expose a FastAPI `POST /predict` endpoint taking `title`, `description`, and `transcript`, returning `is_harmful`, `harmful_probability`, `risk_band`, and `model_version`.
- **Risk band rule**: map probability to `low < 0.35`, `medium 0.35–0.65`, and `high > 0.65`.
- **Reporting outputs**: generate a data quality report, annotator agreement report, model comparison table, pseudo-label acceptance summary, and an error-analysis section grouped by false positives, false negatives, and missing-transcript cases.

## Targets and Deliverables
- **Data engineering target**: produce reproducible bronze/silver/gold Parquet datasets and an audit log that explains every dropped, merged, duplicated, or conflicted record.
- **Agreement-analysis target**: report pairwise percent agreement and Cohen’s kappa for Domain Experts vs GPT, Domain Experts vs Crowd, and GPT vs Crowd.
- **Core modeling target**: beat a majority-class baseline by a clear margin and optimize for **macro-F1** and **harmful-class recall**, not raw accuracy.
- **Required binary metrics**: macro-F1, weighted F1, precision, recall, AUROC, PR-AUC, and confusion matrix on the expert test split.
- **Secondary benchmark target**: report the same binary metrics on the strict `HHH/NNN` consensus subset.
- **Pseudo-label target**: show whether unlabeled-pool augmentation improves recall or macro-F1 without causing a large precision collapse.
- **Demo target**: local API prediction for short text inputs in under 1 second after model load.
- **Report target**: final repo includes code, cleaned data contracts, experiment tables, plots, methodology write-up, limitations, and a presentation-ready result summary.
- **Execution phases**: Phase 1 ingestion and normalization, Phase 2 agreement analysis and baseline model, Phase 3 weak supervision, Phase 4 pseudo-labeling, Phase 5 API/demo, Phase 6 report and slides.

## Test Plan
- Verify ingestion row counts against the local source truth: `5901` full-agreement unique IDs, `13981` subset-agreement unique IDs, `15058/3296` domain-expert harmful/harmless unique IDs, `10494/7796` GPT harmful/harmless unique IDs, `12623/4376` crowd harmful/harmless unique IDs, and `59925` unique unlabeled IDs after deduplication.
- Verify that the pipeline catches and logs malformed CSV lines, NUL-byte cleanup, padded blank Excel rows, and duplicate crowd harmless files.
- Verify schema normalization so every silver record has the same column contract regardless of source file.
- Verify duplicate handling by asserting one canonical row per `video_id` in the wide training table and a non-empty audit record for every collapsed duplicate.
- Verify no train/validation/test leakage by asserting disjoint `video_id` sets after all joins, weak-label augmentation, and pseudo-label augmentation.
- Verify that expert-only, weak-supervision, and pseudo-label models are trained on the same validation/test split so comparisons are fair.
- Verify pseudo-label thresholds using validation-set calibration before unlabeled data is accepted.
- Verify the API contract with empty-field rejection, valid prediction response shape, and deterministic output for repeated identical inputs.
- Verify the report includes one example each of correct high-confidence prediction, false positive, false negative, and a failure case caused by missing or noisy transcript text.

## Implementation Update (2026-06-03)
- The repository now contains a working local implementation under `src/harm_detection/` with CLI entrypoints in `src/harm_detection/cli.py`.
- The implemented modules cover bronze/silver/gold ingestion, canonical text consolidation, audit tables, expert train/validation/test splits, agreement analysis, baseline plus weak-supervision training, report generation, and a FastAPI demo service.
- The demo API is implemented at `src/harm_detection/api/app.py` with `POST /predict` returning `is_harmful`, `harmful_probability`, `risk_band`, and `model_version`.
- The current model bundle is written to `artifacts/models/final_model.joblib`, and generated reports are written to `artifacts/reports/`.
- Automated verification is in place under `tests/`, and the latest full test run passed with `9 passed`.

## Current Results (2026-06-03)
- Bronze, silver, and gold artifacts are being produced successfully in `artifacts/lake/`.
- Current lake summary:
  `silver_row_count = 134420`, `dropped_row_count = 191`, `wide_row_count = 59692`, `expert_labeled_row_count = 18349`.
- The duplicate crowd harmless file is detected and excluded, with an audit event recorded for `crowdworker_harmless_duplicate`.
- Pairwise agreement metrics and model comparison outputs are generated in the report artifacts.
- Final selected model: `expert_plus_weak`.
- Weak supervision accepted `648` additional rows from weak sources.
- Pseudo-labeling accepted `0` rows on the current local dataset, so the final shipped model remains the weak-supervision model.
- Expert test split metrics:
  `expert_only` macro-F1 `0.5704`, weighted F1 `0.7824`, precision `0.8400`, recall `0.9717`, AUROC `0.7448`, PR-AUC `0.9268`.
- Expert test split metrics after weak supervision:
  `expert_plus_weak` macro-F1 `0.5758`, weighted F1 `0.7851`, precision `0.8412`, recall `0.9730`, AUROC `0.7446`, PR-AUC `0.9274`.
- Strict `HHH/NNN` consensus evaluation is also generated, with the weak-supervision model improving macro-F1 from `0.6749` to `0.6845`.

## Development Deviations and Audit Notes
- The local files did not exactly match the reference counts written earlier in this document. Following the project assumptions, the local files were treated as authoritative and the drift is documented in the generated data quality report.
- Observed local authoritative counts after cleanup are:
  `13980` subset-agreement rows with non-empty `video_id`, `15057/3294` domain-expert harmful/harmless, `10494/7794` GPT harmful/harmless, `12622/4374` crowd harmful/harmless, and `59671` unique unlabeled `video_id` values after deduplication.
- The unlabeled CSV contained `816` embedded NUL bytes and `1783` malformed lines, which are now logged and cleaned during ingestion.
- The environment exposed a Windows Spark/Hadoop compatibility issue around `winutils` and local Parquet writes. The implemented solution keeps PySpark in local mode for schema validation and pipeline orchestration, while using `pandas` plus `pyarrow` for Parquet persistence to keep the pipeline reproducible on this machine.
- The default machine Java runtime also caused Spark compatibility problems, so a local JDK 21 runtime was bundled under `tools/jdk21/` and is preferred automatically by the Spark bootstrap code.
- Because no pseudo-labeled rows passed the configured acceptance rules in this dataset version, pseudo-labeling is currently a documented negative result rather than an active improvement stage.
- Phase-2 six-category harm classification has not been implemented yet and remains an extension after the binary system.

## Assumptions and Defaults
- The generalized title is accepted, but the report will explicitly say the evaluation dataset is YouTube-derived MetaHarm.
- Local files are treated as the authoritative dataset version even where they differ from the repository description; those differences will be documented as a data audit finding.
- Version 1 is **text-first** and excludes thumbnails, frame features, live scraping, and real-time event streaming.
- The project is **offline batch + demo API**, not a production moderation service.
- The core deliverable is binary harmful-video detection; multi-label harm-category prediction is an extension, not a blocker.
- The final classifier will not use `channel`, `views`, or `duration` as core predictive features.
- If weak supervision helps but pseudo-labeling does not, the final shipped model remains the weak-supervision model.
- If the category extension is not completed, the API will still ship with binary risk scoring only.
