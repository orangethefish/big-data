# Detailed Report Outline

Project title: **Generalizable Harmful Video Detection for Streaming Platforms Using Weak Supervision and Pseudo-Labeling**

This outline follows the existing report structure:

- Member List and Task Assignment.
- Introduction.
- Theoretical Background.
- Implementation.
- Summary.
- References.

The final report should be written in English first. Translation can be done manually later.

## Report-Writing Guidance

The report should read as a complete project report for external readers, not as a walkthrough of the repository. Do not mention internal file paths, source module names, generated artifact filenames, or the directory layout. The reader should understand what the system does, how it was designed, how it was evaluated, and what the results mean.

The report may include short snippets or pseudocode when they clarify the method. For example, it is acceptable to show the risk-band rule or weak-label formula. These snippets should explain the system behavior, not expose repository structure.

Core positioning:

- The project proposes a generalizable pipeline design for harmful-video detection on streaming-platform-style data.
- All experiments are evaluated on YouTube-derived MetaHarm data.
- The report must not claim proven cross-platform performance.
- The core deliverable is binary harmful vs harmless detection.
- Six-category harm classification is a future extension.
- The first version is text-first and uses title, description, and transcript only.
- The project is an offline batch training/scoring pipeline with a small local demo API.
- The project is not a production moderation service.

Claims to avoid:

- Do not say the model is proven to generalize across all streaming platforms.
- Do not say the system is production-ready moderation infrastructure.
- Do not say pseudo-labeling improved the final model.
- Do not say the project performs image, video-frame, audio, or real-time streaming analysis.
- Do not say six-category harm classification is implemented.
- Do not treat raw accuracy as the main success metric.

Preferred wording:

- “The pipeline is designed to be generalizable.”
- “Evaluation is performed on YouTube-derived MetaHarm data.”
- “The implementation is a local batch pipeline plus a demo API.”
- “Pseudo-labeling was tested and documented as a negative result under the validation-based selection rule.”
- “Macro-F1 and harmful-class recall are prioritized over raw accuracy.”

## Section 1: Member List And Task Assignment

### Purpose

Document the group members, student IDs, task assignments, and contribution percentages.

### Suggested Task Categories

- Data engineering and ingestion pipeline.
- Bronze, silver, and gold data lake design.
- Label normalization, duplicate handling, and audit logic.
- Agreement analysis and evaluation metrics.
- Binary modeling, weak supervision, and pseudo-labeling.
- Demo API and containerized execution.
- Testing, verification, report writing, and result visualization.

### Final-Writing Notes

- Keep this section concise.
- Use the existing table format unless the instructor requires another format.
- Contribution percentages should add up to 100 percent.

## Section 2: Introduction

## 2.1 Background And Motivation

Explain the practical problem:

- Streaming platforms host large volumes of user-generated video content.
- Harmful content detection is difficult because content volume is high and manual moderation is expensive.
- Labeled harmful-content datasets are limited, noisy, and often annotated by different sources.
- Text metadata and transcripts are practical first signals because they are easier to process locally than images or full video frames.

Suggested message:

- The system is intended for moderation triage and research-style experimentation.
- It should support human review rather than replace policy decision-making.

## 2.2 Problem Statement

Define the task:

- Input: video title, description, and transcript.
- Output: binary harmful vs harmless prediction.
- Demo output: harmful decision, calibrated harmful probability, risk band, and model version.

Clarify boundaries:

- The model does not classify every possible moderation-policy violation.
- The model does not prove behavior on platforms outside the evaluated YouTube-derived dataset.
- The model does not use creator-level or popularity features.

## 2.3 Project Objectives

List the concrete objectives:

- Build a reproducible local data pipeline with bronze, silver, and gold data layers.
- Normalize heterogeneous labeled and unlabeled sources into one common schema.
- Preserve audit information for dropped rows, duplicate records, canonical merges, and conflicts.
- Train an expert-only binary classifier using domain expert labels.
- Add weak supervision from GPT-4-Turbo and crowdworker labels.
- Run one strict pseudo-labeling round on the unlabeled pool.
- Compare expert-only, expert plus weak supervision, and expert plus weak supervision plus pseudo-labeling.
- Select the final model using validation macro-F1 and harmful-class recall.
- Expose the selected model through a local prediction API.
- Generate data quality, agreement, model-comparison, pseudo-label, and error-analysis reports.

## 2.4 Dataset And Scope

Describe the dataset:

- The experimental dataset is a local version of YouTube-derived MetaHarm data.
- It contains domain expert labels, GPT-4-Turbo labels, crowdworker labels, agreement subsets, and a large unlabeled pool.
- The local dataset files are treated as the authoritative version for this project.

Important framing:

- The project is about a generalizable engineering design.
- The empirical claims are limited to the local YouTube-derived MetaHarm evaluation.
- Local source counts differ slightly from earlier reference counts; those differences are treated as data quality findings.

## 2.5 Contributions

Summarize the project contributions:

- A local single-node batch pipeline for harmful-video detection.
- A three-layer data lake design with bronze, silver, and gold layers.
- A unified data contract for heterogeneous source files.
- Canonical text consolidation by longest non-empty field.
- Duplicate-source detection and malformed-row handling.
- Agreement analysis across domain expert, GPT, and crowdworker labels.
- A text-only TF-IDF logistic regression baseline.
- Weak-supervision augmentation using estimated source reliability.
- Pseudo-labeling with strict confidence and text-length filters.
- A local prediction API for demonstration.
- Automated verification for data contracts, leakage control, model artifacts, and API behavior.

## Section 3: Theoretical Background

## 3.1 Harmful Video Detection

Explain:

- Harmful video detection is a classification problem over user-generated content.
- Binary detection asks whether a video is harmful or harmless.
- Multi-label harm-category classification is harder because one video may contain multiple harm types.
- This project prioritizes binary detection because it is the stable core deliverable.

Mention phase-2 categories:

- `IH`, `HH`, `CB`, `ADD`, `SXL`, and `PH`.
- These category classifiers are not included in the current completed system.

## 3.2 Text-First Feature Design

Explain why the core model uses only:

- Title.
- Description.
- Transcript.

Explain excluded features:

- Channel information may create creator leakage.
- Views and duration are not consistently available across labeled training sources.
- Thumbnails and frames increase setup risk and are outside the text-first scope.

Small snippet allowed in the report:

```text
model_text = title + "\n" + description + "\n" + transcript
```

Then explain that empty fields are skipped and whitespace is normalized.

## 3.3 Bronze, Silver, And Gold Data Layers

Define the three layers:

- Bronze preserves raw source records and ingestion metadata.
- Silver normalizes every source into one row-level schema.
- Gold creates modeling-ready wide tables, canonical text fields, train/validation/test splits, and audit summaries.

Explain why this matters:

- Reproducibility.
- Schema consistency.
- Traceability.
- Cleaner model training inputs.
- Easier data quality reporting.

## 3.4 Weak Supervision

Explain weak supervision:

- Weak labels are labels from imperfect but useful sources.
- This project treats GPT-4-Turbo and crowdworker labels as weak sources.
- Domain expert labels remain the primary ground truth.

Reliability formula:

```text
reliability(source) =
    agreement(source, domain_expert)
    on expert-labeled training overlap
```

Normalized weights:

```text
w_gpt = reliability_gpt / (reliability_gpt + reliability_crowd)
w_crowd = reliability_crowd / (reliability_gpt + reliability_crowd)
```

Weak probability:

```text
weak_p =
    weighted average of available weak binary labels
```

Acceptance rule:

```text
if weak_p >= 0.75:
    weak_label = harmful
elif weak_p <= 0.25:
    weak_label = harmless
else:
    abstain
```

Sample weight:

```text
sample_weight = 0.5 + abs(weak_p - 0.5)
```

## 3.5 Pseudo-Labeling

Explain pseudo-labeling:

- A trained model labels unlabeled examples.
- Only high-confidence predictions are accepted.
- The project performs one self-training round and then stops.

Acceptance rule:

- Use the best weak-supervision model.
- Require at least one non-empty text field.
- Require at least 30 tokens.
- Accept harmful probability `>= 0.95` or `<= 0.05`.
- Assign fixed sample weight `0.5`.

Small snippet allowed in the report:

```text
if p_harmful >= 0.95 or p_harmful <= 0.05:
    accept pseudo-label
else:
    discard
```

Fallback rule:

- Keep the pseudo-label model only if validation macro-F1 and harmful recall do not drop.
- In the current experiment, pseudo-labeling is exercised but not selected for the final model.

## 3.6 TF-IDF And Logistic Regression

Explain:

- TF-IDF converts text into sparse numeric features.
- Unigrams and bigrams capture individual words and short phrases.
- Logistic regression is a strong and interpretable baseline for text classification.
- Probability calibration supports probability-based thresholds and risk bands.

Implementation details to mention without repository references:

- The text vectorizer uses unigrams and bigrams.
- Rare terms are filtered with a minimum document frequency.
- The maximum feature count is capped for local training efficiency.
- Logistic regression is trained with class-probability calibration.

## 3.7 Evaluation Metrics

Define:

- Macro-F1: balances performance across harmful and harmless classes.
- Weighted F1: accounts for class support.
- Precision: how many predicted harmful examples are truly harmful.
- Recall: how many harmful examples are detected.
- AUROC: ranking quality across thresholds.
- PR-AUC: useful under class imbalance.
- Confusion matrix: counts true negatives, false positives, false negatives, and true positives.

Explain why macro-F1 and harmful-class recall are emphasized:

- Accuracy can be misleading under class imbalance.
- Recall matters because missing harmful content is costly.
- Macro-F1 prevents harmless-class performance from being hidden by the larger harmful class.

## 3.8 Annotator Agreement

Explain:

- Percent agreement measures direct label matching.
- Cohen’s kappa adjusts agreement for chance.
- Pairwise agreement is computed for domain expert vs GPT, domain expert vs crowdworker, and GPT vs crowdworker.

Interpretive setup:

- Low kappa means sources may have similar surface agreement but still weak chance-adjusted consistency.
- This motivates treating GPT and crowdworker labels as weak sources rather than ground truth.

## Section 4: Implementation

This should be the largest section of the report.

## 4.1 Methodology

## 4.1.1 System Overview

Describe the end-to-end workflow:

```text
Raw labeled and unlabeled files
  -> raw preservation layer
  -> normalized common schema
  -> canonical wide label table
  -> expert train/validation/test split
  -> agreement analysis
  -> model training and comparison
  -> final model selection
  -> local prediction API
```

Explain the main design principle:

- The pipeline separates data ingestion, data cleaning, model training, evaluation, and inference into reproducible stages.

## 4.1.2 Technology Stack

Describe:

- Python for pipeline orchestration and modeling.
- Local PySpark for big-data-style processing and schema validation.
- pandas and spreadsheet readers for Excel ingestion.
- CSV parsing with NUL-byte cleanup for the unlabeled pool.
- Parquet for efficient local columnar storage.
- scikit-learn for TF-IDF, logistic regression, calibration, and metrics.
- Matplotlib and Seaborn for report visuals.
- FastAPI for the local demo service.
- Docker-based execution for environment reproducibility.

Engineering note:

- The project encountered Windows-specific Spark/Hadoop filesystem compatibility issues.
- The final design preserves local Spark orchestration while using a reliable Parquet persistence path.
- Containerized Linux execution provides a more portable execution path.

## 4.1.3 Data Sources

Describe source groups:

- Agreement files.
- Domain expert harmful and harmless files.
- GPT-4-Turbo harmful and harmless files.
- Crowdworker harmful and harmless files.
- A duplicate crowdworker harmless file.
- A large unlabeled CSV pool.

Explain source metadata tracked internally:

- Source name.
- Source type.
- Annotator source.
- Binary label where available.
- Harm-label column where available.
- Duplicate-source marker where applicable.

Do not list internal file paths in the final report.

## 4.1.4 Bronze Layer

Explain that the bronze layer:

- Reads every source file.
- Preserves the original columns.
- Adds source metadata and ingestion metadata.
- Assigns a raw record identifier.
- Records row counts and blank video ID counts.

CSV cleanup:

- Embedded NUL bytes are removed.
- Malformed CSV lines are skipped and counted.

Current observed facts:

- 10 raw sources were ingested.
- The unlabeled CSV contained 816 embedded NUL bytes.
- The unlabeled CSV contained 1783 malformed lines.

Suggested table:

- Source group.
- Raw row count.
- Blank video ID rows.
- Cleanup notes.

## 4.1.5 Silver Layer

Describe the normalized schema:

| Field | Meaning |
|---|---|
| `video_id` | Canonical video identifier |
| `platform` | Source platform label |
| `source_group` | Agreement, annotated, or unlabeled source group |
| `annotator_source` | Domain expert, GPT, crowdworker, agreement, or unlabeled source |
| `binary_label` | Harmful/harmless label when available |
| `harm_labels_raw` | Original harm category value |
| `harm_labels_array` | Parsed harm category codes |
| `title` | Normalized title text |
| `description` | Normalized description text |
| `transcript` | Normalized transcript text |
| `published_date` | Normalized date field |
| `text_present` | Whether any model text field is present |
| `ingest_issue_flags` | Cleanup or quality flags |

Source-specific cleanup:

- Handle `link` and `links` column variants.
- Handle `description` and `deacription` spelling variants.
- Handle uppercase unlabeled fields such as `Title`, `Description`, `Transcript`, and `Date`.
- Remove padded blank spreadsheet rows.
- Drop rows with missing video IDs.
- Flag rows with no usable text.
- Exclude the duplicate crowdworker harmless source.

Current observed facts:

- Silver row count: `134420`.
- Dropped row count: `191`.
- All `60904` unlabeled silver rows have at least one text field.
- Unlabeled text coverage includes `60841` titles, `54661` descriptions, and `44134` transcripts.

## 4.1.6 Gold Layer And Wide Label Table

Explain gold construction:

- Detect same-annotator conflicts for the same video ID.
- Exclude conflicted records from training.
- Build canonical text per video ID.
- Create one wide row per video ID.
- Store domain expert, GPT, and crowdworker binary labels side by side.
- Store corresponding harm-label values.
- Add strict consensus codes when all three sources agree.

Canonical text rule:

```text
For each video_id and each text field:
    choose the longest non-empty normalized value
```

Explain:

- This preserves the richest available text when duplicate rows contain partial fields.
- The system tracks which duplicate record supplied the selected canonical value for auditability.

Current observed facts:

- Wide row count: `59692`.
- Expert-labeled rows with usable text: `18349`.
- Conflict count: `0`.
- Duplicate video IDs collapsed: `19464`.
- Strict `HHH` rows: `5901`.
- Strict `NNN` rows: `679`.

## 4.1.7 Train, Validation, And Test Split

Explain:

- Splitting is performed by unique video ID.
- Domain expert labels are the primary ground truth.
- The split is stratified by the binary label.
- The split ratio is 70/15/15.

Current split facts:

- Training rows: `12844`.
- Validation rows: `2752`.
- Test rows: `2753`.
- Total expert split rows: `18349`.

Leakage rule:

- Validation and test video IDs are held out from weak-label and pseudo-label augmentation.
- No weak-labeled or pseudo-labeled record can be added back into training if its video ID appears in expert validation or test.

## 4.1.8 Agreement Analysis

Describe method:

- Build pairwise overlaps from the canonical wide label table.
- Compute percent agreement and Cohen’s kappa.

Current results:

| Pair | Overlap | Percent Agreement | Cohen's Kappa |
|---|---:|---:|---:|
| Domain Expert vs GPT-4-Turbo | 17514 | 0.660614 | 0.248531 |
| Domain Expert vs Crowdworker | 16143 | 0.659729 | 0.033122 |
| GPT-4-Turbo vs Crowdworker | 16087 | 0.552682 | 0.062998 |

Interpretation:

- Agreement is not high.
- GPT has higher chance-adjusted agreement with domain experts than crowdworkers.
- Low kappa supports weighted weak supervision rather than directly merging all labels as truth.

Suggested figure:

- Bar chart of pairwise Cohen’s kappa.

## 4.1.9 Expert-Only Baseline Model

Describe:

- The baseline trains only on domain expert training rows.
- Input text is concatenated title, description, and transcript.
- Feature extraction uses TF-IDF unigrams and bigrams.
- The classifier is calibrated logistic regression.

Purpose:

- Establish a supervised baseline before adding weak labels or pseudo-labels.
- Compare against a majority-class baseline.

## 4.1.10 Weak-Supervision Model

Describe:

- Estimate GPT and crowdworker reliability on expert training overlap.
- Normalize reliabilities into source weights.
- Build weak labels only outside expert validation/test folds.
- Accept confident weak labels using `weak_p >= 0.75` or `weak_p <= 0.25`.
- Combine expert training rows and accepted weak rows.

Current reliability weights:

- GPT-4-Turbo: `0.499106`.
- Crowdworker: `0.500894`.

Current accepted weak labels:

- Candidate rows: `940`.
- Accepted rows: `648`.
- Accepted harmful: `493`.
- Accepted harmless: `155`.

Interpretation:

- Both weak sources receive almost equal normalized weights.
- Weak supervision adds a modest number of extra labeled examples.

## 4.1.11 Pseudo-Labeling Model

Describe:

- Score unlabeled rows using the weak-supervision model.
- Keep only rows outside expert holdouts.
- Require at least one text field and at least 30 tokens.
- Accept probabilities `>= 0.95` or `<= 0.05`.
- Add accepted pseudo-labels with fixed sample weight `0.5`.
- Retrain once.
- Keep pseudo-labeling only if validation macro-F1 and recall do not drop.

Current pseudo-label funnel:

- With any text: `40401`.
- After holdout exclusion: `40401`.
- Candidate rows with at least 30 tokens: `35676`.
- Accepted rows: `9254`.
- Accepted harmful: `9253`.
- Accepted harmless: `1`.
- Kept for final model: `False`.

Interpretation:

- Pseudo-labeling was successfully exercised.
- It accepted many high-confidence rows but was extremely skewed toward harmful predictions.
- Validation macro-F1 dropped relative to the weak-supervision model, so pseudo-labeling was not selected.

## 4.1.12 Model Selection Rule

Describe:

- Compare expert-only, expert plus weak supervision, and expert plus weak supervision plus pseudo-labeling on the same validation/test split.
- Select an augmented model only if it improves validation macro-F1 over expert-only, or ties macro-F1 while preserving or improving recall.
- Keep pseudo-labeling only if it does not reduce validation macro-F1 or harmful recall relative to weak supervision.

Current selected model:

- Final model: `expert_plus_weak`.
- Selected augmented model: `expert_plus_weak`.

Small snippet allowed in the report:

```text
if pseudo_macro_f1 >= weak_macro_f1
   and pseudo_recall >= weak_recall:
    choose pseudo-label model
else:
    keep weak-supervision model
```

## 4.1.13 Demo API

Describe endpoint behavior:

- The API accepts title, description, and transcript.
- At least one text field is required.
- It returns a harmful decision, harmful probability, risk band, and model version.

Risk-band rule:

```text
if probability < 0.35:
    risk_band = "low"
elif probability <= 0.65:
    risk_band = "medium"
else:
    risk_band = "high"
```

Suggested response example:

```json
{
  "is_harmful": true,
  "harmful_probability": 0.84,
  "risk_band": "high",
  "model_version": "expert_plus_weak"
}
```

State that the numeric probability in the example should be replaced with an actual demo result if one is collected for the report.

## 4.1.14 Containerized Execution

Describe:

- A Linux-based container environment was added for reproducible execution.
- The container includes Python and Java dependencies needed by the Spark-based workflow.
- The project can run the batch pipeline and the demo API inside the container.
- The multi-platform build definition targets both `linux/amd64` and `linux/arm64`.

Current verification:

- Container image build succeeded.
- The pipeline command completed inside the container.
- The containerized API started successfully.
- Health and prediction responses were verified inside the container.
- Multi-platform build configuration was checked.

Important limitation:

- Runtime execution was verified locally on `linux/amd64`.
- `linux/arm64` runtime execution was configured but not exercised locally.

## 4.1.15 Automated Verification

Describe test coverage:

- Text normalization and token counting.
- Harm-label parsing.
- Risk-band mapping.
- Uppercase unlabeled field aliases and typo cleanup.
- Fixture pipeline preserving unlabeled text.
- Canonical merge source auditing.
- Bronze-level source count checks.
- Silver schema contract.
- Duplicate crowd harmless exclusion.
- Disjoint train/validation/test splits.
- No weak-label or pseudo-label holdout leakage.
- Validation threshold sweep completeness.
- Final model selection rules.
- API empty-payload rejection.
- API response shape and determinism.
- API latency under 1 second after model load.

Current verification:

- Full automated test suite passed: `16 passed`.
- The Spark-backed fixture can take around 1 to 2 minutes on the local Windows machine.

## 4.2 Results

## 4.2.1 Data Quality Results

Report high-level lake results:

| Metric | Value |
|---|---:|
| Silver row count | 134420 |
| Dropped row count | 191 |
| Wide row count | 59692 |
| Expert-labeled rows with text | 18349 |
| Conflict rows excluded | 0 |
| Duplicate video IDs collapsed | 19464 |
| Strict HHH rows | 5901 |
| Strict NNN rows | 679 |

Discuss:

- The pipeline successfully produced all expected data layers.
- Local source counts drifted slightly from the earlier reference plan.
- Local files were treated as authoritative.
- The drift is documented as a data quality finding.

## 4.2.2 Source Count Drift

Include a table comparing planned and local values:

| Metric | Plan Value | Local Value | Delta |
|---|---:|---:|---:|
| Full-agreement unique IDs | 5901 | 5901 | 0 |
| Subset-agreement unique IDs | 13981 | 13980 | -1 |
| Domain expert harmful unique IDs | 15058 | 15057 | -1 |
| Domain expert harmless unique IDs | 3296 | 3294 | -2 |
| GPT harmful unique IDs | 10494 | 10494 | 0 |
| GPT harmless unique IDs | 7796 | 7794 | -2 |
| Crowdworker harmful unique IDs | 12623 | 12622 | -1 |
| Crowdworker harmless unique IDs | 4376 | 4374 | -2 |
| Unlabeled unique IDs after deduplication | 59925 | 59671 | -254 |

Interpretation:

- Differences are small for labeled files.
- The unlabeled pool has a larger deduplication drift.
- This does not invalidate the local experiment because all downstream processing uses the actual local files.

## 4.2.3 Text Coverage

Summarize:

- Wide text-present count: `59690 / 59692`.
- Title coverage: `59627`.
- Description coverage: `53640`.
- Transcript coverage: `43331`.
- Unlabeled text coverage is complete after uppercase-column normalization.

Interpretation:

- Text-first modeling is feasible because nearly all canonical rows have at least one text field.
- Transcript coverage is lower than title coverage, which matters for error analysis.

## 4.2.4 Annotator Agreement Results

Use the agreement table from the methodology section and include a kappa bar chart.

Interpretation:

- GPT and crowdworker labels are noisy relative to domain experts.
- The weak-supervision design is justified because it weights weak sources instead of merging them blindly.
- Crowdworker kappa with domain expert is especially low despite similar percent agreement, showing the value of chance-adjusted agreement.

## 4.2.5 Model Comparison On Expert Test Split

Report test metrics:

| Model | Macro-F1 | Weighted F1 | Precision | Recall | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Expert only | 0.5701 | 0.7821 | 0.8400 | 0.9712 | 0.7442 | 0.9267 |
| Expert + weak supervision | 0.5740 | 0.7842 | 0.8408 | 0.9726 | 0.7443 | 0.9274 |
| Expert + weak supervision + pseudo-labeling | 0.5665 | 0.7809 | 0.8392 | 0.9726 | 0.7411 | 0.9276 |

Majority baseline:

- Test macro-F1: `0.4507`.
- Test weighted F1: `0.7397`.
- Test recall: `1.0000`.
- AUROC: `0.5000`.

Interpretation:

- All trained models beat the majority baseline on macro-F1.
- Weak supervision provides a small improvement over expert-only.
- Pseudo-labeling does not improve macro-F1 and is therefore not selected.

Suggested figure:

- Bar chart comparing macro-F1 across model variants and splits.

## 4.2.6 Confusion Matrix Discussion

Use expert test split confusion matrices:

Expert-only:

```text
[[76, 418],
 [65, 2194]]
```

Expert plus weak:

```text
[[78, 416],
 [62, 2197]]
```

Expert plus weak plus pseudo:

```text
[[73, 421],
 [62, 2197]]
```

Interpretation:

- The final weak model reduces false negatives from 65 to 62 compared with expert-only.
- It also slightly improves true negatives from 76 to 78.
- False positives remain high, which explains modest macro-F1 despite high harmful recall.

## 4.2.7 Validation Threshold Sweep

Report best validation thresholds by macro-F1:

| Model | Best Threshold | Macro-F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Expert only | 0.70 | 0.6619 | 0.8771 | 0.8853 |
| Expert + weak supervision | 0.70 | 0.6605 | 0.8769 | 0.8835 |
| Expert + weak supervision + pseudo-labeling | 0.75 | 0.6574 | 0.8898 | 0.8264 |

Important nuance:

- The demo API uses a default threshold of 0.5 for the binary harmful decision.
- The threshold sweep is an analysis result showing the precision/recall tradeoff.
- A real moderation workflow could choose a different operating threshold depending on review capacity and false-positive cost.

## 4.2.8 Strict HHH/NNN Consensus Evaluation

Report strict consensus test metrics:

| Model | Macro-F1 | Weighted F1 | Precision | Recall | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Expert only | 0.6732 | 0.8995 | 0.9254 | 0.9867 | 0.8597 | 0.9803 |
| Expert + weak supervision | 0.6845 | 0.9035 | 0.9266 | 0.9901 | 0.8647 | 0.9809 |
| Expert + weak supervision + pseudo-labeling | 0.6722 | 0.9002 | 0.9247 | 0.9901 | 0.8583 | 0.9800 |

Interpretation:

- Performance is stronger on strict consensus examples.
- Weak supervision improves macro-F1 on this robustness benchmark.
- The strict benchmark is easier because all-source agreement likely identifies less ambiguous cases.

## 4.2.9 Pseudo-Labeling Result

Discuss:

- Pseudo-labeling accepted `9254` rows.
- Accepted pseudo-labels were almost entirely harmful: `9253` harmful and `1` harmless.
- The resulting model reduced validation macro-F1 from `0.5899` to `0.5855`.
- The final shipped model remains `expert_plus_weak`.

Suggested wording:

“Pseudo-labeling was useful as an experiment because it revealed that high-confidence model predictions on the unlabeled pool were highly skewed. Under the project’s fallback rule, this model was not shipped because it did not improve validation macro-F1 over the weak-supervision model.”

## 4.2.10 Error Analysis

Include four example types:

- Correct high-confidence prediction.
- False positive.
- False negative.
- Missing-transcript case.

Current examples:

- Correct high-confidence: `QA28Ze6HBME`, probability `0.9967`, true harmful.
- False positive: `ZR5Wx1aTNXU`, probability `0.9812`, true harmless.
- False negative: `OutB8gXxbO0`, probability `0.4951`, true harmful.
- Missing-transcript case: `C7lRtA83SQM`, probability `0.9903`, true harmful.

Analysis angles:

- False positives may come from sensational or risky wording that resembles harmful content.
- False negatives may occur when harm context is subtle, recovery-oriented, or under-expressed in text.
- Missing or noisy transcripts limit the text-only model’s ability to understand context.

## 4.2.11 Demo API Results

Report:

- The API returns the selected model version.
- Empty input is rejected.
- Output is deterministic for repeated identical inputs.
- Average prediction latency after model load is under 1 second in automated tests.

Suggested response example:

```json
{
  "is_harmful": true,
  "harmful_probability": 0.84,
  "risk_band": "high",
  "model_version": "expert_plus_weak"
}
```

Again, replace the probability with an actual demo result if one is collected.

## Section 5: Summary

## 5.1 Discussion

## 5.1.1 What Worked Well

Discuss:

- The data pipeline handles messy heterogeneous files.
- The canonical wide table creates a stable modeling contract.
- Audit logic makes data cleaning traceable.
- Weak supervision slightly improves the expert-only baseline.
- The strict consensus benchmark shows stronger performance on less ambiguous cases.
- The local API meets the demo requirements.
- Containerized execution improves reproducibility.

## 5.1.2 What Did Not Work As Well

Discuss:

- Pseudo-labeling did not improve validation macro-F1.
- Accepted pseudo-labels were extremely skewed toward harmful predictions.
- The final classifier has high recall but many false positives at threshold 0.5.
- Cohen’s kappa values show weak source disagreement and annotation difficulty.
- Text-only inputs miss visual and audio cues.
- Transcript absence or noise affects model reliability.

## 5.1.3 Limitations

Important limitations:

- Evaluation is only on YouTube-derived MetaHarm data.
- Cross-platform performance is not proven.
- The model is not a complete moderation system.
- The binary task does not explain specific harm categories.
- Class imbalance and threshold choice affect operational behavior.
- The system does not use multimodal signals.
- No real-time event streaming is implemented.
- Runtime execution was not locally verified on every possible hardware architecture.
- Some Spark shutdown logs can be noisy even when the batch run succeeds.

## 5.1.4 Ethical And Practical Considerations

Discuss:

- Harmful-content detection can affect user visibility and moderation outcomes.
- False positives can incorrectly flag harmless content.
- False negatives can allow harmful content to pass.
- A model like this should support human review rather than replace policy decision-making.
- Dataset bias and platform-specific language patterns can affect generalization.

## 5.2 Conclusion

## 5.2.1 Completed Deliverables

Summarize:

- Bronze, silver, and gold data processing workflow.
- Data quality and audit reporting.
- Agreement analysis.
- Expert-only baseline.
- Weak-supervision model.
- Pseudo-labeling experiment.
- Final model selection.
- Local demo API.
- Containerized execution workflow.
- Automated verification.

## 5.2.2 Final Result Summary

State:

- Final model: `expert_plus_weak`.
- Expert test macro-F1: `0.5740`.
- Expert test weighted F1: `0.7842`.
- Expert test precision: `0.8408`.
- Expert test recall: `0.9726`.
- Expert test AUROC: `0.7443`.
- Expert test PR-AUC: `0.9274`.
- Strict consensus macro-F1: `0.6845`.

Suggested final interpretation:

“The final system meets the project objective of building a reproducible harmful-video detection pipeline with weak supervision and a local inference demo. Weak supervision provides a modest improvement over the expert-only baseline, while pseudo-labeling is documented as a negative result under the project’s validation-based selection rule.”

## 5.2.3 Future Work

Recommended future work:

- Implement six one-vs-rest harm-category classifiers for `IH`, `HH`, `CB`, `ADD`, `SXL`, and `PH`.
- Add stronger threshold tuning based on moderation cost.
- Improve calibration and pseudo-label class balance.
- Explore transformer-based text models if compute budget allows.
- Add multimodal signals such as thumbnails, frames, or audio transcripts after the text pipeline is stable.
- Evaluate on non-YouTube or cross-platform datasets.
- Add richer explainability for analyst review.
- Improve container log cleanliness.
- Verify runtime execution on additional hardware architectures.

## References

## Recommended Reference Categories

Add references for:

- MetaHarm or the source dataset paper/documentation.
- Harmful content or online safety moderation literature.
- Weak supervision or data programming.
- Pseudo-labeling/self-training.
- TF-IDF and logistic regression for text classification.
- Cohen’s kappa.
- Macro-F1, AUROC, and PR-AUC under class imbalance.
- Apache Spark.
- scikit-learn.
- FastAPI.
- Docker.

## Reference Verification Note

Before finalizing the report, verify exact bibliographic metadata and citation formatting. Do not rely only on memory for paper titles, authors, venues, or years.

## Suggested Figures And Tables

## Figures

- End-to-end architecture diagram.
- Bronze/silver/gold data flow.
- Annotator Cohen’s kappa bar chart.
- Macro-F1 model comparison chart.
- Pseudo-label funnel diagram.
- Confusion matrix visualization for the final model.
- API request/response flow.

## Tables

- Dataset source summary table.
- Silver schema contract table.
- Data quality summary table.
- Source count drift table.
- Text coverage table.
- Annotator agreement table.
- Model comparison table.
- Strict consensus benchmark table.
- Pseudo-label funnel table.
- API request/response contract table.
- Verification summary table.

## Optional Appendices

Use appendices only if the required report length allows.

## Appendix A: API Contract

Include:

- Request fields.
- Response fields.
- Example JSON request.
- Example JSON response.
- Risk-band rule.

## Appendix B: Key Pseudocode

Include short snippets only when they help explain the method:

- Canonical text selection.
- Weak-label acceptance.
- Pseudo-label acceptance.
- Risk-band mapping.
- Final model fallback rule.

## Appendix C: Reproducibility Summary

Describe reproducibility at a conceptual level:

- The pipeline can rebuild processed data from raw local inputs.
- The same expert split is reused across model variants.
- Weak and pseudo labels are blocked from leaking into validation/test holdouts.
- The final model is selected by validation behavior, not by test-set tuning.
- Containerized execution reduces environment-specific setup risk.
