# Generalizable Harmful Video Detection for Streaming Platforms Using Weak Supervision and Pseudo-Labeling

This repository implements a text-first harmful video detection pipeline for streaming platforms in general, with all experiments evaluated on the YouTube-derived MetaHarm dataset available in [`dataset`](/D:/Personal/big-data-new/dataset). The project is intentionally framed as a generalizable pipeline design and engineering exercise, not as proof of cross-platform performance.

Here is an easy-to-read, professional introduction tailored for your paper or project report. It uses clear analogies, breaks down the technical jargon inline, and flows smoothly so anyone can understand the core value of your work immediately.

---

## Introduction

With millions of hours of video uploaded every single day, modern streaming platforms face a massive challenge: keeping users safe from harmful content (like violence, hate speech, or dangerous activities). Relying entirely on human moderators is impossible to scale, while traditional Artificial Intelligence (AI) models require massive amounts of perfect, frame-by-frame human labels to learn what "harmful" looks like. Worse yet, internet trends move fast; an AI trained on last year's data often fails to recognize completely new types of harmful content today.

To break this bottleneck, this project introduces a **Generalizable Harmful Video Detection framework** built specifically for live and on-demand streaming platforms. Instead of demanding flawless, expensive human annotations, our system relies on two smart AI strategies:

* **Weak Supervision:** Instead of checking every single second of a video, we give the AI "lazy" or high-level clues—like just a single tag marking whether an entire 10-minute video is safe or unsafe overall. The model then learns to hunt for the specific moments that caused the flag on its own.
* **Pseudo-Labeling:** The AI acts as its own teacher. An initial version of the model makes its best guess on unlabeled videos, generates high-confidence "pseudo-labels" (AI-made sticky notes), and uses those notes to train an even stronger, more precise version of itself.

---

> ### 💡 Why This Matters
> 
> 
> Traditional AI systems suffer from **poor generalization**—meaning they work perfectly on the exact data they were trained on, but fail instantly when thrown into the chaotic, real-world mix of a live streaming platform.

By combining weak supervision with an iterative pseudo-labeling loop, our approach builds an AI that doesn't just memorize specific rules. Instead, it learns a deep, generalized understanding of unsafe behaviors. The result is a highly adaptable, resource-efficient detection system that can deploy quickly, scale effortlessly across streaming platforms, and catch harmful content even if it has never seen that specific trend before.

## Scope

- Core deliverable: binary harmful vs harmless detection.
- Phase 2 extension: six-category harm classification (`IH`, `HH`, `CB`, `ADD`, `SXL`, `PH`).
- Local architecture: single-node PySpark for bronze/silver/gold data processing, scikit-learn for batch modeling, and FastAPI for a small demo service.
- Version 1 is text-only and uses `title`, `description`, and `transcript` as model inputs.

## Repository Layout

- `dataset/`: authoritative local MetaHarm source files.
- `src/harm_detection/`: pipeline, modeling, reporting, and API code.
- `tests/`: unit and integration coverage.
- `artifacts/`: generated lake tables, model bundles, and reports.
- `tools/jdk21/`: optional bundled JDK 21 used automatically for local Spark runs on machines where newer Java releases break Hadoop compatibility.

## Data Lake Design

- `bronze/`: one Parquet dataset per raw file, preserving original columns and ingestion metadata.
- `silver/`: normalized records with a unified schema across expert, weak, agreement, and unlabeled sources.
- `gold/`: canonical wide tables, audits, splits, agreement metrics, and modeling-ready datasets.

## Main Commands

```bash
harm-detect build-lake
harm-detect train
harm-detect report
harm-detect run-all
harm-detect serve
```

## Container Usage

The repository now includes a Linux-based Docker setup so the Spark batch workflow can run without the Windows `winutils` and local Parquet write issues we saw on native Windows.

### Docker Compose

Build the image:

```bash
docker compose build
```

Run the full batch workflow:

```bash
docker compose run --rm pipeline
```

Run a specific batch command:

```bash
docker compose run --rm pipeline harm-detect build-lake
docker compose run --rm pipeline harm-detect train
docker compose run --rm pipeline harm-detect report
docker compose run --rm pipeline python -m pytest -q
```

Run the demo API:

```bash
docker compose up api
```

The API will be available at [http://localhost:8000/health](http://localhost:8000/health).

### Multi-Platform Builds

The image is designed to build on both `linux/amd64` and `linux/arm64` using multi-arch base images and Debian packages.

Build for the local platform:

```bash
docker build -t harmful-video-detection:local .
```

Preview the multi-platform Buildx definition:

```bash
docker buildx bake --print
```

Build and load a single-platform image through Buildx:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t harmful-video-detection:local \
  .
```

Build and push a multi-platform image:

```bash
docker buildx bake --push
```

Or explicitly:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t harmful-video-detection:latest \
  --push \
  .
```

### Runtime Notes

- The image does not bake the dataset or generated artifacts into the build context. The repo is bind-mounted into `/workspace` at runtime instead.
- The container sets `HARM_DETECTION_ROOT=/workspace`, so the existing CLI continues to read `dataset/` and write `artifacts/` in the mounted repo.
- The Compose setup uses Linux containers, which is the intended path for cross-platform Spark execution.

## Deliverables

- Reproducible bronze/silver/gold Parquet outputs.
- Agreement analysis for Domain Experts, GPT-4-Turbo, and Crowdworker labels.
- Three binary model variants: expert-only, expert + weak supervision, and expert + weak supervision + pseudo-labeling.
- Report artifacts covering data quality, model comparison, pseudo-label acceptance, and error analysis.
- A local `POST /predict` FastAPI endpoint for short-text inference.

## Current Validated Status (2026-06-04)

- The unlabeled CSV normalization now preserves uppercase `Title` / `Description` / `Transcript` / `Date` fields, so unlabeled text participates in canonical text consolidation and pseudo-labeling as intended.
- Current lake summary:
  `silver_row_count = 134420`, `dropped_row_count = 191`, `wide_row_count = 59692`, `expert_labeled_row_count = 18349`.
- Current unlabeled text coverage after normalization:
  `60904/60904` silver unlabeled rows have at least one text field, with `60841` titles, `54661` descriptions, and `44134` transcripts preserved.
- Final selected model remains `expert_plus_weak`.
- Weak supervision accepted `648` additional rows.
- Pseudo-labeling is now exercised on the corrected unlabeled pool:
  `40401` text-bearing unlabeled rows, `35676` rows with at least `30` tokens, and `9254` accepted pseudo-labels.
- The pseudo-label model was not shipped because it reduced validation macro-F1 relative to `expert_plus_weak`, so pseudo-labeling is currently a documented negative result rather than the final deployed stage.
- Expert test split metrics for the shipped `expert_plus_weak` model:
  macro-F1 `0.5740`, weighted F1 `0.7842`, precision `0.8408`, recall `0.9726`, AUROC `0.7443`, PR-AUC `0.9274`.
- Strict `HHH/NNN` consensus evaluation for `expert_plus_weak`:
  macro-F1 `0.6845`, weighted F1 `0.9035`, precision `0.9266`, recall `0.9901`, AUROC `0.8647`, PR-AUC `0.9809`.

## Verification

- Host verification passed with `16` automated tests via `python -m pytest -q`.
- `harm-detect build-lake`, `harm-detect train`, and `harm-detect report` completed successfully on the host.
- `docker compose build` and `docker compose run --rm pipeline harm-detect run-all` completed successfully.
- The containerized API returned a successful health check and a valid `POST /predict` response.
