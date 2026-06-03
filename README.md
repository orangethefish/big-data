# Generalizable Harmful Video Detection for Streaming Platforms Using Weak Supervision and Pseudo-Labeling

This repository implements a text-first harmful video detection pipeline for streaming platforms in general, with all experiments evaluated on the YouTube-derived MetaHarm dataset available in [`dataset`](/D:/Personal/big-data-new/dataset). The project is intentionally framed as a generalizable pipeline design and engineering exercise, not as proof of cross-platform performance.

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
