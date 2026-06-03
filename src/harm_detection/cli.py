from __future__ import annotations

import argparse
import json

import uvicorn

from harm_detection.api.app import app
from harm_detection.modeling.training import train_models
from harm_detection.pipeline.data_pipeline import build_data_lake
from harm_detection.reporting.reports import generate_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalizable harmful video detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-lake", help="Build bronze/silver/gold Parquet layers.")
    subparsers.add_parser("train", help="Train binary models and save the selected bundle.")
    subparsers.add_parser("report", help="Generate markdown reports and plots from saved artifacts.")
    subparsers.add_parser("run-all", help="Build the data lake, train models, and generate reports.")

    serve = subparsers.add_parser("serve", help="Run the FastAPI demo service.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "build-lake":
        print(json.dumps(build_data_lake(), indent=2))
        return
    if args.command == "train":
        print(json.dumps(train_models(), indent=2))
        return
    if args.command == "report":
        print(json.dumps(generate_reports(), indent=2))
        return
    if args.command == "run-all":
        lake_summary = build_data_lake()
        training_summary = train_models()
        report_summary = generate_reports()
        print(
            json.dumps(
                {
                    "lake": lake_summary,
                    "training": training_summary,
                    "reports": report_summary,
                },
                indent=2,
            )
        )
        return
    if args.command == "serve":
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
