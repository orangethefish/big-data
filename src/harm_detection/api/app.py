from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator

from harm_detection.config import MODELS_DIR
from harm_detection.utils.text import build_model_text, risk_band


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = None
    description: str | None = None
    transcript: str | None = None

    @model_validator(mode="after")
    def validate_non_empty_text(self) -> "PredictRequest":
        if not any([self.title, self.description, self.transcript]):
            raise ValueError("At least one of title, description, or transcript must be provided.")
        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    is_harmful: bool
    harmful_probability: float
    risk_band: str
    model_version: str


def _model_path() -> str:
    return os.environ.get("HARM_MODEL_PATH", str(MODELS_DIR / "final_model.joblib"))


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    path = _model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle was not found at {path}. Run `harm-detect train` first.")
    return joblib.load(path)


app = FastAPI(title="Generalizable Harmful Video Detection Demo API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    text = build_model_text(request.title, request.description, request.transcript)
    probability = float(model.predict_proba([text])[:, 1][0])
    return PredictResponse(
        is_harmful=probability >= 0.5,
        harmful_probability=probability,
        risk_band=risk_band(probability),
        model_version=str(metadata.get("model_version", "unknown")),
    )
