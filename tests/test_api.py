from __future__ import annotations

import os
import time
import json

import pytest
from fastapi.testclient import TestClient

from conftest import ARTIFACTS_READY, MODELS_DIR
from harm_detection.api.app import app, load_model_bundle


pytestmark = pytest.mark.artifacts


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_predict_rejects_empty_payload() -> None:
    os.environ["HARM_MODEL_PATH"] = str(MODELS_DIR / "final_model.joblib")
    load_model_bundle.cache_clear()
    client = TestClient(app)
    response = client.post("/predict", json={"title": "", "description": "", "transcript": ""})
    assert response.status_code == 422


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_predict_response_shape_and_determinism() -> None:
    os.environ["HARM_MODEL_PATH"] = str(MODELS_DIR / "final_model.joblib")
    load_model_bundle.cache_clear()
    client = TestClient(app)
    payload = {
        "title": "policy rant about a public figure",
        "description": "this is a short synthetic description for a regression test",
        "transcript": "the speaker criticizes a group and repeats the same hostile framing several times",
    }
    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    body = first.json()
    assert set(body) == {"is_harmful", "harmful_probability", "risk_band", "model_version"}
    assert 0.0 <= body["harmful_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    summary = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    assert body["model_version"] == summary["final_model_name"]


@pytest.mark.skipif(not ARTIFACTS_READY, reason="Generated artifacts are required for this test.")
def test_predict_average_latency_stays_under_one_second_after_model_load() -> None:
    os.environ["HARM_MODEL_PATH"] = str(MODELS_DIR / "final_model.joblib")
    load_model_bundle.cache_clear()
    client = TestClient(app)
    payload = {
        "title": "policy rant about a public figure",
        "description": "this is a short synthetic description for a regression test",
        "transcript": "the speaker criticizes a group and repeats the same hostile framing several times",
    }
    warmup = client.post("/predict", json=payload)
    assert warmup.status_code == 200

    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    average_latency = (time.perf_counter() - start) / iterations
    assert average_latency < 1.0
