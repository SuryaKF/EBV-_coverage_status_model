from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from coverage_pipeline.config import settings
from coverage_pipeline.db import get_conn
from coverage_pipeline.ml.artifact_store import load_artifact_path
from coverage_pipeline.ml.registry import get_approved_model


class PredictRequest(BaseModel):
    payer_name: str = Field(min_length=1)
    state_name: str = Field(min_length=1)
    acronym: str = Field(min_length=1)
    expansion: str = ""
    explanation: str = ""


class PredictResponse(BaseModel):
    prediction: dict
    metadata: dict


app = FastAPI(title="Coverage Prediction API", version="1.0.0")

_MODEL = None
_MODEL_VERSION = None


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings.api_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key is not configured")
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _load_model() -> None:
    global _MODEL, _MODEL_VERSION
    model_row = get_approved_model()
    artifact_path = load_artifact_path(model_row["artifact_uri"])
    _MODEL = joblib.load(artifact_path)
    _MODEL_VERSION = model_row["model_version"]


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": _MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(check_api_key)])
def predict(payload: PredictRequest) -> PredictResponse:
    if _MODEL is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    started = time.perf_counter()

    combined_text = f"{payload.expansion} {payload.explanation}".lower().strip()
    text_features = _MODEL["tfidf"].transform([combined_text]).toarray()

    cat_df = pd.DataFrame(
        {
            "payer_name": [payload.payer_name],
            "state_name": [payload.state_name],
            "acronym_clean": [payload.acronym.strip().upper()],
        }
    )
    cat_features = _MODEL["onehot"].transform(cat_df)
    x = np.hstack([text_features, cat_features])

    pred_enc = _MODEL["model"].predict(x)
    pred_label = _MODEL["label_encoder"].inverse_transform(pred_enc)[0]

    probs = {}
    if hasattr(_MODEL["model"], "predict_proba"):
        proba = _MODEL["model"].predict_proba(x)[0]
        for cls, prob in zip(_MODEL["label_encoder"].classes_, proba):
            probs[str(cls)] = round(float(prob), 4)

    latency_ms = int((time.perf_counter() - started) * 1000)
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_log (
                    request_id,
                    payer_name,
                    state_name,
                    acronym,
                    expansion,
                    explanation,
                    prediction,
                    confidence_json,
                    model_version,
                    latency_ms,
                    predicted_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                (
                    request_id,
                    payload.payer_name,
                    payload.state_name,
                    payload.acronym,
                    payload.expansion,
                    payload.explanation,
                    pred_label,
                    json.dumps(probs),
                    _MODEL_VERSION,
                    latency_ms,
                    ts,
                ),
            )
        conn.commit()

    return PredictResponse(
        prediction={
            "coverage_status": pred_label,
            "confidence": probs,
        },
        metadata={
            "request_id": request_id,
            "model_version": _MODEL_VERSION,
            "timestamp": ts.isoformat(),
            "latency_ms": latency_ms,
        },
    )
