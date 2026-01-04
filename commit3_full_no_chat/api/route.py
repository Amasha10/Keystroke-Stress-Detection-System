from fastapi import APIRouter, HTTPException
import pandas as pd
import joblib
from pathlib import Path

from schemas import KeystrokeRequest, StressResponse
from services.feature_extraction import extract_features
from core.config import settings

router = APIRouter(tags=["Stress Detection"])

MODEL_PATH = Path(settings.MODEL_PATH)
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_cols = bundle["feature_cols"]


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/predict", response_model=StressResponse)
def predict_stress(req: KeystrokeRequest):
    if not req.events:
        raise HTTPException(status_code=400, detail="No keystroke events provided")

    features_df = extract_features(req.events)
    X = features_df.reindex(columns=feature_cols)

    stress_probability = float(model.predict_proba(X)[0, 1])
    stress_pred = int(stress_probability >= settings.STRESS_PROB_THRESHOLD)

    return StressResponse(
        stress_probability=stress_probability,
        stress_pred=stress_pred
    )
