"""
PlaceMux · Proctoring Hardening · FastAPI Inference Server
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json, uuid
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError("pip install fastapi uvicorn pydantic")

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"

# ── Load artefacts ────────────────────────────────────────────────────
model      = joblib.load(MODELS_DIR / "best_model.pkl")
engineer   = joblib.load(MODELS_DIR / "feature_engineer.pkl")
feat_cols  = joblib.load(MODELS_DIR / "feature_cols.pkl")
with open(MODELS_DIR / "experiment_log.json") as f:
    experiment = json.load(f)

THRESHOLD = 0.40

app = FastAPI(
    title="PlaceMux Proctoring Hardening API",
    description="Real-time cheating detection for PlaceMux skill assessments",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request / Response models ─────────────────────────────────────────
class SessionInput(BaseModel):
    session_id: Optional[str] = None
    gaze_off_screen_pct: float = 10.0
    avg_gaze_deviation_px: float = 45.0
    gaze_fixation_count: int = 180
    keystroke_rhythm_score: float = 0.82
    avg_keypress_interval_ms: float = 210.0
    copy_paste_count: int = 1
    backspace_ratio: float = 0.12
    mouse_speed_variance: float = 0.3
    suspicious_click_pattern: int = 0
    tab_switch_count: int = 1
    window_blur_duration_sec: float = 2.0
    fullscreen_exit_count: int = 0
    audio_anomaly_count: int = 0
    background_noise_level_db: float = 28.0
    multiple_face_detected: int = 0
    face_absent_pct: float = 5.0
    phone_detected: int = 0
    lighting_score: float = 0.78
    session_duration_min: float = 52.0
    answer_edit_cycles: int = 3


class PredictionResponse(BaseModel):
    session_id: str
    risk_score: float
    verdict: str
    risk_level: str
    threshold: float
    reasons: list
    model: str
    model_auc: float


# ── Helper ────────────────────────────────────────────────────────────
def session_to_df(inp: SessionInput) -> pd.DataFrame:
    row = inp.model_dump(exclude={"session_id"})
    return pd.DataFrame([row])


def get_risk_level(score: float) -> str:
    if score >= 0.5:  return "HIGH"
    if score >= 0.3:  return "MEDIUM"
    return "LOW"


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    best = next(r for r in experiment["results"] if r["name"] == experiment["best_model"])
    return {
        "status": "ok",
        "models": experiment["best_model"],
        "roc_auc": best["roc_auc"],
        "f1": best["f1"],
        "fpr": best["fpr"],
        "trained_on": experiment["n_samples"],
        "timestamp": experiment["timestamp"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(inp: SessionInput):
    sid = inp.session_id or f"SES-{uuid.uuid4().hex[:8].upper()}"
    df_raw = session_to_df(inp)

    # engineer features
    df_eng = engineer.transform(df_raw)

    # align to training feature columns
    for col in feat_cols:
        if col not in df_eng.columns:
            df_eng[col] = 0.0
    X = df_eng[feat_cols]

    risk_score = float(model.predict_proba(X)[0, 1])
    verdict    = "FLAGGED" if risk_score >= THRESHOLD else "CLEARED"

    # build explanation from engineered row
    row = df_eng.iloc[0]
    explanation = engineer.explain(row)

    best = next(r for r in experiment["results"] if r["name"] == experiment["best_model"])

    return PredictionResponse(
        session_id=sid,
        risk_score=round(risk_score, 4),
        verdict=verdict,
        risk_level=explanation["risk_level"],
        threshold=THRESHOLD,
        reasons=explanation["reasons"],
        model=experiment["best_model"],
        model_auc=best["roc_auc"],
    )


@app.get("/explain/{session_id}")
def explain(session_id: str):
    return {
        "session_id": session_id,
        "message": "POST to /predict with session signals to get a full explanation.",
        "explainability": "Every prediction returns a plain-English reasons list. No black boxes.",
    }


@app.get("/experiment")
def get_experiment():
    return experiment


@app.get("/features")
def get_features():
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0])
    else:
        return {"error": "Model does not expose feature importances"}

    ranked = sorted(zip(feat_cols, imps), key=lambda x: x[1], reverse=True)
    return {"features": [{"feature": f, "importance": round(float(i), 5)} for f, i in ranked]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
