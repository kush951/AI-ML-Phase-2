
"""
PlaceMux · FastAPI Inference Endpoint
POST /match        — returns ranked match score + SHAP reasons
POST /match/batch  — batch predictions
GET  /health       — liveness check
GET  /metrics      — evaluation metrics
GET  /             — frontend homepage
"""

from __future__ import annotations
import json
import time
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="PlaceMux Matching Engine",
    description="AI/ML candidate-job matching with explainability",
    version="2.0.0",
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# STATIC FILES
# =========================================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =========================================================
# GLOBALS
# =========================================================

_model = None
_threshold = 0.42
_eval_metrics = {}

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def load_model():
    global _model, _threshold, _eval_metrics

    model_path = MODELS_DIR / "best_model.pkl"

    if model_path.exists():
        _model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")
    else:
        print("⚠ No trained model found.")

    thr_path = MODELS_DIR / "threshold.json"

    if thr_path.exists():
        _threshold = json.load(open(thr_path))["optimal_threshold"]

    eval_path = MODELS_DIR / "evaluation_results.json"

    if eval_path.exists():
        _eval_metrics = json.load(open(eval_path))

# =========================================================
# FRONTEND ROUTE
# =========================================================

@app.get("/")
async def home():
    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "PlaceMux API Running 🚀",
        "docs": "/docs"
    }

# =========================================================
# SCHEMAS
# =========================================================

class MatchRequest(BaseModel):

    candidate_id: str = Field(..., example="C0001")

    skill_overlap: float = Field(..., ge=0.0, le=1.0)

    exp_match_norm: float = Field(..., ge=0.0, le=1.0)

    semantic_similarity: float = Field(..., ge=0.0, le=1.0)

    verified_score_norm: float = Field(..., ge=0.0, le=1.0)

    loc_salary_match: float = Field(..., ge=0.0, le=1.0)

    domain_match: int = Field(..., ge=0, le=1)

    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class MatchResponse(BaseModel):

    candidate_id: str

    match_score: float

    verdict: str

    shap_reasons: Dict[str, float]

    plain_english: str

    latency_ms: float

    threshold_used: float

# =========================================================
# FEATURES
# =========================================================

FEATURE_COLS = [
    "skill_overlap",
    "exp_match_norm",
    "semantic_similarity",
    "verified_score_norm",
    "loc_salary_match",
    "domain_match",
]

FEATURE_LABELS = {
    "skill_overlap": "Skill overlap",
    "exp_match_norm": "Experience match",
    "semantic_similarity": "Semantic similarity",
    "verified_score_norm": "Verified test score",
    "loc_salary_match": "Location & salary fit",
    "domain_match": "Same domain",
}

# =========================================================
# HELPERS
# =========================================================

def _fallback_shap(x: pd.DataFrame):

    weights = [0.38, 0.10, 0.22, 0.30, 0.08, 0.05]

    return {
        FEATURE_LABELS[f]: round(float(x[f].values[0]) * w, 4)
        for f, w in zip(FEATURE_COLS, weights)
    }


def _compute_shap(model, x: pd.DataFrame):

    try:
        import shap

        explainer = shap.TreeExplainer(model)

        sv = explainer.shap_values(x)

        if isinstance(sv, list):
            sv = sv[1]

        vals = sv[0] if sv.ndim == 2 else sv

        attr = {
            FEATURE_LABELS[f]: round(float(v), 4)
            for f, v in zip(FEATURE_COLS, vals)
        }

        return dict(
            sorted(attr.items(), key=lambda i: abs(i[1]), reverse=True)
        )

    except Exception:

        return _fallback_shap(x)


def _verdict(score: float):

    if score >= 0.75:
        return "Strong Match ✅"

    if score >= 0.50:
        return "Partial Match ⚠"

    return "Weak Match ❌"


def _plain_english(shap_reasons: Dict[str, float], score: float):

    top3 = list(shap_reasons.items())[:3]

    parts = [f"{name} ({val:+.3f})" for name, val in top3]

    return (
        f"Match score {score:.2f} driven by "
        f"{'; '.join(parts)}."
    )

# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "threshold": _threshold,
    }


@app.get("/metrics")
def get_metrics():

    if not _eval_metrics:
        raise HTTPException(
            status_code=404,
            detail="No evaluation metrics found."
        )

    return _eval_metrics


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):

    if _model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    t0 = time.perf_counter()

    x = pd.DataFrame([
        {f: getattr(req, f) for f in FEATURE_COLS}
    ])

    score = float(_model.predict_proba(x)[:, 1][0])

    thr = (
        req.threshold
        if req.threshold is not None
        else _threshold
    )

    shap_r = _compute_shap(_model, x)

    latency_ms = round(
        (time.perf_counter() - t0) * 1000,
        2
    )

    return MatchResponse(
        candidate_id=req.candidate_id,
        match_score=round(score, 4),
        verdict=_verdict(score),
        shap_reasons=shap_r,
        plain_english=_plain_english(shap_r, score),
        latency_ms=latency_ms,
        threshold_used=thr,
    )


@app.post("/match/batch")
def match_batch(requests: list[MatchRequest]):

    return [match(r) for r in requests]

