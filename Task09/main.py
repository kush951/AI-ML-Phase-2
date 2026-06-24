"""
main.py  —  PlaceMux AI/ML API
Task 9 · Failure Handling & Resilience
Endpoints:
  GET  /health              — liveness probe
  GET  /models/summary      — all model metrics from experiment log
  POST /match               — predict match for (student, job) pair
  GET  /conversion-check    — paywall relevance check results
  GET  /explain             — one-example walkthrough
  GET  /demo                — end-to-end demo with real sample data
"""

import json
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODEL_DIR = BASE / "models"
DATA_DIR  = BASE / "data"
LOG_DIR   = BASE / "logs"
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(MODEL_DIR))

app = FastAPI(
    title="PlaceMux AI Matching API",
    description="Task 9 — Conversion-quality check & job matching inference",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    "skill_coverage", "avg_score_on_required", "n_skills_matched",
    "nice_to_have_coverage", "edu_match", "exp_match", "loc_match",
    "student_avg_skill", "years_experience", "n_student_skills",
    "n_job_required",
]

# ── Load artefacts (lazy, with error handling) ─────────────────────────────────
_model  = None
_scaler = None
_log    = None
_df     = None


def load_artefacts():
    global _model, _scaler, _log, _df
    if _model is None:
        try:
            _model  = joblib.load(MODEL_DIR / "best_model.pkl")
            _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        except FileNotFoundError:
            raise HTTPException(500, "Model not trained yet. Run train_models.py first.")
    if _log is None:
        try:
            with open(LOG_DIR / "experiment_log.json") as f:
                _log = json.load(f)
        except FileNotFoundError:
            raise HTTPException(500, "Experiment log missing. Run train_models.py first.")
    if _df is None:
        try:
            _df = pd.read_csv(DATA_DIR / "matches.csv")
        except FileNotFoundError:
            raise HTTPException(500, "Dataset missing. Run generate_data.py first.")


# ── Request / Response schemas ─────────────────────────────────────────────────
class MatchRequest(BaseModel):
    student: dict
    job: dict


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "PlaceMux AI Matching API",
    }


@app.get("/models/summary")
def models_summary():
    """Return all model metrics from the experiment log."""
    load_artefacts()
    models = _log["models"]
    rows = []
    for name, res in models.items():
        row = {
            "model": name,
            "precision": res["test"]["precision"],
            "recall":    res["test"]["recall"],
            "f1":        res["test"]["f1"],
            "fpr":       res["test"]["fpr"],
            "auc_roc":   res["test"].get("auc_roc"),
            "cv_f1":     res.get("cv_f1"),
            "is_best":   name == _log["best_model"],
        }
        rows.append(row)
    return {
        "best_model": _log["best_model"],
        "n_train":    _log["n_train"],
        "n_val":      _log["n_val"],
        "n_test":     _log["n_test"],
        "models":     rows,
        "timestamp":  _log["timestamp"],
    }


@app.get("/conversion-check")
def conversion_check():
    """Return the conversion/paywall quality check results."""
    load_artefacts()
    cc = _log["conversion_quality_check"]
    return {
        "verdict":  cc["verdict"],
        "relevance_regression_detected": cc["relevance_regression_detected"],
        "threshold": cc["threshold"],
        "paid_metrics":  cc["paid_metrics"],
        "free_metrics":  cc["free_metrics"],
        "gaps": {
            "precision": cc["precision_gap"],
            "recall":    cc["recall_gap"],
            "f1":        cc["f1_gap"],
        },
        "interpretation": (
            "The paid/free precision gap exceeds the 8 pp threshold — "
            "paywall may be influencing match relevance. Investigate."
            if cc["relevance_regression_detected"]
            else "Match quality is consistent across paid and free tiers. "
                 "Paywall has NOT skewed relevance."
        ),
    }


@app.post("/match")
def predict_match(req: MatchRequest):
    """Predict match score for a (student, job) pair with explanation."""
    load_artefacts()
    try:
        from generate_data import compute_match_features
        from train_models import explain_prediction
        result = explain_prediction(
            _model, _scaler, req.student, req.job, FEATURE_COLS
        )
        return result
    except Exception as e:
        raise HTTPException(400, f"Prediction error: {e}")


@app.get("/explain")
def explain_example():
    """One worked example: this student, this job, and why it's a match."""
    load_artefacts()
    students = json.loads((DATA_DIR / "students.json").read_text())
    jobs     = json.loads((DATA_DIR / "jobs.json").read_text())

    # Pick a known positive example
    pos_pairs = _df[_df["label"] == 1].head(20)
    if pos_pairs.empty:
        raise HTTPException(404, "No positive examples found in dataset")

    row   = pos_pairs.iloc[0]
    s_id  = row["student_id"]
    j_id  = row["job_id"]

    student = next((s for s in students if s["student_id"] == s_id), None)
    job     = next((j for j in jobs     if j["job_id"]     == j_id), None)
    if not student or not job:
        raise HTTPException(404, "Example entities not found")

    from generate_data import compute_match_features
    from train_models import explain_prediction
    explanation = explain_prediction(_model, _scaler, student, job, FEATURE_COLS)

    return {
        "student": {
            "id":        student["student_id"],
            "skills":    student["verified_skills"],
            "experience": student["years_experience"],
            "education": student["education_level"],
            "tier":      student["payment_tier"],
        },
        "job": {
            "id":               job["job_id"],
            "role":             job["role"],
            "required_skills":  job["required_skills"],
            "min_experience":   job["min_experience"],
            "education":        job["education_required"],
        },
        "prediction": explanation,
        "ground_truth_label": int(row["label"]),
    }


@app.get("/demo")
def demo():
    """End-to-end demo: real data, real metrics, real explanation."""
    load_artefacts()
    summary    = models_summary()
    conv_check = conversion_check()
    example    = explain_example()

    return {
        "demo_title":        "PlaceMux AI — Task 9 Live Demo",
        "timestamp":         datetime.utcnow().isoformat(),
        "dataset_size":      len(_df),
        "positive_rate":     round(_df["label"].mean(), 3),
        "best_model":        summary["best_model"],
        "test_metrics":      next(
            m for m in summary["models"] if m["is_best"]
        ),
        "conversion_check":  conv_check,
        "live_example":      example,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
