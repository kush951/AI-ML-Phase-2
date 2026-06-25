"""
PlaceMux · Matching API
FastAPI server exposing matching, ranking and explanation endpoints.

Run:
    uvicorn api.main:app --reload --port 8000
"""
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import extract_features, explain_features, FEATURE_NAMES
from src.explainer import explain_match, plain_english_summary

# ─── Load artefacts ────────────────────────────────────────────────────────────

artefacts_path = ROOT / "experiments" / "best_model.pkl"
with open(artefacts_path, "rb") as f:
    ARTEFACTS = pickle.load(f)

MODEL = ARTEFACTS["models"]
VECTORIZER = ARTEFACTS["vectorizer"]
MODEL_NAME = ARTEFACTS["model_name"]
FEATURE_IMPORTANCES = ARTEFACTS.get("feature_importances", {})

with open(ROOT / "experiments" / "experiment_log.json") as f:
    EXP_LOG = json.load(f)

# Load dataset for live demo
with open(ROOT / "data" / "dataset.json") as f:
    _data = json.load(f)

STUDENTS_BY_ID = {s["id"]: s for s in _data["students"]}
JOBS_BY_ID = {j["id"]: j for j in _data["jobs"]}

# ─── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlaceMux Matching API",
    description="AI-powered student–job matching with explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ───────────────────────────────────────────────────────────────────

class VerifiedSkills(BaseModel):
    __root__: dict[str, float]


class StudentIn(BaseModel):
    id: str = "STU0001"
    name: str = "Alice"
    domain: str = "software_engineering"
    education_level: str = "Bachelor"
    education_level_idx: int = 2
    seniority: str = "Mid"
    seniority_idx: int = 2
    experience_years: float = 3.0
    verified_skills: dict[str, float] = Field(default_factory=dict)
    skill_list: list[str] = Field(default_factory=list)
    city: str = "Bangalore"
    salary_expectation_lpa: float = 12.0
    open_to_remote: bool = True
    bio: str = "Mid-level software engineer with 3 years experience."


class JobIn(BaseModel):
    id: str = "JOB0001"
    title: str = "Software Engineer"
    company: str = "TechCorp"
    domain: str = "software_engineering"
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_skill_threshold: int = 65
    min_experience_years: float = 2.0
    seniority_required: str = "Mid"
    seniority_required_idx: int = 2
    education_required: str = "Bachelor"
    education_required_idx: int = 2
    city: str = "Bangalore"
    remote_ok: bool = True
    salary_budget_lpa: float = 15.0
    description: str = "Exciting engineering role."


class MatchRequest(BaseModel):
    student: StudentIn
    job: JobIn


class RankRequest(BaseModel):
    job: JobIn
    student_ids: Optional[list[str]] = None  # None = use all students


# ─── Inference helpers ─────────────────────────────────────────────────────────

def _predict(student: dict, job: dict) -> tuple[float, np.ndarray]:
    feats = extract_features(student, job, VECTORIZER).reshape(1, -1)
    proba = MODEL.predict_proba(feats)[0, 1]
    return float(proba), feats[0]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "PlaceMux Matching API",
        "models": MODEL_NAME,
        "version": "1.0.0",
        "endpoints": ["/match", "/rank", "/explain", "/metrics", "/models",
                      "/students", "/jobs"]
    }


@app.post("/match")
def match_student_to_job(req: MatchRequest):
    """Score a single student–job pair and return probability + explanation."""
    s = req.student.dict()
    j = req.job.dict()

    proba, _ = _predict(s, j)
    feature_breakdown = explain_features(s, j, VECTORIZER)
    explanation = explain_match(s, j, proba, feature_breakdown, FEATURE_IMPORTANCES)
    summary = plain_english_summary(explanation)

    return {
        "student_id": s["id"],
        "job_id": j["id"],
        "match_score": round(proba, 4),
        "score_pct": round(proba * 100),
        "verdict": explanation["verdict"],
        "explanation": explanation,
        "summary": summary,
        "model_used": MODEL_NAME,
    }


@app.post("/rank")
def rank_candidates_for_job(req: RankRequest):
    """Rank all (or given) students for a job, return sorted list with scores."""
    job = req.job.dict()

    if req.student_ids:
        students = [STUDENTS_BY_ID[sid] for sid in req.student_ids
                    if sid in STUDENTS_BY_ID]
    else:
        students = list(STUDENTS_BY_ID.values())

    results = []
    for student in students:
        proba, _ = _predict(student, job)
        feature_breakdown = explain_features(student, job, VECTORIZER)
        explanation = explain_match(student, job, proba, feature_breakdown)
        results.append({
            "rank": None,
            "student_id": student["id"],
            "student_name": student["name"],
            "match_score": round(proba, 4),
            "score_pct": round(proba * 100),
            "verdict": explanation["verdict"],
            "top_reasons": explanation["reasons_for"][:2],
            "top_concerns": explanation["reasons_against"][:1],
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {
        "job_id": job["id"],
        "job_title": job["title"],
        "company": job["company"],
        "total_candidates": len(results),
        "shortlist_threshold": 0.5,
        "shortlisted": [r for r in results if r["match_score"] >= 0.5],
        "not_shortlisted": [r for r in results if r["match_score"] < 0.5],
        "model_used": MODEL_NAME,
    }


@app.get("/explain/{student_id}/{job_id}")
def explain_existing_pair(student_id: str, job_id: str):
    """Explain an existing student–job pair from the dataset."""
    if student_id not in STUDENTS_BY_ID:
        raise HTTPException(404, f"Student {student_id} not found")
    if job_id not in JOBS_BY_ID:
        raise HTTPException(404, f"Job {job_id} not found")

    s = STUDENTS_BY_ID[student_id]
    j = JOBS_BY_ID[job_id]

    proba, _ = _predict(s, j)
    feature_breakdown = explain_features(s, j, VECTORIZER)
    explanation = explain_match(s, j, proba, feature_breakdown, FEATURE_IMPORTANCES)
    summary = plain_english_summary(explanation)

    return {
        "student": s,
        "job": j,
        "match_score": round(proba, 4),
        "verdict": explanation["verdict"],
        "explanation": explanation,
        "summary": summary,
    }


@app.get("/metrics")
def get_metrics():
    """Return all models evaluation metrics from the experiment log."""
    return {
        "best_model": EXP_LOG["best_model"],
        "best_metrics": EXP_LOG["best_metrics"],
        "baseline_metrics": EXP_LOG["baseline_metrics"],
        "improvement_over_baseline": {
            k: round(EXP_LOG["best_metrics"][k] - EXP_LOG["baseline_metrics"][k], 4)
            for k in ["f1", "roc_auc", "precision", "recall"]
        },
        "feature_importances": EXP_LOG["feature_importances"],
        "dataset_stats": EXP_LOG["dataset_stats"],
    }


@app.get("/models")
def get_all_model_results():
    """Return comparison of all trained models."""
    return {
        "models": EXP_LOG["all_results"],
        "best": EXP_LOG["best_model"],
        "selection_criterion": "Harmonic mean of F1 + AUC-ROC",
    }


@app.get("/students")
def list_students(limit: int = 20, offset: int = 0):
    """List available students."""
    all_students = list(STUDENTS_BY_ID.values())
    return {
        "total": len(all_students),
        "students": all_students[offset: offset + limit],
    }


@app.get("/jobs")
def list_jobs(limit: int = 20, offset: int = 0):
    """List available jobs."""
    all_jobs = list(JOBS_BY_ID.values())
    return {
        "total": len(all_jobs),
        "jobs": all_jobs[offset: offset + limit],
    }


@app.get("/students/{student_id}")
def get_student(student_id: str):
    if student_id not in STUDENTS_BY_ID:
        raise HTTPException(404, "Student not found")
    return STUDENTS_BY_ID[student_id]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS_BY_ID:
        raise HTTPException(404, "Job not found")
    return JOBS_BY_ID[job_id]
