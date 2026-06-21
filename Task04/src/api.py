"""
PlaceMux — Explainability Task
Stage C.1 — Wire the deliverable into an end-to-end servable flow.

Run:  uvicorn src.api:app --reload --port 8000   (from project root)
Docs: http://localhost:8000/docs
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.explain import MatchExplainer, FRIENDLY_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

app = FastAPI(
    title="PlaceMux Explainable Matching API",
    description="Serves match scores with a plain-English explanation payload "
                "for every student-job pair, plus the model evaluation behind it.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load once at startup ----
explainer = MatchExplainer()
students = json.loads((DATA_DIR / "students.json").read_text())
jobs = json.loads((DATA_DIR / "jobs.json").read_text())
pairs_df = pd.read_csv(DATA_DIR / "match_pairs.csv")
with open(MODELS_DIR / "model_meta.json") as f:
    model_meta = json.load(f)

students_by_id = {s["student_id"]: s for s in students}
jobs_by_id = {j["job_id"]: j for j in jobs}


def build_feature_row(student_id: str, job_id: str) -> dict:
    row = pairs_df[(pairs_df.student_id == student_id) & (pairs_df.job_id == job_id)]
    if not row.empty:
        return row.iloc[0].to_dict()

    # compute on the fly for any arbitrary pair not in the pre-built table
    s = students_by_id.get(student_id)
    j = jobs_by_id.get(job_id)
    if s is None or j is None:
        raise HTTPException(404, f"Unknown student_id or job_id")

    s_skills = set(s["skills"])
    req = set(j["required_skills"])
    nice = set(j["nice_to_have_skills"])
    req_overlap = s_skills & req
    nice_overlap = s_skills & nice
    overlap_scores = [s["verified_scores"][sk] for sk in req_overlap]
    avg_req_score = float(sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0
    min_req_score = float(min(overlap_scores)) if overlap_scores else 0.0

    return {
        "domain_match": 1.0 if s["domain"] == j["domain"] else 0.0,
        "required_overlap_count": len(req_overlap),
        "required_overlap_ratio": len(req_overlap) / max(len(req), 1),
        "nice_overlap_count": len(nice_overlap),
        "nice_overlap_ratio": len(nice_overlap) / max(len(nice), 1),
        "avg_required_skill_score": avg_req_score,
        "min_required_skill_score": min_req_score,
        "meets_min_score_threshold": 1.0 if avg_req_score >= j["min_verified_score"] else 0.0,
        "years_experience": s["years_experience"],
        "min_years_required": j["min_years_experience"],
        "years_gap": s["years_experience"] - j["min_years_experience"],
        "verified_skill_breadth": len(s["skills"]),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": model_meta["best_model_name"]}


@app.get("/students")
def list_students(limit: int = 50):
    return [
        {"student_id": s["student_id"], "domain": s["domain"], "skills": s["skills"],
         "years_experience": s["years_experience"]}
        for s in students[:limit]
    ]


@app.get("/jobs")
def list_jobs(limit: int = 50):
    return [
        {"job_id": j["job_id"], "domain": j["domain"], "required_skills": j["required_skills"],
         "nice_to_have_skills": j["nice_to_have_skills"]}
        for j in jobs[:limit]
    ]


@app.get("/match/{student_id}/{job_id}")
def match(student_id: str, job_id: str):
    feature_row = build_feature_row(student_id, job_id)
    result = explainer.explain(feature_row)
    result["student"] = students_by_id.get(student_id)
    result["job"] = jobs_by_id.get(job_id)
    return result


@app.get("/shortlist/{job_id}")
def shortlist(job_id: str, top_n: int = 10):
    """Rank all students against one job — the company-facing view."""
    if job_id not in jobs_by_id:
        raise HTTPException(404, "Unknown job_id")
    feature_rows = [build_feature_row(s["student_id"], job_id) for s in students]
    explained = explainer.explain_batch(feature_rows, top_k=3)
    results = [
        {
            "student_id": s["student_id"],
            "match_score": r["match_score"],
            "prediction": r["prediction"],
            "explanation": r["explanation"],
            "top_factors": r["top_factors"],
        }
        for s, r in zip(students, explained)
    ]
    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results[:top_n]


@app.get("/metrics")
def metrics():
    """Evaluation numbers behind the model — for the live-verification demo."""
    return model_meta


@app.get("/metrics/segments")
def segment_metrics():
    seg_path = REPORTS_DIR / "segment_breakdown.csv"
    if not seg_path.exists():
        raise HTTPException(404, "Run src/train.py first")
    return pd.read_csv(seg_path).to_dict(orient="records")


@app.get("/metrics/experiments")
def experiment_log():
    log_path = REPORTS_DIR / "experiment_log.csv"
    if not log_path.exists():
        raise HTTPException(404, "Run src/train.py first")
    return pd.read_csv(log_path).to_dict(orient="records")
