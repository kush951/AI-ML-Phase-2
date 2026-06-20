"""
PlaceMux Task 2 -- AI/ML Engineer backend.

Run with:
    cd backend && uvicorn app:app --reload

Then open http://localhost:8000 in your browser.
"""
from pathlib import Path
from typing import Dict
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data import save_sample_data, SKILLS
from matching import build_match_vector, baseline_score, explain_match, validate_thresholds, competency_band
from model import train_and_evaluate

app = FastAPI(title="PlaceMux Match Service - Task 2")

DATA_DIR = Path(__file__).parent.parent / "data"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ---- Stage A: load/generate sample data once at startup -------------------
STUDENTS, JOBS = save_sample_data(str(DATA_DIR))
JOBS_BY_ID = {j["id"]: j for j in JOBS}

print("Training match-quality model on sample data...")
MODEL, METRICS = train_and_evaluate(STUDENTS, JOBS, log_path=str(DATA_DIR / "experiment_log.csv"))
print("Done. Baseline vs model metrics:", METRICS)


class JobPosting(BaseModel):
    title: str
    company: str
    thresholds: Dict[str, int]  # skill -> L1-L100


@app.get("/api/skills")
def get_skills():
    return {"skills": SKILLS}


@app.get("/api/students")
def get_students():
    return {"students": STUDENTS}


@app.get("/api/jobs")
def get_jobs():
    return {"jobs": list(JOBS_BY_ID.values())}


@app.post("/api/jobs")
def post_job(job: JobPosting):
    """A company posts a job gated by L1-L100 skill thresholds."""
    errors = validate_thresholds(job.thresholds)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    job_id = f"J{str(uuid.uuid4())[:6].upper()}"
    record = {
        "id": job_id,
        "title": job.title,
        "company": job.company,
        "thresholds": job.thresholds,
    }
    JOBS_BY_ID[job_id] = record
    return record


@app.get("/api/jobs/{job_id}/candidates")
def get_candidates(job_id: str, top_k: int = 10):
    job = JOBS_BY_ID.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    thresholds = job["thresholds"]
    results = []
    for student in STUDENTS:
        scores = student["verified_skill_scores"]
        feats = build_match_vector(scores, thresholds)
        from matching import FEATURE_NAMES
        x = [[feats[name] for name in FEATURE_NAMES]]
        model_prob = float(MODEL.predict_proba(x)[0][1])
        base = baseline_score(scores, thresholds)
        met_reasons, missing_reasons = explain_match(scores, thresholds)

        results.append({
            "student_id": student["id"],
            "name": student["name"],
            "baseline_score": round(base, 3),
            "model_score": round(model_prob, 3),
            "meets_all_thresholds": len(missing_reasons) == 0,
            "coverage_ratio": round(feats["coverage_ratio"], 3),
            "reasons_met": met_reasons,
            "reasons_missing": missing_reasons,
        })

    results.sort(key=lambda r: r["model_score"], reverse=True)
    return {"job": job, "candidates": results[:top_k]}


@app.get("/api/metrics")
def get_metrics():
    """Real numbers, not vibes -- precision/recall/FPR, baseline vs model."""
    return METRICS


@app.get("/api/competency-band/{level}")
def get_band(level: int):
    if not (1 <= level <= 100):
        raise HTTPException(status_code=400, detail="Level must be between L1 and L100")
    return {"level": level, "band": competency_band(level)}


# ---- Serve frontend ---------------------------------------------------
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
