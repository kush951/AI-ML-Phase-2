"""
PlaceMux · AI/ML · FastAPI Inference Server
Exposes matching endpoints for frontend integration.
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inference import PlaceMuxMatcher, SKILLS

app = FastAPI(
    title="PlaceMux Matching API",
    description="AI-powered student-job matching with explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = PlaceMuxMatcher()


class StudentProfile(BaseModel):
    student_id: str
    years_experience: float
    cgpa: float
    skills: Dict[str, float]  # {"Python": 85.0, "SQL": 70.0}


class JobProfile(BaseModel):
    job_id: str
    role: str
    min_experience: float
    min_cgpa: float
    required_skills: Dict[str, float]  # {"Python": 80.0}
    optional_skills: Optional[Dict[str, float]] = {}


class MatchRequest(BaseModel):
    student: StudentProfile
    job: JobProfile


class RankRequest(BaseModel):
    student: StudentProfile
    jobs: List[JobProfile]


def _flatten_student(s: StudentProfile) -> dict:
    d = {"student_id": s.student_id,
         "years_experience": s.years_experience, "cgpa": s.cgpa}
    for skill in SKILLS:
        col = skill.replace(' ', '_').lower()
        d[f"skill_{col}"] = s.skills.get(skill, 0.0)
    return d


def _flatten_job(j: JobProfile) -> dict:
    d = {"job_id": j.job_id, "role": j.role,
         "min_experience": j.min_experience, "min_cgpa": j.min_cgpa}
    for skill in SKILLS:
        col = skill.replace(' ', '_').lower()
        d[f"req_skill_{col}"] = j.required_skills.get(skill, 0.0)
        d[f"opt_skill_{col}"] = (j.optional_skills or {}).get(skill, 0.0)
    return d


@app.post("/api/match")
async def match(req: MatchRequest):
    """Score a single student-job pair with explanation."""
    try:
        student = _flatten_student(req.student)
        job = _flatten_job(req.job)
        result = matcher.score_pair(student, job)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rank")
async def rank(req: RankRequest):
    """Rank a list of jobs for a student."""
    try:
        student = _flatten_student(req.student)
        jobs = [_flatten_job(j) for j in req.jobs]
        ranked = matcher.rank_jobs(student, jobs)
        return {"status": "ok", "ranked_jobs": ranked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "healthy", "model": "loaded", "version": "1.0.0"}


@app.get("/api/skills")
async def get_skills():
    return {"skills": SKILLS}
