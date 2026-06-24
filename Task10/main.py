"""
PlaceMux · FastAPI Inference Server
Exposes /match, /rank, /health endpoints.
Run: uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
sys.path.insert(0, ".")
from models.inference import PlaceMuxMatcher
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

app = FastAPI(
    title="PlaceMux Matching API",
    description="AI-powered student–job matching with explainable scores",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = PlaceMuxMatcher()


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class StudentProfile(BaseModel):
    student_id: str = "STU_001"
    years_experience: float = Field(default=2.0, ge=0, le=20)
    gpa: float = Field(default=3.0, ge=0, le=4.0)
    # Skill scores 0–1
    skill_python: float = 0.5; skill_sql: float = 0.5
    skill_machine_learning: float = 0.5; skill_data_analysis: float = 0.5
    skill_deep_learning: float = 0.3; skill_javascript: float = 0.3
    skill_react: float = 0.3; skill_node_js: float = 0.3
    skill_java: float = 0.3; skill_aws: float = 0.3
    skill_docker: float = 0.3; skill_kubernetes: float = 0.2
    skill_communication: float = 0.7; skill_teamwork: float = 0.7
    skill_problem_solving: float = 0.7; skill_project_management: float = 0.5
    skill_excel: float = 0.5; skill_tableau: float = 0.4
    skill_spark: float = 0.3; skill_statistics: float = 0.5


class JobProfile(BaseModel):
    job_id: str = "JOB_001"
    title: str = "Data Scientist"
    salary_lpa: float = 15.0
    req_python: float = 0.9; req_sql: float = 0.7
    req_machine_learning: float = 0.9; req_data_analysis: float = 0.9
    req_deep_learning: float = 0.6; req_javascript: float = 0.1
    req_react: float = 0.1; req_node_js: float = 0.1
    req_java: float = 0.1; req_aws: float = 0.5
    req_docker: float = 0.4; req_kubernetes: float = 0.2
    req_communication: float = 0.7; req_teamwork: float = 0.6
    req_problem_solving: float = 0.8; req_project_management: float = 0.4
    req_excel: float = 0.5; req_tableau: float = 0.5
    req_spark: float = 0.55; req_statistics: float = 0.8


class MatchRequest(BaseModel):
    student: StudentProfile
    job: JobProfile


class RankRequest(BaseModel):
    student: StudentProfile
    jobs: list[JobProfile]
    top_k: int = Field(default=5, ge=1, le=20)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "SVM (RBF)", "version": "1.0.0"}


@app.post("/match")
def match_endpoint(request: MatchRequest):
    try:
        result = matcher.predict(request.student.model_dump(), request.job.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rank")
def rank_endpoint(request: RankRequest):
    try:
        results = matcher.rank_jobs_for_student(
            request.student.model_dump(),
            [j.model_dump() for j in request.jobs],
            top_k=request.top_k,
        )
        return {"student_id": request.student.student_id, "ranked_matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo")
def demo():
    """Returns a pre-built walkthrough for the demo."""
    student = {
        "student_id": "STU_DEMO_01", "years_experience": 3.5, "gpa": 3.7,
        "skill_python": 0.88, "skill_sql": 0.75, "skill_machine_learning": 0.82,
        "skill_data_analysis": 0.79, "skill_deep_learning": 0.65,
        "skill_statistics": 0.72, "skill_communication": 0.80,
        "skill_teamwork": 0.75, "skill_problem_solving": 0.85,
        "skill_javascript": 0.20, "skill_react": 0.15, "skill_aws": 0.50,
        "skill_docker": 0.45, "skill_kubernetes": 0.25, "skill_java": 0.30,
        "skill_node_js": 0.20, "skill_project_management": 0.55,
        "skill_excel": 0.60, "skill_tableau": 0.55, "skill_spark": 0.40,
    }
    jobs = [
        {"job_id": "JOB_DS_01", "title": "Data Scientist", "salary_lpa": 18.0,
         "req_python": 0.90, "req_sql": 0.70, "req_machine_learning": 0.90,
         "req_statistics": 0.80, "req_deep_learning": 0.60, "req_data_analysis": 0.90,
         "req_communication": 0.70, "req_javascript": 0.10, "req_react": 0.05,
         "req_node_js": 0.05, "req_java": 0.10, "req_aws": 0.50, "req_docker": 0.40,
         "req_kubernetes": 0.20, "req_teamwork": 0.65, "req_problem_solving": 0.80,
         "req_project_management": 0.40, "req_excel": 0.50, "req_tableau": 0.50, "req_spark": 0.55},
        {"job_id": "JOB_FE_01", "title": "Frontend Developer", "salary_lpa": 14.0,
         "req_javascript": 0.90, "req_react": 0.90, "req_node_js": 0.50,
         "req_communication": 0.70, "req_problem_solving": 0.70,
         "req_python": 0.20, "req_sql": 0.20, "req_machine_learning": 0.05,
         "req_statistics": 0.10, "req_deep_learning": 0.05, "req_data_analysis": 0.20,
         "req_aws": 0.30, "req_docker": 0.30, "req_kubernetes": 0.20, "req_teamwork": 0.65,
         "req_java": 0.10, "req_project_management": 0.30, "req_excel": 0.20,
         "req_tableau": 0.15, "req_spark": 0.05},
    ]
    from models.inference import PlaceMuxMatcher
    m = PlaceMuxMatcher()
    return {
        "demo_student": "STU_DEMO_01 (Data Science background, 3.5yr exp, GPA 3.7)",
        "results": [m.predict(student, job) for job in jobs],
    }
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )