"""
PlaceMux Matching Service — FastAPI app.

Implements the two task deliverables:
  1. The student<->job feature space (features.py + matching.py)
  2. The matching API contract with Backend (this file)

Run:
    uvicorn app.main:app --reload --port 8000

Then open frontend/index.html (or serve it) — it talks to this API.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect
from typing import List

from . import models, schemas
from .database import engine, get_db, Base
from .matching import load_model, load_metrics, score_pair, explain, train_and_evaluate
from .seed_data import generate_companies, generate_students, generate_jobs

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PlaceMux Matching API",
    description="Student<->Job matching engine: feature space, scoring, and explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _student_to_dict(s: models.Student) -> dict:
    return {
        "years_experience": s.years_experience,
        "education_level": s.education_level,
        "verified_skills": s.verified_skills or {},
        "location": s.location,
        "remote_ok": s.remote_ok,
    }


def _job_to_dict(j: models.Job) -> dict:
    return {
        "min_experience": j.min_experience,
        "role_level": j.role_level,
        "required_skills": j.required_skills or {},
        "location": j.location,
        "remote_ok": j.remote_ok,
    }


# --------------------------------------------------------------- bootstrap
@app.on_event("startup")
def seed_if_empty():
    db = next(get_db())
    if db.query(models.Company).count() == 0:
        companies = generate_companies()
        for c in companies:
            db.add(models.Company(**c))
        db.commit()

        company_ids = [c.id for c in db.query(models.Company).all()]
        students = generate_students(120)
        for s in students:
            db.add(models.Student(**s))
        jobs = generate_jobs(company_ids, 25)
        for j in jobs:
            db.add(models.Job(**j))
        db.commit()
    db.close()
    # ensure a trained model + metrics exist on first boot
    get_model()


# ---------------------------------------------------------------- companies
@app.post("/companies/signup", response_model=schemas.CompanyOut, tags=["companies"])
def company_signup(payload: schemas.CompanySignup, db: Session = Depends(get_db)):
    existing = db.query(models.Company).filter(models.Company.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="A company with this email already signed up.")
    company = models.Company(**payload.dict())
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


@app.get("/companies", response_model=List[schemas.CompanyOut], tags=["companies"])
def list_companies(db: Session = Depends(get_db)):
    return db.query(models.Company).all()


# --------------------------------------------------------------------- jobs
@app.post("/companies/{company_id}/jobs", response_model=schemas.JobOut, tags=["jobs"])
def post_job(company_id: int, payload: schemas.JobCreate, db: Session = Depends(get_db)):
    company = db.query(models.Company).get(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found. Sign up first.")
    job = models.Job(company_id=company_id, **payload.dict())
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@app.get("/companies/{company_id}/jobs", response_model=List[schemas.JobOut], tags=["jobs"])
def list_company_jobs(company_id: int, db: Session = Depends(get_db)):
    return db.query(models.Job).filter(models.Job.company_id == company_id).all()


@app.get("/jobs", response_model=List[schemas.JobOut], tags=["jobs"])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(models.Job).all()


# ----------------------------------------------------------------- students
@app.post("/students", response_model=schemas.StudentOut, tags=["students"])
def create_student(payload: schemas.StudentCreate, db: Session = Depends(get_db)):
    student = models.Student(**payload.dict())
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@app.get("/students", response_model=List[schemas.StudentOut], tags=["students"])
def list_students(db: Session = Depends(get_db)):
    return db.query(models.Student).limit(200).all()


# ------------------------------------------------------------------ match
@app.get("/jobs/{job_id}/candidates", response_model=List[schemas.MatchResult], tags=["matching"])
def candidates_for_job(job_id: int, top_k: int = 10, db: Session = Depends(get_db)):
    job = db.query(models.Job).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    students = db.query(models.Student).all()
    if not students:
        raise HTTPException(status_code=404, detail="No students on file yet.")

    clf = get_model()
    job_dict = _job_to_dict(job)
    results = []
    for s in students:
        model_p, base_p, feats = score_pair(clf, _student_to_dict(s), job_dict)
        exp = explain(_student_to_dict(s), job_dict, feats)
        results.append((s, model_p, base_p, exp))

    results.sort(key=lambda r: r[1], reverse=True)
    results = results[:top_k]

    out = []
    for rank, (s, model_p, base_p, exp) in enumerate(results, start=1):
        log = models.MatchLog(
            student_id=s.id, job_id=job_id, model_score=model_p, baseline_score=base_p, explanation=exp
        )
        db.add(log)
        out.append(schemas.MatchResult(
            student_id=s.id, student_name=s.name, job_id=job_id,
            model_score=round(model_p, 4), baseline_score=round(base_p, 4),
            rank=rank, explanation=schemas.MatchExplanation(**exp),
        ))
    db.commit()
    return out


@app.get("/students/{student_id}/jobs", response_model=List[schemas.MatchResult], tags=["matching"])
def jobs_for_student(student_id: int, top_k: int = 10, db: Session = Depends(get_db)):
    student = db.query(models.Student).get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    jobs = db.query(models.Job).all()
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs posted yet.")

    clf = get_model()
    student_dict = _student_to_dict(student)
    results = []
    for j in jobs:
        model_p, base_p, feats = score_pair(clf, student_dict, _job_to_dict(j))
        exp = explain(student_dict, _job_to_dict(j), feats)
        results.append((j, model_p, base_p, exp))

    results.sort(key=lambda r: r[1], reverse=True)
    results = results[:top_k]

    return [
        schemas.MatchResult(
            student_id=student_id, student_name=student.name, job_id=j.id,
            model_score=round(model_p, 4), baseline_score=round(base_p, 4),
            rank=rank, explanation=schemas.MatchExplanation(**exp),
        )
        for rank, (j, model_p, base_p, exp) in enumerate(results, start=1)
    ]


@app.get("/match/explain", response_model=schemas.MatchResult, tags=["matching"])
def explain_one_match(student_id: int, job_id: int, db: Session = Depends(get_db)):
    student = db.query(models.Student).get(student_id)
    job = db.query(models.Job).get(job_id)
    if not student or not job:
        raise HTTPException(status_code=404, detail="Student or job not found.")
    clf = get_model()
    model_p, base_p, feats = score_pair(clf, _student_to_dict(student), _job_to_dict(job))
    exp = explain(_student_to_dict(student), _job_to_dict(job), feats)
    return schemas.MatchResult(
        student_id=student_id, student_name=student.name, job_id=job_id,
        model_score=round(model_p, 4), baseline_score=round(base_p, 4),
        rank=1, explanation=schemas.MatchExplanation(**exp),
    )


# ----------------------------------------------------------------- metrics
@app.get("/metrics", response_model=schemas.MetricsOut, tags=["metrics"])
def get_metrics():
    """Held-out precision / recall / false-positive rate for model vs baseline."""
    m = load_metrics()
    return schemas.MetricsOut(**m)


@app.post("/metrics/retrain", response_model=schemas.MetricsOut, tags=["metrics"])
def retrain():
    """Re-run training on a fresh sample draw and log the run (experiment log)."""
    global _model
    m = train_and_evaluate()
    _model = load_model.__wrapped__ if False else None
    _model = None
    return schemas.MetricsOut(**m)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
