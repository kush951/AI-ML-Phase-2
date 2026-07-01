"""
PlaceMux - FastAPI Backend
Serves ML models and handles job-student matching predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import pickle
import json
import os
import numpy as np
from datetime import datetime
import pandas as pd

# Import custom modules
from feature_store import FeatureStore
from data_generator import DataGenerator

app = FastAPI(
    title="PlaceMux ML API",
    description="Job-Student Matching System API",
    version="1.0.0"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Global Variables
# =========================

model = None
feature_store = None
students_db = None
jobs_db = None
model_name = None

# =========================
# Pydantic Models
# =========================

class StudentProfile(BaseModel):
    student_id: str
    years_experience: float
    gpa: float
    certifications: int
    projects_completed: int
    aptitude_score: int
    skills: Dict[str, int]


class JobPost(BaseModel):
    job_id: str
    title: str
    min_experience: int
    min_gpa: float
    required_certifications: int
    salary_range_min: int
    salary_range_max: int
    company_rating: float
    required_skills: Dict[str, int]


class MatchRequest(BaseModel):
    student_id: str
    job_id: str


class MatchResponse(BaseModel):
    student_id: str
    job_id: str
    match_score: float
    match_probability: float
    recommendation: str
    explainability: Dict
    timestamp: str


# =========================
# Utility Functions
# =========================

def clean_dataframe(df):
    """
    Clean dataframe for JSON serialization
    """

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


# =========================
# Model Loading
# =========================

def load_model():

    global model, feature_store, model_name

    model_path = 'models/best_model.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Train the model first."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open('models/models_info.json', 'r') as f:
        info = json.load(f)
        model_name = info['best_model']

    feature_store = FeatureStore()
    feature_store.load_feature_store()

    print(f"✓ Model loaded: {model_name}")


def load_sample_data():

    global students_db, jobs_db

    if os.path.exists('data/students.csv'):

        students_db = pd.read_csv('data/students.csv')
        jobs_db = pd.read_csv('data/jobs.csv')

    else:

        generator = DataGenerator(
            n_students=50,
            n_jobs=30
        )

        students_db = generator.generate_students()
        jobs_db = generator.generate_jobs()

    # Clean invalid values
    students_db = clean_dataframe(students_db)
    jobs_db = clean_dataframe(jobs_db)


# =========================
# Startup Event
# =========================

@app.on_event("startup")
async def startup_event():

    try:

        load_model()
        load_sample_data()

        print("✓ PlaceMux API ready!")

    except Exception as e:

        print(f"⚠ Startup error: {e}")


# =========================
# Health Routes
# =========================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": model_name if model else "Not Loaded"
    }


@app.get("/model-card")
async def model_card():

    return {
        "model_name": model_name,
        "version": "1.0",
        "status": "active",
        "framework": "scikit-learn",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def get_model_info():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    with open('models/results.json', 'r') as f:
        results = json.load(f)

    return {
        "model_name": model_name,
        "features_count": len(feature_store.feature_columns),
        "validation_metrics": results.get(model_name, {})
    }


# =========================
# Prediction Routes
# =========================

@app.post("/predict/match", response_model=MatchResponse)
async def predict_match(request: MatchRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:

        student_data = students_db[
            students_db['student_id'] == request.student_id
        ].iloc[0].to_dict()

        job_data = jobs_db[
            jobs_db['job_id'] == request.job_id
        ].iloc[0].to_dict()

    except IndexError:

        raise HTTPException(
            status_code=404,
            detail="Student or Job not found"
        )

    try:

        features = feature_store.create_feature_vector(
            student_data,
            job_data
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Feature engineering failed: {str(e)}"
        )

    features_2d = features.reshape(1, -1)

    prediction = model.predict(features_2d)[0]

    probability = float(
        model.predict_proba(features_2d)[0][1]
    )

    explainability = _generate_explanation(
        student_data,
        job_data,
        probability
    )

    # Recommendation logic
    if probability > 0.75:
        recommendation = "STRONG MATCH"

    elif probability > 0.60:
        recommendation = "GOOD MATCH"

    elif probability > 0.40:
        recommendation = "MODERATE MATCH"

    else:
        recommendation = "WEAK MATCH"

    return MatchResponse(
        student_id=request.student_id,
        job_id=request.job_id,
        match_score=round(probability * 100, 2),
        match_probability=round(probability, 4),
        recommendation=recommendation,
        explainability=explainability,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/batch")
async def predict_batch_matches(
    student_id: str,
    top_k: int = 5
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    if student_id not in students_db['student_id'].values:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    student_data = students_db[
        students_db['student_id'] == student_id
    ].iloc[0].to_dict()

    matches = []

    for _, job_data in jobs_db.iterrows():

        job_dict = job_data.to_dict()

        features = feature_store.create_feature_vector(
            student_data,
            job_dict
        )

        features_2d = features.reshape(1, -1)

        probability = float(
            model.predict_proba(features_2d)[0][1]
        )

        matches.append({
            "job_id": job_dict['job_id'],
            "job_title": job_dict.get('title', 'Unknown'),
            "match_probability": round(probability, 4),
            "match_score": round(probability * 100, 2)
        })

    matches = sorted(
        matches,
        key=lambda x: x['match_probability'],
        reverse=True
    )[:top_k]

    return {
        "student_id": student_id,
        "timestamp": datetime.now().isoformat(),
        "top_matches": matches
    }


# =========================
# Student Routes
# =========================

@app.get("/students/{student_id}")
async def get_student(student_id: str):

    try:

        student = students_db[
            students_db['student_id'] == student_id
        ].iloc[0]

        return student.to_dict()

    except IndexError:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )


@app.get("/students")
async def list_students(limit: int = 20):

    clean_students = clean_dataframe(
        students_db.copy()
    )

    return clean_students.head(limit).to_dict('records')


# =========================
# Job Routes
# =========================

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):

    try:

        job = jobs_db[
            jobs_db['job_id'] == job_id
        ].iloc[0]

        return job.to_dict()

    except IndexError:

        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )


@app.get("/jobs")
async def list_jobs(limit: int = 20):

    clean_jobs = clean_dataframe(
        jobs_db.copy()
    )

    return clean_jobs.head(limit).to_dict('records')


# =========================
# Explainability
# =========================

def _generate_explanation(
    student_data: Dict,
    job_data: Dict,
    probability: float
):

    explanation = {
        "reasoning": [],
        "strengths": [],
        "gaps": [],
        "confidence": (
            "HIGH"
            if probability > 0.7
            else "MEDIUM"
            if probability > 0.5
            else "LOW"
        )
    }

    # Experience Analysis
    student_exp = student_data.get(
        'years_experience',
        0
    )

    job_min_exp = job_data.get(
        'min_experience',
        0
    )

    if student_exp >= job_min_exp:

        explanation["strengths"].append(
            f"Experience matches requirement "
            f"({student_exp} years)"
        )

    else:

        explanation["gaps"].append(
            f"Experience below requirement "
            f"({student_exp} < {job_min_exp})"
        )

    # GPA Analysis
    student_gpa = student_data.get('gpa', 0)

    job_min_gpa = job_data.get('min_gpa', 0)

    if student_gpa >= job_min_gpa:

        explanation["strengths"].append(
            f"GPA requirement satisfied "
            f"({student_gpa:.2f})"
        )

    else:

        explanation["gaps"].append(
            f"GPA below required threshold"
        )

    # Aptitude Analysis
    student_aptitude = student_data.get(
        'aptitude_score',
        0
    )

    aptitude_level = (
        "strong"
        if student_aptitude > 70
        else "moderate"
        if student_aptitude > 50
        else "developing"
    )

    explanation["reasoning"].append(
        f"Student aptitude score of "
        f"{student_aptitude} indicates "
        f"{aptitude_level} capacity for role"
    )

    return explanation


# =========================
# Main
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )