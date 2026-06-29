"""
PlaceMux API Server
FastAPI backend for recommendation serving and college/recruiter portals
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

app = FastAPI(
    title="PlaceMux Recommendation Engine",
    description="AI-powered student-job matching and ranking",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
MATCHES_CSV = 'data/matches.csv'
STUDENTS_CSV = 'data/students.csv'
JOBS_CSV = 'data/jobs.csv'

# Load models
BEST_MODEL_PATH = 'models/Gradient_Boosting.pkl'

# Global state
best_model = None
matches_df = None
students_df = None
jobs_df = None


def load_models():
    """Load best model on startup"""
    global best_model
    try:
        with open(BEST_MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            best_model = {
                'model': data['model'],
                'scaler': data['scaler'],
                'feature_names': data['feature_names'],
                'metrics': data['metrics']
            }
        print(f"✓ Loaded best model: {data['model_name']}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")


def load_data():
    """Load datasets on startup"""
    global matches_df, students_df, jobs_df
    try:
        matches_df = pd.read_csv(MATCHES_CSV)
        students_df = pd.read_csv(STUDENTS_CSV)
        jobs_df = pd.read_csv(JOBS_CSV)
        print(f"✓ Loaded {len(matches_df)} matches")
        print(f"✓ Loaded {len(students_df)} students")
        print(f"✓ Loaded {len(jobs_df)} jobs")
    except Exception as e:
        print(f"✗ Error loading data: {e}")


@app.on_event("startup")
async def startup():
    """Initialize on server startup"""
    load_models()
    load_data()


# Pydantic Models
class RecommendationRequest(BaseModel):
    student_id: str
    job_id: str
    tech_overlap: float
    soft_overlap: float
    student_avg_tech_score: float
    student_avg_soft_score: float
    gpa_fit: float
    exp_fit: float
    student_gpa: float
    job_min_gpa: float
    student_years_exp: int
    job_exp_required: int


class RecommendationResponse(BaseModel):
    student_id: str
    job_id: str
    match_score: float
    confidence: float
    recommendation: str
    explanation: Dict
    match_quality: str


class StudentProfile(BaseModel):
    student_id: str
    college: str
    gpa: float
    years_of_experience: int
    avg_technical_score: float
    avg_soft_score: float


class JobPosting(BaseModel):
    job_id: str
    company: str
    role: str
    min_gpa: float
    experience_required: int


class AnalyticsRequest(BaseModel):
    college: Optional[str] = None
    company: Optional[str] = None


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "PlaceMux Recommendation Engine",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy" if best_model else "unhealthy",
        "model_loaded": best_model is not None,
        "data_loaded": matches_df is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=RecommendationResponse)
async def predict_recommendation(request: RecommendationRequest):
    """Get recommendation for a student-job pair"""

    if not best_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        feature_order = best_model['feature_names']
        features_dict = {
            'tech_overlap': request.tech_overlap,
            'soft_overlap': request.soft_overlap,
            'student_avg_tech_score': request.student_avg_tech_score,
            'student_avg_soft_score': request.student_avg_soft_score,
            'gpa_fit': request.gpa_fit,
            'exp_fit': request.exp_fit,
            'student_gpa': request.student_gpa,
            'job_min_gpa': request.job_min_gpa,
            'student_years_exp': request.student_years_exp,
            'job_exp_required': request.job_exp_required
        }

        X = np.array([[features_dict[f] for f in feature_order]])

        # Predict
        model = best_model['model']
        scaler = best_model['scaler']
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0]

        match_score = float(confidence[1])
        confidence_pct = float(confidence[1] * 100)

        # Determine recommendation
        if match_score > 0.75:
            recommendation = "HIGHLY RECOMMENDED"
            match_quality = "excellent"
        elif match_score > 0.6:
            recommendation = "RECOMMENDED"
            match_quality = "good"
        elif match_score > 0.5:
            recommendation = "CONSIDER"
            match_quality = "fair"
        else:
            recommendation = "NOT RECOMMENDED"
            match_quality = "poor"

        # Generate explanation
        explanation = {
            "technical_skills_match": f"{request.tech_overlap * 100:.1f}% overlap",
            "soft_skills_match": f"{request.soft_overlap * 100:.1f}% overlap",
            "academic_fit": f"GPA difference: {abs(request.student_gpa - request.job_min_gpa):.1f}",
            "experience_fit": f"Years: Student={request.student_years_exp}, Required={request.job_exp_required}",
            "overall_score": f"Score: {match_score:.3f}"
        }

        return RecommendationResponse(
            student_id=request.student_id,
            job_id=request.job_id,
            match_score=match_score,
            confidence=confidence_pct,
            recommendation=recommendation,
            explanation=explanation,
            match_quality=match_quality
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/student/{student_id}")
async def get_student(student_id: str):
    """Get student profile"""
    if students_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    student = students_df[students_df['student_id'] == student_id]
    if student.empty:
        raise HTTPException(status_code=404, detail="Student not found")

    return student.to_dict('records')[0]


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Get job posting"""
    if jobs_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    job = jobs_df[jobs_df['job_id'] == job_id]
    if job.empty:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_dict('records')[0]


@app.get("/students")
async def list_students(college: Optional[str] = None, limit: int = 50):
    """List students with optional college filter"""
    if students_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = students_df
    if college:
        df = df[df['college'] == college]

    return df.head(limit).to_dict('records')


@app.get("/jobs")
async def list_jobs(company: Optional[str] = None, role: Optional[str] = None, limit: int = 50):
    """List jobs with optional filters"""
    if jobs_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = jobs_df
    if company:
        df = df[df['company'] == company]
    if role:
        df = df[df['role'] == role]

    return df.head(limit).to_dict('records')


@app.get("/analytics/college/{college}")
async def college_analytics(college: str):
    """Get analytics for a college"""
    if matches_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    college_matches = matches_df[matches_df['college'] == college]

    if college_matches.empty:
        raise HTTPException(status_code=404, detail="No data for this college")

    return {
        "college": college,
        "total_student_job_pairs": len(college_matches),
        "good_matches": int(college_matches['is_good_match'].sum()),
        "match_rate": float(college_matches['is_good_match'].mean()),
        "avg_match_score": float(college_matches['match_score'].mean()),
        "avg_tech_overlap": float(college_matches['tech_overlap'].mean()),
        "avg_soft_overlap": float(college_matches['soft_overlap'].mean()),
    }


@app.get("/analytics/company/{company}")
async def company_analytics(company: str):
    """Get analytics for a company"""
    if matches_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    company_matches = matches_df[matches_df['company'] == company]

    if company_matches.empty:
        raise HTTPException(status_code=404, detail="No data for this company")

    return {
        "company": company,
        "total_candidate_matches": len(company_matches),
        "good_matches": int(company_matches['is_good_match'].sum()),
        "match_quality_rate": float(company_matches['is_good_match'].mean()),
        "avg_candidate_quality": float(company_matches['student_avg_tech_score'].mean()),
        "applicant_pool_size": len(company_matches['company'].unique()),
    }


@app.get("/model/metrics")
async def model_metrics():
    """Get best model metrics"""
    if not best_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = best_model['metrics']
    return {
        "model_type": "Gradient Boosting",
        "metrics": {
            "precision": float(metrics.get('precision', 0)),
            "recall": float(metrics.get('recall', 0)),
            "f1_score": float(metrics.get('f1', 0)),
            "roc_auc": float(metrics.get('roc_auc', 0)),
            "accuracy": float(metrics.get('accuracy', 0)),
            "false_positive_rate": float(metrics.get('false_positive_rate', 0)),
        }
    }


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    if matches_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    return {
        "total_matches": len(matches_df),
        "total_students": len(students_df),
        "total_jobs": len(jobs_df),
        "total_colleges": len(students_df['college'].unique()),
        "total_companies": len(jobs_df['company'].unique()),
        "match_rate": float(matches_df['is_good_match'].mean()),
        "avg_match_score": float(matches_df['match_score'].mean()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)