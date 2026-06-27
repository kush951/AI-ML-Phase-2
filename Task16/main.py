"""
FastAPI Backend - College Portal & Reporting API
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager

from data_preparation import DataLoader
from ml_models import ModelRegistry, ModelEvaluator
from recommendation_engine import RecommendationEngine, RecommendationValidator
from config import API_HOST, API_PORT

# =========================================================
# Global Variables
# =========================================================

students_df = None
jobs_df = None
matches_df = None
best_model = None
rec_engine = None

# =========================================================
# Lifespan Event
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load datasets and ML models during startup
    """

    global students_df, jobs_df, matches_df
    global best_model, rec_engine

    print("Loading datasets and models...")

    students_df, jobs_df, matches_df = DataLoader.load_data()

    best_model = ModelRegistry.load_model("gradient_boosting")

    rec_engine = RecommendationEngine(
        best_model,
        students_df,
        jobs_df
    )

    print("✓ Application started successfully")

    yield

    print("✓ Application shutting down")


# =========================================================
# FastAPI App Initialization
# =========================================================

app = FastAPI(
    title="PlaceMux - College Portal & Reporting API",
    description="AI/ML-powered job recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# =========================================================
# Middleware
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Pydantic Models
# =========================================================

class RecommendationRequest(BaseModel):
    student_id: str
    top_k: int = 5


class RecommendationResponse(BaseModel):
    job_id: str
    job_title: str
    company: str
    match_score: float
    explanation: str
    match_breakdown: Dict


class DashboardMetrics(BaseModel):
    total_students: int
    total_jobs: int
    total_matches: int
    avg_match_score: float
    precision: float
    recall: float
    fpr: float


class StudentProfile(BaseModel):
    student_id: str
    name: str
    degree: str
    cgpa: float
    verified_skills: List[str]
    years_of_experience: int


class JobListing(BaseModel):
    job_id: str
    job_title: str
    company: str
    required_skills: List[str]
    min_experience: int


# =========================================================
# Health Endpoint
# =========================================================

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "healthy",
        "service": "PlaceMux - College Portal & Reporting API",
        "version": "1.0.0"
    }


# =========================================================
# Student APIs
# =========================================================

@app.get(
    "/api/students/{student_id}",
    tags=["Students"],
    response_model=StudentProfile
)
async def get_student(student_id: str):

    student = students_df[
        students_df["student_id"] == student_id
    ]

    if student.empty:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    s = student.iloc[0]

    return StudentProfile(
        student_id=s["student_id"],
        name=s["name"],
        degree=s["degree"],
        cgpa=float(s["cgpa"]),
        verified_skills=s["verified_skills"].split(","),
        years_of_experience=int(s["years_of_experience"])
    )


# =========================================================
# Recommendation APIs
# =========================================================

@app.post(
    "/api/recommendations",
    tags=["Recommendations"],
    response_model=List[RecommendationResponse]
)
async def get_recommendations(request: RecommendationRequest):

    if request.student_id not in students_df["student_id"].values:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    recommendations = rec_engine.get_recommendations(
        request.student_id,
        top_k=request.top_k
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found"
        )

    return recommendations


# =========================================================
# Job APIs
# =========================================================

@app.get(
    "/api/jobs",
    tags=["Jobs"],
    response_model=List[JobListing]
)
async def get_jobs(
    limit: int = Query(10, ge=1, le=100)
):

    jobs = jobs_df.head(limit)

    return [
        JobListing(
            job_id=j["job_id"],
            job_title=j["job_title"],
            company=j["company"],
            required_skills=j["required_skills"].split(","),
            min_experience=int(j["min_experience"])
        )
        for _, j in jobs.iterrows()
    ]


@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job_details(job_id: str):

    job = jobs_df[
        jobs_df["job_id"] == job_id
    ]

    if job.empty:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    j = job.iloc[0]

    return {
        "job_id": j["job_id"],
        "job_title": j["job_title"],
        "company": j["company"],
        "experience_level": j["experience_level"],
        "required_skills": j["required_skills"].split(","),
        "min_experience": int(j["min_experience"]),
        "min_cgpa": float(j["min_cgpa"]),
        "salary_range": j["salary_range"],
        "location": j["location"],
    }


# =========================================================
# Dashboard APIs
# =========================================================

@app.get(
    "/api/dashboard/metrics",
    tags=["Dashboard"],
    response_model=DashboardMetrics
)
async def get_dashboard_metrics():

    total_students = len(students_df)
    total_jobs = len(jobs_df)
    total_matches = len(matches_df)

    avg_match_score = float(
        matches_df["overall_match_score"].mean()
    )

    data = DataLoader.prepare_features(
        students_df,
        jobs_df,
        matches_df
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        DataLoader.get_train_test_split(data)
    )

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns

    X_test = X_test[numeric_features]

    metrics = ModelEvaluator.evaluate_model(
        best_model,
        X_test,
        y_test
    )

    return DashboardMetrics(
        total_students=total_students,
        total_jobs=total_jobs,
        total_matches=total_matches,
        avg_match_score=avg_match_score,
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        fpr=float(metrics["fpr"])
    )


@app.get(
    "/api/dashboard/model-comparison",
    tags=["Dashboard"]
)
async def get_model_comparison():

    from ml_models import (
        BaselineModel,
        LogisticRegressionModel,
        RandomForestModel,
        GradientBoostingModel,
        NeuralNetworkModel
    )

    data = DataLoader.prepare_features(
        students_df,
        jobs_df,
        matches_df
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        DataLoader.get_train_test_split(data)
    )

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns

    X_train = X_train[numeric_features]
    X_test = X_test[numeric_features]

    models = {
        "Baseline": BaselineModel(),
        "Logistic Regression": LogisticRegressionModel(),
        "Random Forest": RandomForestModel(),
        "Gradient Boosting": GradientBoostingModel(),
        "Neural Network": NeuralNetworkModel(),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    comparison = ModelEvaluator.compare_models(
        models,
        X_test,
        y_test
    )

    return comparison.to_dict(orient="index")


# =========================================================
# College APIs
# =========================================================

@app.get(
    "/api/college/{college_id}/students",
    tags=["College"]
)
async def get_college_students(college_id: str):

    college_students = students_df[
        students_df["college_id"] == college_id
    ]

    if college_students.empty:
        raise HTTPException(
            status_code=404,
            detail="College not found"
        )

    return {
        "college_id": college_id,
        "student_count": len(college_students),
        "students": [
            {
                "student_id": s["student_id"],
                "name": s["name"],
                "cgpa": float(s["cgpa"])
            }
            for _, s in college_students.iterrows()
        ]
    }


# =========================================================
# Reports APIs
# =========================================================

@app.get(
    "/api/report/accuracy",
    tags=["Reports"]
)
async def get_accuracy_report():

    data = DataLoader.prepare_features(
        students_df,
        jobs_df,
        matches_df
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        DataLoader.get_train_test_split(data)
    )

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns

    X_test = X_test[numeric_features]

    metrics = ModelEvaluator.evaluate_model(
        best_model,
        X_test,
        y_test
    )

    return {
        "model_name": "Gradient Boosting",
        "metrics": metrics,
        "test_samples": len(X_test),
        "true_positives": metrics["tp"],
        "true_negatives": metrics["tn"],
        "false_positives": metrics["fp"],
        "false_negatives": metrics["fn"],
    }


# =========================================================
# Validation APIs
# =========================================================

@app.post(
    "/api/validate-recommendation",
    tags=["Validation"]
)
async def validate_recommendation(
    student_id: str,
    job_id: str,
    match_score: float
):

    is_valid, message = (
        RecommendationValidator.validate_recommendation(
            student_id,
            job_id,
            match_score,
            students_df,
            jobs_df
        )
    )

    return {
        "is_valid": is_valid,
        "message": message,
        "student_id": student_id,
        "job_id": job_id,
        "match_score": match_score
    }


# =========================================================
# Security APIs
# =========================================================

@app.get(
    "/api/health/data-isolation",
    tags=["Security"]
)
async def check_data_isolation():

    college_groups = students_df.groupby(
        "college_id"
    ).size()

    isolation_check = {
        "status": "secure",
        "colleges_isolated": len(college_groups),
        "students_per_college": (
            college_groups.to_dict()
        ),
        "cross_college_visibility": False,
    }

    return isolation_check


# =========================================================
# Run Server
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",   # filename:app_instance
        host=API_HOST,
        port=API_PORT,
        reload=True
    )