from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
import joblib

from feature_engineering import SkillExtractor, FeatureEngineer
from models import load_model

# Initialize FastAPI app
app = FastAPI(
    title="PlaceMux Skill Matching API",
    description="End-to-End Status Tracking & Parsing for Job Matching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
feature_names = None
skill_extractor = None
feature_engineer = None


@app.on_event("startup")
def load_models():
    """Load models on startup"""
    global model, feature_names, skill_extractor, feature_engineer
    
    try:
        model = load_model('models/best_model.pkl')
        print(f"✓ Loaded model: {model.name}")
        
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        skill_extractor = SkillExtractor()
        feature_engineer = FeatureEngineer()
        
        print("✓ Models loaded successfully")
    except FileNotFoundError:
        print("⚠ Models not found. Please run train.py first.")


# Request/Response models
class Resume(BaseModel):
    candidate_id: str
    verified_skills: str
    experience_years: int


class JobPosting(BaseModel):
    job_id: str
    job_title: str
    required_skills: str
    experience_required: int


class MatchRequest(BaseModel):
    resume: Resume
    job: JobPosting


class MatchResponse(BaseModel):
    candidate_id: str
    job_id: str
    job_title: str
    match_score: float
    is_match: bool
    confidence: float
    explanation: str
    skill_overlap: List[str]
    missing_skills: List[str]


class BatchMatchRequest(BaseModel):
    resumes: List[Resume]
    jobs: List[JobPosting]


class BatchMatchResponse(BaseModel):
    matches: List[MatchResponse]
    total_matches: int
    average_score: float


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model.name if model else "None"
    }


@app.get("/model-info")
def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model.name,
        "features": feature_names,
        "feature_count": len(feature_names)
    }


@app.post("/match", response_model=MatchResponse)
def match_candidate_job(request: MatchRequest):
    """Match a single candidate resume with a job posting"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract skills
        resume_skills = skill_extractor.extract_skills(request.resume.verified_skills)
        job_skills = skill_extractor.extract_skills(request.job.required_skills)
        
        # Calculate overlap
        skill_overlap = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))
        
        # Create features for single pair
        features_dict = feature_engineer._create_pair_features(
            {
                'verified_skills': request.resume.verified_skills,
                'experience_years': request.resume.experience_years
            },
            {
                'required_skills': request.job.required_skills,
                'experience_required': request.job.experience_required
            }
        )
        
        # Create DataFrame for prediction
        X_single = pd.DataFrame([features_dict])[feature_names]
        
        # Make prediction
        prediction = model.predict(X_single)[0]
        probability = model.predict_proba(X_single)[0, 1]
        
        # Generate explanation
        explanation = f"Skill match: {features_dict['skill_overlap']:.1%}. " \
                     f"Experience match: {features_dict['exp_match']:.1%}. " \
                     f"Required skills covered: {len(skill_overlap)}/{len(job_skills)}."
        
        return MatchResponse(
            candidate_id=request.resume.candidate_id,
            job_id=request.job.job_id,
            job_title=request.job.job_title,
            match_score=float(probability),
            is_match=bool(prediction),
            confidence=float(max(probability, 1 - probability)),
            explanation=explanation,
            skill_overlap=skill_overlap,
            missing_skills=missing_skills
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-match", response_model=BatchMatchResponse)
def batch_match(request: BatchMatchRequest):
    """Match multiple candidates with multiple jobs"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    matches = []
    
    try:
        for resume in request.resumes:
            for job in request.jobs:
                # Create match request
                match_req = MatchRequest(resume=resume, job=job)
                
                # Get match
                match = match_candidate_job(match_req)
                matches.append(match)
        
        # Filter to only positive matches and sort by score
        positive_matches = [m for m in matches if m.is_match]
        positive_matches.sort(key=lambda x: x.match_score, reverse=True)
        
        avg_score = np.mean([m.match_score for m in matches]) if matches else 0.0
        
        return BatchMatchResponse(
            matches=positive_matches,
            total_matches=len(positive_matches),
            average_score=float(avg_score)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/skill-extract")
def extract_skills(text: str):
    """Extract skills from text"""
    
    if skill_extractor is None:
        raise HTTPException(status_code=503, detail="Skill extractor not loaded")
    
    try:
        skills = skill_extractor.extract_skills(text)
        return {
            "extracted_skills": skills,
            "skill_count": len(skills)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ontology")
def get_skill_ontology():
    """Get skill ontology"""
    
    if skill_extractor is None:
        raise HTTPException(status_code=503, detail="Skill extractor not loaded")
    
    return skill_extractor.skill_ontology


@app.get("/")
def root():
    """Root endpoint with API documentation"""
    return {
        "service": "PlaceMux Skill Matching API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "match_single": "/match (POST)",
            "batch_match": "/batch-match (POST)",
            "extract_skills": "/skill-extract (POST)",
            "ontology": "/ontology (GET)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
