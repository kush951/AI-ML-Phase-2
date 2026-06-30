"""
PlaceMux FastAPI Backend
Serves ML model predictions with explanations and metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlaceMux Job Matching API",
    description="AI-powered student-job matching with fairness auditing",
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

# Global variables
model = None
scaler = None
encoders = None
feature_names = None
metrics = None

# Request/Response Models
class StudentProfile(BaseModel):
    student_id: int
    years_exp: float
    gpa: float
    num_projects: int
    internships: int
    certifications: int
    gender: str
    region: str
    background: str
    verified_python: float = 0
    verified_javascript: float = 0
    verified_sql: float = 0
    verified_aws: float = 0

class JobRequirements(BaseModel):
    job_id: int
    salary_min: int
    salary_max: int
    required_exp_min: int
    required_exp_max: int
    required_gpa_min: float
    urgency_level: str
    company_size: str
    req_python: float = 0
    req_javascript: float = 0
    req_sql: float = 0
    req_aws: float = 0

class MatchRequest(BaseModel):
    student: StudentProfile
    job: JobRequirements

class MatchResponse(BaseModel):
    student_id: int
    job_id: int
    is_match: bool
    confidence: float
    match_score: float
    explanation: Dict
    key_factors: List[str]
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    model: str
    metrics: Optional[Dict]

# Load model at startup
def load_model():
    global model, scaler, encoders, feature_names, metrics
    
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/scalers.pkl', 'rb') as f:
            scaler = pickle.load(f).get('default')
        
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        with open('model_comparison.json', 'r') as f:
            data = json.load(f)
            metrics = data['best_metrics']
        
        logger.info("✓ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        return False

# Initialize model
@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.error("Failed to load model at startup")

# API Endpoints
@app.get("/", tags=["Info"])
def read_root():
    return {
        "name": "PlaceMux Job Matching API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API and model health"""
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model="RandomForest" if model else "None",
        metrics=metrics
    )

@app.get("/metrics", tags=["Analytics"])
def get_model_metrics():
    """Get model performance metrics"""
    if not metrics:
        raise HTTPException(status_code=503, detail="Model metrics not available")
    
    return {
        "accuracy": metrics.get('accuracy'),
        "precision": metrics.get('precision'),
        "recall": metrics.get('recall'),
        "f1_score": metrics.get('f1'),
        "auc": metrics.get('auc'),
        "false_positive_rate": metrics.get('fpr'),
    }

def prepare_features(student: StudentProfile, job: JobRequirements) -> np.ndarray:
    #"\"\"Prepare features for model prediction\"\"\"
    features = {
        'gender': encoders.get('gender', LabelEncoder()).transform([student.gender])[0] if 'gender' in encoders else 0,
        'region': encoders.get('region', LabelEncoder()).transform([student.region])[0] if 'region' in encoders else 0,
        'background': encoders.get('background', LabelEncoder()).transform([student.background])[0] if 'background' in encoders else 0,
        'company_size': encoders.get('company_size', LabelEncoder()).transform([job.company_size])[0] if 'company_size' in encoders else 0,
        'urgency_level': encoders.get('urgency_level', LabelEncoder()).transform([job.urgency_level])[0] if 'urgency_level' in encoders else 0,
        
        'years_exp': student.years_exp,
        'gpa': student.gpa,
        'num_projects': student.num_projects,
        'internships': student.internships,
        'certifications': student.certifications,
        
        'salary_min': job.salary_min,
        'salary_max': job.salary_max,
        'required_exp_min': job.required_exp_min,
        'required_exp_max': job.required_exp_max,
        'required_gpa_min': job.required_gpa_min,
        
        'python_match': min(student.verified_python, job.req_python) / 100 if max(student.verified_python, job.req_python) > 0 else 0,
        'javascript_match': min(student.verified_javascript, job.req_javascript) / 100 if max(student.verified_javascript, job.req_javascript) > 0 else 0,
        'sql_match': min(student.verified_sql, job.req_sql) / 100 if max(student.verified_sql, job.req_sql) > 0 else 0,
        'aws_match': min(student.verified_aws, job.req_aws) / 100 if max(student.verified_aws, job.req_aws) > 0 else 0,
    }
    
    # Convert to array in feature order
    feature_array = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)
    return feature_array

def generate_explanation(student: StudentProfile, job: JobRequirements, prediction: int, confidence: float) -> Dict:

    
    if prediction == 1:
        summary = (f"This student is a potential match for this role. "
                  f"With {int(student.years_exp)} years of experience and a "
                  f"{student.gpa:.1f} GPA, they align well with the job requirements.")
    else:
        summary = (f"This student doesn't meet the requirements for this role. "
                  f"There may be gaps in experience or skill alignment.")
    
    factors = []
    
    # Experience check
    if student.years_exp >= job.required_exp_min:
        factors.append(f"✓ Experience sufficient ({int(student.years_exp)} years >= {int(job.required_exp_min)} required)")
    else:
        factors.append(f"⚠ Experience gap ({int(student.years_exp)} years < {int(job.required_exp_min)} required)")
    
    # GPA check
    if student.gpa >= job.required_gpa_min:
        factors.append(f"✓ Academic credentials adequate (GPA {student.gpa:.1f})")
    else:
        factors.append(f"⚠ Academic credentials below requirement (GPA {student.gpa:.1f})")
    
    # Projects
    if student.num_projects > 0:
        factors.append(f"✓ Practical experience ({student.num_projects} projects)")
    else:
        factors.append(f"⚠ Limited project experience")
    
    # Internships
    if student.internships > 0:
        factors.append(f"✓ Professional experience ({student.internships} internships)")
    else:
        factors.append(f"⚠ No formal internship experience")
    
    return {
        'summary': summary,
        'factors': factors,
    }

def get_recommendations(prediction: int, confidence: float) -> List[str]:

    recommendations = []
    
    if prediction == 1:
        if confidence > 0.85:
            recommendations.append("Ready for immediate interview")
            recommendations.append("Strong technical alignment")
        elif confidence > 0.75:
            recommendations.append("Recommended for interview after skills validation")
            recommendations.append("Verify specific technical skills")
        else:
            recommendations.append("Consider for interview but assess fit carefully")
            recommendations.append("Conduct technical assessment")
    else:
        recommendations.append("Suggest skill development path")
        recommendations.append("Recommend applying for entry-level positions")
        recommendations.append("Focus on building project portfolio")
    
    return recommendations

@app.post("/predict", response_model=MatchResponse, tags=["Prediction"])
def predict_match(request: MatchRequest):

    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        X = prepare_features(request.student, request.job)
        
        # Make prediction
        prediction = model.predict(X)[0]
        confidence = float(model.predict_proba(X)[0][1])
        
        # Compute match score
        match_score = confidence * 100
        
        # Generate explanation
        explanation = generate_explanation(request.student, request.job, prediction, confidence)
        
        # Get recommendations
        recommendations = get_recommendations(prediction, confidence)
        
        return MatchResponse(
            student_id=request.student.student_id,
            job_id=request.job.job_id,
            is_match=bool(prediction == 1),
            confidence=confidence,
            match_score=match_score,
            explanation=explanation,
            key_factors=explanation['factors'],
            recommendations=recommendations,
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/fairness-report", tags=["Fairness"])
def get_fairness_report():
    try:
        with open('fairness_audit_report.json', 'r') as f:
            report = json.load(f)
        return report
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fairness report not found. Run fairness audit first.")

@app.get("/model-info", tags=["Info"])
def get_model_info():
    return {
        "model_type": "Random Forest Classifier",
        "num_features": len(feature_names) if feature_names else 0,
        "features": feature_names,
        "metrics": metrics,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
