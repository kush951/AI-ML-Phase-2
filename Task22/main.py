"""
FastAPI Backend - PlaceMux ML Service
Serves predictions, monitoring, and retraining endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import json
from datetime import datetime
from train_pipeline import MLPipeline
from drift_detection import DriftDetector, AutoRetrainer
import os

# Initialize app
app = FastAPI(
    title="PlaceMux ML API",
    description="Job-Skill Matching with Drift Detection & Auto-Retraining",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pipeline = MLPipeline()
drift_detector = DriftDetector(threshold=2.0)
auto_retrainer = AutoRetrainer(pipeline, drift_detector)

# Global state
model_loaded = False
prediction_cache = []

# ============================================================================
# SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    features: List[float]
    feature_names: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    match: bool
    confidence: float
    model_used: str
    explanation: str
    timestamp: str

class DriftCheckRequest(BaseModel):
    features: List[List[float]]

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    false_positive_rate: float

class ModelStatsResponse(BaseModel):
    models_available: List[str]
    best_model: str
    training_timestamp: str
    metrics: Dict

class DriftReportResponse(BaseModel):
    total_drift_events: int
    recent_drifts: List
    retraining_needed: bool
    last_training_time: str

# ============================================================================
# ENDPOINTS - MODEL INFO
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelStatsResponse)
async def get_model_info():
    """Get information about trained models"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /model/train first")
    
    return {
        "models_available": list(pipeline.models.keys()),
        "best_model": pipeline.best_model_name,
        "training_timestamp": pipeline.training_timestamp,
        "metrics": pipeline.metrics
    }

@app.get("/model/metrics/{model_name}", response_model=MetricsResponse)
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    if model_name not in pipeline.metrics:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    metrics = pipeline.metrics[model_name]
    return {
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1_score": metrics['f1_score'],
        "auc_roc": metrics['auc_roc'],
        "false_positive_rate": metrics['false_positive_rate']
    }

# ============================================================================
# ENDPOINTS - TRAINING
# ============================================================================

@app.post("/model/train")
async def train_model(n_samples: int = 1000):
    """Train all models and select best one"""
    global model_loaded
    
    try:
        pipeline.run_full_pipeline(n_samples=n_samples)
        drift_detector.set_baseline(
            np.random.randn(n_samples, 10)  # For demonstration
        )
        model_loaded = True
        
        return {
            "status": "success",
            "message": "Model training completed",
            "best_model": pipeline.best_model_name,
            "models_trained": list(pipeline.models.keys()),
            "timestamp": pipeline.training_timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks, force: bool = False):
    """Trigger model retraining"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    # Check if retraining is needed
    should_retrain, reason = drift_detector.should_retrain(force=force)
    
    if not should_retrain and not force:
        return {
            "status": "skipped",
            "message": reason,
            "retraining_needed": False
        }
    
    # Perform retraining in background
    background_tasks.add_task(pipeline.run_full_pipeline, 500)
    
    return {
        "status": "started",
        "message": "Retraining started in background",
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/model/save")
async def save_model(path: str = "./models"):
    """Save trained models to disk"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        pipeline.save_models(path)
        return {
            "status": "success",
            "message": f"Models saved to {path}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.post("/model/load")
async def load_model(path: str = "./models"):
    """Load models from disk"""
    global model_loaded
    
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        pipeline.load_models(path)
        model_loaded = True
        
        return {
            "status": "success",
            "message": f"Models loaded from {path}",
            "best_model": pipeline.best_model_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

# ============================================================================
# ENDPOINTS - PREDICTIONS
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /model/train first")
    
    try:
        features = np.array(request.features)
        
        explanation = pipeline.explain_prediction(features, request.feature_names or [])
        
        # Cache prediction
        prediction_cache.append({
            'features': request.features,
            'prediction': explanation,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            "match": explanation['match'],
            "confidence": explanation['confidence'],
            "model_used": explanation['model_used'],
            "explanation": explanation['prediction_text'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(requests: List[PredictionRequest]):
    """Make batch predictions"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    results = []
    for request in requests:
        try:
            features = np.array(request.features)
            explanation = pipeline.explain_prediction(features, request.feature_names or [])
            results.append({
                "match": explanation['match'],
                "confidence": explanation['confidence'],
                "model_used": explanation['model_used'],
                "explanation": explanation['prediction_text'],
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return results

# ============================================================================
# ENDPOINTS - DRIFT MONITORING
# ============================================================================

@app.post("/drift/check")
async def check_drift(request: DriftCheckRequest):
    """Check for data drift in incoming features"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        X_current = np.array(request.features)
        drift_detected, magnitude, affected_features = drift_detector.detect_statistical_drift(X_current)
        
        return {
            "drift_detected": drift_detected,
            "drift_magnitude": float(magnitude),
            "affected_features": int(len(affected_features)),
            "threshold": drift_detector.threshold,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Drift check failed: {str(e)}")

@app.post("/drift/performance")
async def check_performance_drift(y_true: List[int], y_pred: List[int]):
    """Check for performance drift"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        drift_detected, f1_drop = drift_detector.detect_performance_drift(y_true, y_pred)
        
        return {
            "performance_drift_detected": drift_detected,
            "f1_score_drop": float(f1_drop),
            "retraining_needed": drift_detector.retraining_needed,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance check failed: {str(e)}")

@app.get("/drift/report", response_model=DriftReportResponse)
async def get_drift_report():
    """Get comprehensive drift report"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    report = drift_detector.get_drift_report()
    return report

# ============================================================================
# ENDPOINTS - MONITORING & STATS
# ============================================================================

@app.get("/stats/predictions")
async def get_prediction_stats(limit: int = 100):
    """Get recent prediction statistics"""
    recent = prediction_cache[-limit:] if limit else prediction_cache
    
    return {
        "total_predictions": len(prediction_cache),
        "recent_predictions": recent,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats/retrain-history")
async def get_retrain_history():
    """Get model retraining history"""
    return {
        "total_retrains": len(auto_retrainer.retrain_history),
        "history": auto_retrainer.retrain_history,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats/model-comparison")
async def get_model_comparison():
    """Compare all trained models"""
    if not model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    comparison = {}
    for model_name, metrics in pipeline.metrics.items():
        comparison[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'false_positive_rate': metrics['false_positive_rate']
        }
    
    return {
        "model_comparison": comparison,
        "best_model": pipeline.best_model_name,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """API Documentation"""
    return {
        "service": "PlaceMux ML API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "training": [
                "/model/train",
                "/model/retrain",
                "/model/save",
                "/model/load"
            ],
            "predictions": [
                "/predict",
                "/predict/batch"
            ],
            "drift_monitoring": [
                "/drift/check",
                "/drift/performance",
                "/drift/report"
            ],
            "statistics": [
                "/stats/predictions",
                "/stats/retrain-history",
                "/stats/model-comparison"
            ],
            "info": [
                "/model/info",
                "/model/metrics/{model_name}"
            ]
        },
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
