"""
Configuration settings for PlaceMux AI/ML Engine
"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Model configuration
    MODEL_CONFIG = {
        "logistic_regression": {
            "C": 0.1,
            "max_iter": 1000,
            "random_state": 42,
        },
        "svm": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "subsample": 0.8,
            "random_state": 42,
        }
    }
    
    # Data configuration
    DATA_CONFIG = {
        "n_samples": 500,
        "test_size": 0.2,
        "random_state": 42,
    }
    
    # Feature names
    FEATURE_NAMES = [
        'skill_overlap_ratio',
        'required_skills_covered',
        'skill_gap',
        'avg_matching_skill_score',
        'avg_all_skill_scores',
        'experience_gap',
        'excess_experience',
        'experience_match_ratio',
        'gpa_normalized',
        'total_verified_skills',
        'required_skills_count',
        'college_match'
    ]
    
    # Server configuration
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
