"""
PlaceMux - Configuration Module
College Portal & Reporting API Foundations
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "train_test_split": 0.8,
    "validation_split": 0.2,
    "random_state": 42,
    "test_size": 0.2,
}

# Feature Engineering
FEATURES = {
    "skill_features": [
        "skill_match_score",      # Overlap of verified vs required skills
        "skill_relevance_score",   # Relevance match
        "skill_level_match",       # Level of skill required vs possessed
    ],
    "experience_features": [
        "years_of_experience",
        "experience_relevance_score",
        "domain_match_score",
    ],
    "education_features": [
        "degree_match_score",
        "specialization_match_score",
    ],
    "soft_skill_features": [
        "communication_score",
        "teamwork_score",
        "leadership_score",
    ]
}

# Model Selection
MODELS_TO_USE = [
    "baseline",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "neural_network",
]

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
API_DEBUG = True

# Database
DATABASE_URL = "sqlite:///placemux.db"

# Metrics Thresholds
PRECISION_THRESHOLD = 0.75
RECALL_THRESHOLD = 0.70
FPR_THRESHOLD = 0.15
MATCH_SCORE_THRESHOLD = 0.60

# Logging
LOG_LEVEL = "INFO"
