"""
PlaceMux Intelligence Layer
ML Pipeline for Student-Job Matching, Explainability, and Quality Assurance
"""

__version__ = "1.0.0"
__author__ = "PlaceMux AI/ML Team"
__description__ = "Bulk Onboarding & Recruiter Views Intelligence Layer"

from .data_generator import DataGenerator
from .feature_engineering import FeatureEngineer
from .matching_models import MatchingModelEnsemble, BaselineModel
from .explainability import MatchExplainer, QualityAnalyzer
from .recruiter_views import RecruiterDashboard, AdminPanel
from .pipeline import PlaceMuxPipeline

__all__ = [
    'DataGenerator',
    'FeatureEngineer',
    'MatchingModelEnsemble',
    'BaselineModel',
    'MatchExplainer',
    'QualityAnalyzer',
    'RecruiterDashboard',
    'AdminPanel',
    'PlaceMuxPipeline'
]
