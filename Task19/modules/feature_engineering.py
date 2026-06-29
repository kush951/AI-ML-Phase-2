"""
Feature Engineering Module
Extracts and transforms features from student-job data for ML models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List
import json

class FeatureEngineer:
    """Extract and engineer features for student-job matching"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalized_scaler = MinMaxScaler()
        self.feature_names = []
        self.fitted = False
    
    def extract_features(self, students: pd.DataFrame, jobs: pd.DataFrame, 
                        matches: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features from student-job pairs
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: List of feature names
        """
        
        features_list = []
        labels = []
        
        # Create lookup dictionaries for O(1) access
        student_dict = {row['student_id']: row for _, row in students.iterrows()}
        job_dict = {row['job_id']: row for _, row in jobs.iterrows()}
        
        for _, match in matches.iterrows():
            student_id = match['student_id']
            job_id = match['job_id']
            
            if student_id not in student_dict or job_id not in job_dict:
                continue
            
            student = student_dict[student_id]
            job = job_dict[job_id]
            
            # Extract features
            features = self._extract_pair_features(student, job, match)
            features_list.append(features)
            labels.append(match['is_good_match'])
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Set feature names if not already set
        if not self.feature_names:
            self.feature_names = self._get_feature_names()
        
        return X, y, self.feature_names
    
    def _extract_pair_features(self, student: pd.Series, job: pd.Series, 
                               match: pd.Series) -> np.ndarray:
        """Extract features for a single student-job pair"""
        
        features = []
        
        # 1. SKILL FEATURES (4 features)
        student_skills = set(student['skills'].keys())
        required_skills = set(job['required_skills'])
        
        # Skill overlap ratio
        skill_overlap = len(student_skills & required_skills)
        skill_match_ratio = skill_overlap / len(required_skills) if required_skills else 0
        features.append(skill_match_ratio)
        
        # Average skill confidence for matching skills
        if skill_overlap > 0:
            matching_skills = student_skills & required_skills
            avg_skill_confidence = np.mean([student['skills'][s] for s in matching_skills])
        else:
            avg_skill_confidence = 0
        features.append(avg_skill_confidence)
        
        # Student's minimum skill confidence (quality indicator)
        if student['skills']:
            min_skill_conf = min(student['skills'].values())
        else:
            min_skill_conf = 0
        features.append(min_skill_conf)
        
        # Number of extra skills student has
        extra_skills = len(student_skills - required_skills)
        features.append(extra_skills / max(len(student_skills), 1))
        
        # 2. EXPERIENCE FEATURES (3 features)
        # Experience match (binary)
        exp_match = 1.0 if student['years_of_experience'] >= job['required_years_experience'] else 0.7
        features.append(exp_match)
        
        # Experience difference
        exp_diff = student['years_of_experience'] - job['required_years_experience']
        features.append(np.tanh(exp_diff / 3))  # Normalize to [-1, 1]
        
        # Internship experience (as proxy for practical skills)
        features.append(min(student['internship_experience'] / 3, 1.0))
        
        # 3. ACADEMIC FEATURES (3 features)
        # GPA match
        gpa_match = 1.0 if student['gpa'] >= job['minimum_gpa'] else 0.7
        features.append(gpa_match)
        
        # GPA difference
        gpa_diff = student['gpa'] - job['minimum_gpa']
        features.append(np.tanh(gpa_diff / 1.0))
        
        # Projects count (indicates practical experience)
        features.append(min(student['projects_count'] / 5, 1.0))
        
        # 4. LOCATION FEATURES (2 features)
        # Location match (binary)
        location_match = 1.0 if (student['location'] == job['location'] or 
                                student['location'] == 'Remote' or 
                                job['location'] == 'Remote') else 0.0
        features.append(location_match)
        
        # Is student remote-ready (in remote location)
        is_remote = 1.0 if student['location'] == 'Remote' else 0.0
        features.append(is_remote)
        
        # 5. JOB QUALITY FEATURES (2 features)
        # Job quality score
        features.append(job['quality_score'])
        
        # Job activity (applications count indicates popularity)
        job_activity = min(job['applications_count'] / 100, 1.0)
        features.append(job_activity)
        
        # 6. STUDENT QUALITY FEATURES (2 features)
        # Student verification status
        features.append(1.0 if student['verified'] else 0.0)
        
        # Overall student quality (avg skill + gpa normalized)
        avg_skill = np.mean(list(student['skills'].values())) if student['skills'] else 0
        student_quality = (avg_skill * 0.6 + (student['gpa'] / 4) * 0.4)
        features.append(student_quality)
        
        # 7. RECENCY FEATURES (2 features)
        # Days since job posted (fresher jobs are better)
        from datetime import datetime
        days_since_posted = (datetime.now().date() - job['posted_date']).days
        features.append(np.exp(-days_since_posted / 30))  # Decay over 30 days
        
        # Days since student onboarded (fresher students might be more motivated)
        days_since_onboard = (datetime.now().date() - student['onboarding_date']).days
        features.append(np.exp(-days_since_onboard / 60))  # Decay over 60 days
        
        return np.array(features)
    
    def _get_feature_names(self) -> List[str]:
        """Define feature names for interpretability"""
        return [
            # Skill features
            'skill_match_ratio', 'avg_matching_skill_confidence', 'min_skill_confidence', 'extra_skills_ratio',
            # Experience features
            'experience_match', 'experience_difference_normalized', 'internship_experience_ratio',
            # Academic features
            'gpa_match', 'gpa_difference_normalized', 'projects_count_ratio',
            # Location features
            'location_match', 'student_is_remote',
            # Job quality features
            'job_quality_score', 'job_activity_ratio',
            # Student quality features
            'student_verified', 'student_overall_quality',
            # Recency features
            'job_freshness', 'student_freshness'
        ]
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit the scaler on training data"""
        self.scaler.fit(X)
        self.fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features"""
        self.fit_scaler(X)
        return self.transform(X)
    
    def get_feature_importance_explanation(self, feature_names: List[str], 
                                         importances: np.ndarray) -> str:
        """Generate human-readable explanation of feature importance"""
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1][:5]
        
        explanation = "Key factors in this match:\n"
        for idx in sorted_indices:
            feature_name = feature_names[idx]
            importance = importances[idx]
            # Convert feature name to readable format
            readable_name = feature_name.replace('_', ' ').title()
            explanation += f"• {readable_name}: {importance:.1%}\n"
        
        return explanation
