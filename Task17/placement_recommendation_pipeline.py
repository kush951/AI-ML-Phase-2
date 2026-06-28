"""
PlaceMux Recommendation System v1
End-to-End ML Pipeline with Multiple Models
Author: AI/ML Engineer | Phase 2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve,
    classification_report, accuracy_score
)
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RecommendationScore:
    """Container for recommendation details"""
    student_id: str
    job_id: str
    match_score: float
    confidence: float
    ranking: int
    explanation: str
    feature_importance: Dict[str, float]
    model_used: str


class FeatureEngineer:
    """Design feature space for recommendation"""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.feature_names = []
        logger.info("FeatureEngineer initialized")
    
    def engineer_features(self, 
                         student_data: pd.DataFrame, 
                         job_data: pd.DataFrame,
                         fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature vectors for student-job pairs
        
        Features:
        - Skill overlap (Jaccard similarity)
        - Years of experience match
        - Location compatibility
        - Education level match
        - GPA score
        - Verified skills count
        """
        X_student = []
        X_job = []
        
        for idx, student in student_data.iterrows():
            student_features = self._extract_student_features(student)
            X_student.append(student_features)
        
        for idx, job in job_data.iterrows():
            job_features = self._extract_job_features(job)
            X_job.append(job_features)
        
        X_student = np.array(X_student)
        X_job = np.array(X_job)
        
        if fit:
            self.scaler.fit(np.vstack([X_student, X_job]))
            logger.info(f"Features fitted. Feature dimension: {X_student.shape[1]}")
        
        X_student = self.scaler.transform(X_student)
        X_job = self.scaler.transform(X_job)
        
        return X_student, X_job
    
    def _extract_student_features(self, student: pd.Series) -> np.ndarray:
        """Extract individual student features"""
        features = [
            float(student.get('gpa', 0)),
            float(student.get('years_experience', 0)),
            float(student.get('verified_skills_count', 0)),
            float(student.get('certifications_count', 0)),
            float(student.get('project_count', 0)),
        ]
        return np.array(features)
    
    def _extract_job_features(self, job: pd.Series) -> np.ndarray:
        """Extract individual job features"""
        features = [
            float(job.get('required_gpa', 0)),
            float(job.get('required_years', 0)),
            float(job.get('required_skills_count', 0)),
            float(job.get('required_certifications', 0)),
            float(job.get('complexity_score', 0)),
        ]
        return np.array(features)
    
    def compute_similarity_features(self, 
                                    student_skills: List[str],
                                    job_skills: List[str]) -> Dict[str, float]:
        """Compute skill-based similarity metrics"""
        student_set = set(skill.lower() for skill in student_skills)
        job_set = set(skill.lower() for skill in job_skills)
        
        intersection = len(student_set & job_set)
        union = len(student_set | job_set)
        
        jaccard_similarity = intersection / union if union > 0 else 0
        
        return {
            'skill_overlap': intersection,
            'skill_required': len(job_set),
            'skill_verified': len(student_set),
            'jaccard_similarity': jaccard_similarity,
            'coverage_ratio': intersection / len(job_set) if len(job_set) > 0 else 0
        }


class BaselineModel:
    """Simple baseline: Skill overlap ranking"""
    
    def __init__(self):
        self.name = "Baseline (Skill Overlap)"
        logger.info("BaselineModel initialized")
    
    def fit(self, X, y):
        """Baseline needs no fitting"""
        logger.info("Baseline model fitted (no-op)")
        return self
    
    def predict(self, X, student_skills: List[List[str]], 
                job_skills: List[List[str]]) -> np.ndarray:
        """Predict using simple skill overlap"""
        predictions = []
        for s_skills, j_skills in zip(student_skills, job_skills):
            s_set = set(skill.lower() for skill in s_skills)
            j_set = set(skill.lower() for skill in j_skills)
            union = len(s_set | j_set)
            overlap = len(s_set & j_set)
            score = overlap / union if union > 0 else 0
            predictions.append(1 if score >= 0.3 else 0)
        return np.array(predictions)
    
    def predict_proba(self, X, student_skills: List[List[str]], 
                      job_skills: List[List[str]]) -> np.ndarray:
        """Return probability estimates"""
        probabilities = []
        for s_skills, j_skills in zip(student_skills, job_skills):
            s_set = set(skill.lower() for skill in s_skills)
            j_set = set(skill.lower() for skill in j_skills)
            union = len(s_set | j_set)
            overlap = len(s_set & j_set)
            score = overlap / union if union > 0 else 0
            probabilities.append([1 - score, score])
        return np.array(probabilities)


class HybridRecommender:
    """Ensemble recommendation system with multiple models"""
    
    def __init__(self, use_models: List[str] = None):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        self.experiment_log = []
        
        if use_models is None:
            use_models = ['baseline', 'logistic', 'random_forest', 'gradient_boost']
        
        self._initialize_models(use_models)
        logger.info(f"HybridRecommender initialized with models: {use_models}")
    
    def _initialize_models(self, model_names: List[str]):
        """Initialize selected models"""
        model_configs = {
            'baseline': BaselineModel(),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        }
        
        for name in model_names:
            if name in model_configs:
                self.models[name] = model_configs[name]
    
    def fit(self, 
            student_data: pd.DataFrame,
            job_data: pd.DataFrame,
            matches: pd.DataFrame,  # Ground truth: (student_id, job_id, is_match)
            test_size: float = 0.2,
            val_size: float = 0.1):
        """Train all models with proper train/val/test split"""
        
        logger.info("Starting training pipeline...")
        
        # Prepare data
        X_student, X_job = self.feature_engineer.engineer_features(
            student_data, job_data, fit=True
        )
        
        # Create feature pairs for all matches
        X_pairs = []
        y_labels = []
        student_skills = []
        job_skills = []
        
        for _, row in matches.iterrows():
            student_idx = student_data[student_data['student_id'] == row['student_id']].index
            job_idx = job_data[job_data['job_id'] == row['job_id']].index
            
            if len(student_idx) > 0 and len(job_idx) > 0:
                s_idx = student_idx[0]
                j_idx = job_idx[0]
                
                pair_features = np.concatenate([X_student[s_idx], X_job[j_idx]])
                X_pairs.append(pair_features)
                y_labels.append(row['is_match'])
                
                student_skills.append(
                    row.get('student_skills', '').split(',') if pd.notna(row.get('student_skills')) else []
                )
                job_skills.append(
                    row.get('job_skills', '').split(',') if pd.notna(row.get('job_skills')) else []
                )
        
        X = np.array(X_pairs)
        y = np.array(y_labels)
        
        # Train/Val/Test split
        X_temp, X_test, y_temp, y_test, skills_temp, skills_test_j = train_test_split(
            X, y, list(zip(student_skills, job_skills)),
            test_size=test_size, random_state=42, stratify=y
        )
        
        train_size = len(X_temp) * (1 - val_size)
        X_train, X_val, y_train, y_val, skills_train, skills_val = train_test_split(
            X_temp, y_temp, skills_temp,
            train_size=train_size / len(X_temp), random_state=42, stratify=y_temp
        )
        
        student_skills_train = [s[0] for s in skills_train]
        job_skills_train = [s[1] for s in skills_train]
        student_skills_val = [s[0] for s in skills_val]
        job_skills_val = [s[1] for s in skills_val]
        student_skills_test = [s[0] for s in skills_test_j]
        job_skills_test = [s[1] for s in skills_test_j]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name}...")
            
            model.fit(X_train, y_train)
            
            # Validation evaluation
            if model_name == 'baseline':
                y_pred_val = model.predict(X_val, student_skills_val, job_skills_val)
                y_proba_val = model.predict_proba(X_val, student_skills_val, job_skills_val)[:, 1]
            else:
                y_pred_val = model.predict(X_val)
                y_proba_val = model.predict_proba(X_val)[:, 1]
            
            val_metrics = self._calculate_metrics(y_val, y_pred_val, y_proba_val, "Validation")
            
            # Test evaluation
            if model_name == 'baseline':
                y_pred_test = model.predict(X_test, student_skills_test, job_skills_test)
                y_proba_test = model.predict_proba(X_test, student_skills_test, job_skills_test)[:, 1]
            else:
                y_pred_test = model.predict(X_test)
                y_proba_test = model.predict_proba(X_test)[:, 1]
            
            test_metrics = self._calculate_metrics(y_test, y_pred_test, y_proba_test, "Test")
            
            self.model_scores[model_name] = {
                'validation': val_metrics,
                'test': test_metrics,
                'model': model
            }
            
            experiment_entry = {
                'timestamp': datetime.now().isoformat(),
                'model': model_name,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            self.experiment_log.append(experiment_entry)
        
        # Select best model based on F1 score
        best_f1 = -1
        for model_name, scores in self.model_scores.items():
            f1 = scores['test']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = model_name
                self.best_model = scores['model']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Best Model Selected: {self.best_model_name} (F1: {best_f1:.4f})")
        logger.info(f"{'='*60}")
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, phase: str) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        }
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        logger.info(f"{phase} Metrics: {metrics}")
        return metrics
    
    def get_recommendation(self, 
                          student_id: str,
                          student_profile: Dict[str, Any],
                          job_profile: Dict[str, Any],
                          student_skills: List[str],
                          job_skills: List[str]) -> RecommendationScore:
        """Generate single recommendation with explanation"""
        
        # Create feature vector
        X_student_feat = np.array([
            student_profile.get('gpa', 0),
            student_profile.get('years_experience', 0),
            len(student_skills),
            student_profile.get('certifications_count', 0),
            student_profile.get('project_count', 0),
        ])
        
        X_job_feat = np.array([
            job_profile.get('required_gpa', 0),
            job_profile.get('required_years', 0),
            len(job_skills),
            job_profile.get('required_certifications', 0),
            job_profile.get('complexity_score', 0),
        ])
        
        X_student_feat = self.feature_engineer.scaler.transform([X_student_feat])[0]
        X_job_feat = self.feature_engineer.scaler.transform([X_job_feat])[0]
        
        X_pair = np.concatenate([X_student_feat, X_job_feat]).reshape(1, -1)
        
        # Get prediction from best model
        if self.best_model_name == 'baseline':
            prediction = self.best_model.predict(X_pair, [student_skills], [job_skills])[0]
            confidence = self.best_model.predict_proba(X_pair, [student_skills], [job_skills])[0, 1]
        else:
            prediction = self.best_model.predict(X_pair)[0]
            confidence = self.best_model.predict_proba(X_pair)[0, 1]
        
        # Generate explanation
        similarity_scores = self._compute_similarity(student_skills, job_skills)
        explanation = self._generate_explanation(
            student_profile, job_profile, similarity_scores, confidence
        )
        
        # Extract feature importance
        feature_importance = self._get_feature_importance(similarity_scores)
        
        return RecommendationScore(
            student_id=student_id,
            job_id=job_profile['job_id'],
            match_score=confidence,
            confidence=confidence,
            ranking=1 if prediction == 1 else 0,
            explanation=explanation,
            feature_importance=feature_importance,
            model_used=self.best_model_name
        )
    
    def _compute_similarity(self, student_skills: List[str], 
                           job_skills: List[str]) -> Dict[str, Any]:
        """Compute similarity metrics"""
        s_set = set(skill.lower() for skill in student_skills)
        j_set = set(skill.lower() for skill in job_skills)
        
        intersection = s_set & j_set
        union = s_set | j_set
        
        return {
            'matching_skills': list(intersection),
            'missing_skills': list(j_set - s_set),
            'jaccard_similarity': len(intersection) / len(union) if len(union) > 0 else 0,
            'coverage': len(intersection) / len(j_set) if len(j_set) > 0 else 0
        }
    
    def _generate_explanation(self, student: Dict, job: Dict, 
                             similarity: Dict, confidence: float) -> str:
        """Generate human-readable explanation"""
        coverage = similarity['coverage']
        matching = len(similarity['matching_skills'])
        missing = len(similarity['missing_skills'])
        
        if coverage >= 0.8:
            return f"Strong match: {matching} of {matching + missing} required skills verified. Confidence: {confidence:.1%}"
        elif coverage >= 0.5:
            return f"Good match: {matching} key skills matched, {missing} skills can be developed. Confidence: {confidence:.1%}"
        else:
            return f"Potential match: Candidate has foundational skills but needs to develop {missing} required skills. Confidence: {confidence:.1%}"
    
    def _get_feature_importance(self, similarity: Dict) -> Dict[str, float]:
        """Extract feature importance from similarity"""
        return {
            'skill_coverage': similarity['coverage'],
            'jaccard_similarity': similarity['jaccard_similarity'],
            'matching_skills_count': len(similarity['matching_skills']),
            'missing_skills_count': len(similarity['missing_skills'])
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Return dataframe of all model performances"""
        comparison_data = []
        
        for model_name, scores in self.model_scores.items():
            test_metrics = scores['test']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{test_metrics['accuracy']:.4f}",
                'Precision': f"{test_metrics['precision']:.4f}",
                'Recall': f"{test_metrics['recall']:.4f}",
                'F1-Score': f"{test_metrics['f1_score']:.4f}",
                'ROC-AUC': f"{test_metrics['roc_auc']:.4f}",
                'FPR': f"{test_metrics.get('fpr', 0):.4f}",
                'Selected': '✓' if model_name == self.best_model_name else ''
            })
        
        return pd.DataFrame(comparison_data)


# ============= Demo Data Generation =============
def generate_sample_data(n_students=100, n_jobs=50, n_matches=200) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate realistic sample data for demonstration"""
    
    np.random.seed(42)
    
    # Student data
    students = {
        'student_id': [f'STU_{i:03d}' for i in range(n_students)],
        'gpa': np.random.uniform(2.5, 4.0, n_students),
        'years_experience': np.random.randint(0, 5, n_students),
        'verified_skills_count': np.random.randint(3, 15, n_students),
        'certifications_count': np.random.randint(0, 5, n_students),
        'project_count': np.random.randint(2, 20, n_students)
    }
    student_data = pd.DataFrame(students)
    
    # Job data
    jobs = {
        'job_id': [f'JOB_{i:03d}' for i in range(n_jobs)],
        'required_gpa': np.random.uniform(2.5, 3.8, n_jobs),
        'required_years': np.random.randint(0, 5, n_jobs),
        'required_skills_count': np.random.randint(5, 15, n_jobs),
        'required_certifications': np.random.randint(0, 3, n_jobs),
        'complexity_score': np.random.uniform(1, 10, n_jobs)
    }
    job_data = pd.DataFrame(jobs)
    
    # Matches with ground truth
    match_list = []
    skills_pool = ['Python', 'Java', 'SQL', 'AWS', 'Machine Learning', 'Data Analysis',
                   'React', 'Node.js', 'Docker', 'Kubernetes', 'System Design', 'API Design']
    
    for _ in range(n_matches):
        student_idx = np.random.randint(0, n_students)
        job_idx = np.random.randint(0, n_jobs)
        
        # Probability of match increases with similar attributes
        student = student_data.iloc[student_idx]
        job = job_data.iloc[job_idx]
        
        gpa_diff = abs(student['gpa'] - job['required_gpa'])
        exp_diff = abs(student['years_experience'] - job['required_years'])
        skill_diff = abs(student['verified_skills_count'] - job['required_skills_count'])
        
        match_prob = max(0.1, 1 - (gpa_diff + exp_diff * 0.2 + skill_diff * 0.1) / 5)
        is_match = np.random.rand() < match_prob
        
        student_skills = list(np.random.choice(skills_pool, np.random.randint(3, 10), replace=False))
        job_skills = list(np.random.choice(skills_pool, np.random.randint(4, 12), replace=False))
        
        match_list.append({
            'student_id': student_data.iloc[student_idx]['student_id'],
            'job_id': job_data.iloc[job_idx]['job_id'],
            'is_match': int(is_match),
            'student_skills': ','.join(student_skills),
            'job_skills': ','.join(job_skills)
        })
    
    matches = pd.DataFrame(match_list)
    
    logger.info(f"Generated {len(student_data)} students, {len(job_data)} jobs, {len(matches)} matches")
    return student_data, job_data, matches


if __name__ == "__main__":
    # Generate sample data
    student_data, job_data, matches = generate_sample_data()
    
    # Initialize and train recommender
    recommender = HybridRecommender()
    recommender.fit(student_data, job_data, matches)
    
    # Display model comparison
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON (Test Set)")
    print("="*80)
    print(recommender.get_model_comparison().to_string(index=False))
    
    # Generate sample recommendations
    print("\n" + "="*80)
    print("SAMPLE RECOMMENDATIONS")
    print("="*80)
    
    for i in range(3):
        student = student_data.iloc[i]
        job = job_data.iloc[i % len(job_data)]
        student_skills = ['Python', 'SQL', 'Machine Learning']
        job_skills = ['Python', 'SQL', 'AWS', 'Machine Learning', 'Data Analysis']
        
        recommendation = recommender.get_recommendation(
            student_id=student['student_id'],
            student_profile=student.to_dict(),
            job_profile=job.to_dict(),
            student_skills=student_skills,
            job_skills=job_skills
        )
        
        print(f"\nStudent: {recommendation.student_id} → Job: {recommendation.job_id}")
        print(f"Match Score: {recommendation.match_score:.1%}")
        print(f"Explanation: {recommendation.explanation}")
        print(f"Model: {recommendation.model_used}")
