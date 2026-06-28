"""
Explainability Engine - Provides plain-English explanations for predictions
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Generate human-readable explanations for job match predictions
    
    Implements:
    - Feature extraction and engineering
    - SHAP-like local explanations
    - Feature importance ranking
    - Natural language explanation generation
    """
    
    def __init__(self, model_trainer):
        """
        Initialize explainability engine
        
        Args:
            model_trainer: Trained ModelTrainer instance
        """
        self.model_trainer = model_trainer
        self.feature_names = model_trainer.feature_names
        
    def extract_features(self, student, job) -> np.ndarray:
        """
        Extract features from student and job profiles
        
        Feature Engineering:
        1. Skill Match Features
           - Skill overlap ratio
           - Verified skill coverage
           - Skill gap count
        
        2. Experience Features
           - Years match
           - Experience gap
        
        3. Performance Features
           - GPA match
           - Average skill score
        
        Args:
            student: StudentProfile
            job: JobDescription
        
        Returns:
            Feature vector (numpy array)
        """
        features = []
        feature_names_used = []
        
        # ===== Skill Match Features =====
        student_skills = set(student.verified_skills)
        required_skills = set(job.required_skills)
        
        # Feature 1: Skill overlap ratio
        if len(required_skills) > 0:
            skill_overlap = len(student_skills.intersection(required_skills)) / len(required_skills)
        else:
            skill_overlap = 0.0
        features.append(skill_overlap)
        feature_names_used.append("skill_overlap_ratio")
        
        # Feature 2: Skill coverage (how many required skills student has)
        required_skills_covered = len(student_skills.intersection(required_skills))
        features.append(required_skills_covered)
        feature_names_used.append("required_skills_covered")
        
        # Feature 3: Skill gap (missing required skills)
        skill_gap = len(required_skills) - required_skills_covered
        features.append(skill_gap)
        feature_names_used.append("skill_gap")
        
        # Feature 4: Average score of matching skills
        matching_skills = student_skills.intersection(required_skills)
        if matching_skills:
            avg_matching_score = np.mean([
                student.skill_scores.get(skill, 0)
                for skill in matching_skills
            ])
        else:
            avg_matching_score = 0.0
        features.append(avg_matching_score)
        feature_names_used.append("avg_matching_skill_score")
        
        # Feature 5: Average score of all student skills
        avg_all_skills = np.mean([
            score for score in student.skill_scores.values()
        ]) if student.skill_scores else 0.0
        features.append(avg_all_skills)
        feature_names_used.append("avg_all_skill_scores")
        
        # ===== Experience Features =====
        
        # Feature 6: Experience gap
        exp_gap = max(0, job.required_exp_years - student.experience_years)
        features.append(exp_gap)
        feature_names_used.append("experience_gap")
        
        # Feature 7: Excess experience (useful - student brings more)
        excess_exp = max(0, student.experience_years - job.required_exp_years)
        features.append(excess_exp)
        feature_names_used.append("excess_experience")
        
        # Feature 8: Experience match (perfect is 1.0)
        if job.required_exp_years > 0:
            exp_match = min(1.0, student.experience_years / job.required_exp_years)
        else:
            exp_match = 1.0
        features.append(exp_match)
        feature_names_used.append("experience_match_ratio")
        
        # ===== Performance Features =====
        
        # Feature 9: GPA score (normalized to 0-1)
        gpa_normalized = min(1.0, student.gpa / 4.0)
        features.append(gpa_normalized)
        feature_names_used.append("gpa_normalized")
        
        # Feature 10: Total verified skills
        features.append(len(student_skills))
        feature_names_used.append("total_verified_skills")
        
        # Feature 11: Required skills count
        features.append(len(required_skills))
        feature_names_used.append("required_skills_count")
        
        # Feature 12: College match (same college = 1, different = 0)
        college_match = 1.0 if student.college_id == job.college_id else 0.0
        features.append(college_match)
        feature_names_used.append("college_match")
        
        # Convert to numpy array
        feature_vector = np.array(features, dtype=np.float32)
        
        # Store for reference
        self._last_features = feature_vector
        self._last_feature_names = feature_names_used
        
        return feature_vector
    
    def explain_prediction(
        self,
        features: np.ndarray,
        student,
        job,
        model_name: str,
        score: float
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate human-readable explanation for prediction
        
        Args:
            features: Feature vector
            student: Student profile
            job: Job description
            model_name: Name of model used
            score: Prediction score (0-1)
        
        Returns:
            Tuple of (explanation_text, top_factors)
        """
        # Get feature importance from the model
        feature_importance = self.model_trainer.get_feature_importance(model_name)
        
        # Calculate normalized features for interpretation
        feature_values = {}
        for name, value in zip(self._last_feature_names, features):
            feature_values[name] = value
        
        # Determine top contributing factors
        top_factors = []
        for feat_name in self._last_feature_names:
            if feat_name in feature_importance:
                top_factors.append({
                    "feature": feat_name,
                    "value": float(feature_values.get(feat_name, 0)),
                    "importance": float(feature_importance[feat_name]),
                    "impact": "positive" if feature_values.get(feat_name, 0) > 0.5 else "negative"
                })
        
        # Sort by importance
        top_factors = sorted(
            top_factors,
            key=lambda x: x["importance"],
            reverse=True
        )[:5]  # Top 5 factors
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            student=student,
            job=job,
            score=score,
            feature_values=feature_values,
            top_factors=top_factors
        )
        
        return explanation, top_factors
    
    def _generate_explanation(
        self,
        student,
        job,
        score: float,
        feature_values: Dict[str, float],
        top_factors: List[Dict[str, Any]]
    ) -> str:
        """
        Generate natural language explanation
        """
        explanation_parts = []
        
        # Header
        match_quality = "Strong" if score >= 0.7 else ("Moderate" if score >= 0.4 else "Weak")
        explanation_parts.append(
            f"{match_quality} match ({score*100:.1f}%) between {student.name} and {job.title} at {job.company}."
        )
        
        # Skill analysis
        skill_overlap = feature_values.get("skill_overlap_ratio", 0)
        required_covered = feature_values.get("required_skills_covered", 0)
        
        if skill_overlap > 0.75:
            explanation_parts.append(
                f"Strong skill alignment: {int(required_covered)} of the {int(feature_values.get('required_skills_count', 0))} "
                f"required skills are verified and matched."
            )
        elif skill_overlap > 0.5:
            explanation_parts.append(
                f"Moderate skill match: {int(required_covered)} required skills are present, "
                f"but some gaps remain to cover all requirements."
            )
        else:
            skill_gap = feature_values.get("skill_gap", 0)
            explanation_parts.append(
                f"Limited skill overlap: {int(skill_gap)} required skills are missing. "
                f"This represents the main gap for this match."
            )
        
        # Experience analysis
        exp_match = feature_values.get("experience_match_ratio", 0)
        if exp_match >= 1.0:
            explanation_parts.append(
                f"✓ Experience requirement met: {student.experience_years} years matches or exceeds "
                f"the required {int(feature_values.get('required_skills_count', 0))} years."
            )
        elif exp_match >= 0.75:
            explanation_parts.append(
                f"✓ Close experience match: {student.experience_years} years is close to the "
                f"required {int(feature_values.get('required_skills_count', 0))} years."
            )
        else:
            exp_gap = feature_values.get("experience_gap", 0)
            explanation_parts.append(
                f"⚠ Experience gap: {student.experience_years} years vs {exp_gap + student.experience_years} years required. "
                f"This is a notable limitation."
            )
        
        # Performance
        gpa = feature_values.get("gpa_normalized", 0) * 4.0
        if gpa >= 3.5:
            explanation_parts.append(
                f"Strong academic performance with {gpa:.2f} GPA."
            )
        elif gpa >= 3.0:
            explanation_parts.append(
                f"Good academic background with {gpa:.2f} GPA."
            )
        
        # College match
        if feature_values.get("college_match", 0) == 1.0:
            explanation_parts.append(
                f"✓ Same college: Both student and job are from the same institution, "
                f"which can simplify logistics."
            )
        
        # Top factors
        if top_factors:
            factor_names = [f["feature"].replace("_", " ").title() 
                          for f in top_factors[:2]]
            explanation_parts.append(
                f"Top contributing factors: {', '.join(factor_names)}."
            )
        
        # Confidence note
        explanation_parts.append(
            f"This prediction is based on {len(self._last_feature_names)} verified signals "
            f"from the student profile and job requirements."
        )
        
        return " ".join(explanation_parts)
