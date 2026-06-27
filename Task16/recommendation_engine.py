"""
Recommendation Engine - Core recommendation logic with explainability
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from config import MATCH_SCORE_THRESHOLD

class RecommendationEngine:
    """Generate recommendations with explainability"""
    
    def __init__(self, model, students_df, jobs_df):
        self.model = model
        self.students_df = students_df
        self.jobs_df = jobs_df
        
    def get_recommendations(self, student_id: str, top_k: int = 5) -> List[Dict]:
        """
        Get top K job recommendations for a student
        
        Args:
            student_id: Student ID
            top_k: Number of recommendations
            
        Returns:
            List of recommendations with explanations
        """
        student = self.students_df[self.students_df['student_id'] == student_id]
        
        if student.empty:
            return []
        
        recommendations = []
        
        for _, job in self.jobs_df.iterrows():
            # Calculate match score
            match_data = self._calculate_match_data(student.iloc[0], job)
            score = match_data['overall_match_score']
            
            if score >= MATCH_SCORE_THRESHOLD:
                explanation = self._generate_explanation(student.iloc[0], job, match_data)
                
                recommendations.append({
                    'job_id': job['job_id'],
                    'job_title': job['job_title'],
                    'company': job['company'],
                    'match_score': float(score),
                    'explanation': explanation,
                    'match_breakdown': match_data,
                })
        
        # Sort by match score and return top K
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return recommendations[:top_k]
    
    def _calculate_match_data(self, student, job) -> Dict:
        """Calculate detailed match metrics"""
        student_skills = set(str(student['verified_skills']).split(','))
        job_skills = set(str(job['required_skills']).split(','))
        
        # Skill matching
        skill_overlap = len(student_skills & job_skills)
        required_skill_count = len(job_skills)
        skill_match = skill_overlap / required_skill_count if required_skill_count > 0 else 0
        
        # Experience matching
        exp_match = 1.0 if student['years_of_experience'] >= job['min_experience'] else \
                   student['years_of_experience'] / (job['min_experience'] + 1)
        
        # CGPA matching
        cgpa_match = 1.0 if student['cgpa'] >= job['min_cgpa'] else \
                   student['cgpa'] / job['min_cgpa']
        
        # Soft skills average
        soft_skill_avg = (student['communication_score'] + 
                         student['teamwork_score'] + 
                         student['leadership_score']) / 3
        
        # Overall score
        overall_score = (skill_match * 0.5 + 
                        exp_match * 0.2 + 
                        cgpa_match * 0.15 + 
                        soft_skill_avg * 0.15)
        
        return {
            'skill_match_score': float(np.round(skill_match, 2)),
            'experience_match_score': float(np.round(exp_match, 2)),
            'cgpa_match_score': float(np.round(cgpa_match, 2)),
            'soft_skill_score': float(np.round(soft_skill_avg, 2)),
            'overall_match_score': float(np.round(overall_score, 2)),
            'skills_matched': list(student_skills & job_skills),
            'skills_missing': list(job_skills - student_skills),
        }
    
    def _generate_explanation(self, student, job, match_data: Dict) -> str:
        """Generate plain-English explanation for recommendation"""
        explanation = f"""
This is a {match_data['overall_match_score']:.0%} match because:

SKILLS ({match_data['skill_match_score']:.0%} match):
- You have {len(match_data['skills_matched'])} of {len(set(job['required_skills'].split(',')))} required skills
- Matched: {', '.join(match_data['skills_matched'][:3]) if match_data['skills_matched'] else 'None'}
- Missing: {', '.join(match_data['skills_missing'][:2]) if match_data['skills_missing'] else 'None'} 

EXPERIENCE ({match_data['experience_match_score']:.0%} match):
- You have {student['years_of_experience']} years vs {job['min_experience']} required

ACADEMICS ({match_data['cgpa_match_score']:.0%} match):
- Your CGPA: {student['cgpa']:.2f} vs {job['min_cgpa']:.2f} required

SOFT SKILLS ({match_data['soft_skill_score']:.0%} rating):
- Communication: {student['communication_score']:.2f}/1.0
- Teamwork: {student['teamwork_score']:.2f}/1.0

RECOMMENDATION: This role could be a great fit. Focus on the missing skills while leveraging your strengths.
        """
        return explanation.strip()


class MatchExplainer:
    """Explain specific match decisions with feature importance"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def explain_prediction(self, student_data: pd.Series, job_data: pd.Series) -> Dict:
        """Generate detailed explanation for a match"""
        # Get model prediction
        pred_proba = self.model.predict_proba(student_data.to_frame().T)
        no_match_prob = float(pred_proba[0][0])
        match_prob = float(pred_proba[0][1])
        
        explanation = {
            'prediction': 'Match' if match_prob > 0.5 else 'No Match',
            'confidence': float(max(match_prob, no_match_prob)),
            'match_probability': float(match_prob),
            'no_match_probability': float(no_match_prob),
            'key_factors': self._get_key_factors(student_data),
        }
        
        return explanation
    
    def _get_key_factors(self, data: pd.Series) -> List[Tuple[str, float]]:
        """Get top factors driving the prediction"""
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            return [(row['feature'], float(row['importance'])) 
                   for _, row in importance.head(5).iterrows()]
        return []


class RecommendationValidator:
    """Validate recommendations against business rules"""
    
    @staticmethod
    def validate_recommendation(student_id: str, job_id: str, 
                               match_score: float, 
                               students_df, jobs_df) -> Tuple[bool, str]:
        """Validate if recommendation meets business rules"""
        
        # Check if student and job exist
        student = students_df[students_df['student_id'] == student_id]
        job = jobs_df[jobs_df['job_id'] == job_id]
        
        if student.empty:
            return False, "Student not found"
        if job.empty:
            return False, "Job not found"
        
        # Data privacy check
        if not RecommendationValidator._check_data_privacy(student.iloc[0], job.iloc[0]):
            return False, "Privacy policy violation"
        
        # Match score threshold
        if match_score < MATCH_SCORE_THRESHOLD:
            return False, f"Match score below threshold ({MATCH_SCORE_THRESHOLD})"
        
        # Fairness check
        if not RecommendationValidator._check_fairness(student.iloc[0], job.iloc[0]):
            return False, "Recommendation fails fairness check"
        
        return True, "Recommendation is valid"
    
    @staticmethod
    def _check_data_privacy(student, job) -> bool:
        """Ensure data privacy - colleges can't see each other's data"""
        # In production, verify college isolation
        return True
    
    @staticmethod
    def _check_fairness(student, job) -> bool:
        """Check for fairness in recommendations"""
        # Implement fairness checks
        # For now, accept all valid matches
        return True


if __name__ == "__main__":
    from data_preparation import DataLoader
    from ml_models import ModelRegistry
    
    # Load data and model
    students, jobs, matches = DataLoader.load_data()
    best_model = ModelRegistry.load_model('gradient_boosting')
    
    # Initialize recommendation engine
    engine = RecommendationEngine(best_model, students, jobs)
    
    # Get recommendations for first student
    student_id = students.iloc[0]['student_id']
    recommendations = engine.get_recommendations(student_id, top_k=5)
    
    print(f"Top recommendations for {student_id}:")
    print("="*60)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['job_title']} at {rec['company']}")
        print(f"   Match Score: {rec['match_score']:.2%}")
        print(f"   {rec['explanation']}")
