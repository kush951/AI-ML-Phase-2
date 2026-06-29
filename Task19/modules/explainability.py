"""
Explainability Module
Provides clear, human-readable explanations for student-job matches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class MatchExplainer:
    """Generate explainable match explanations"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def explain_match(self, 
                     student: Dict[str, Any],
                     job: Dict[str, Any],
                     features: np.ndarray,
                     prediction: float,
                     probability: float,
                     model_importances: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive match explanation
        
        Args:
            student: Student data dictionary
            job: Job data dictionary
            features: Feature vector for this pair
            prediction: Binary prediction (0 or 1)
            probability: Match probability
            model_importances: Feature importance from model
            
        Returns:
            Dictionary with explanation details
        """
        
        explanation = {
            'match_score': float(probability),
            'is_match': int(prediction) == 1,
            'confidence': float(probability) if prediction == 1 else float(1 - probability),
            'summary': "",
            'strengths': [],
            'gaps': [],
            'recommendations': [],
            'detailed_analysis': {}
        }
        
        # Generate summary
        match_quality = self._assess_match_quality(probability)
        explanation['summary'] = f"This is a {match_quality} match ({probability:.1%} confidence)"
        
        # Analyze skills
        student_skills = set(student['skills'].keys())
        required_skills = set(job['required_skills'])
        skill_overlap = student_skills & required_skills
        missing_skills = required_skills - student_skills
        
        explanation['detailed_analysis']['Skills'] = {
            'match_count': len(skill_overlap),
            'total_required': len(required_skills),
            'matching_skills': list(skill_overlap),
            'missing_skills': list(missing_skills),
            'extra_skills': list(student_skills - required_skills),
            'avg_confidence': self._safe_mean([student['skills'][s] for s in skill_overlap]) if skill_overlap else 0
        }
        
        if len(skill_overlap) >= len(required_skills) * 0.7:
            explanation['strengths'].append(f"✓ Strong skill match: {len(skill_overlap)}/{len(required_skills)} required skills")
        else:
            missing = list(missing_skills)[:3]
            explanation['gaps'].append(f"✗ Missing skills: {', '.join(missing)}")
        
        # Analyze experience
        exp_diff = student['years_of_experience'] - job['required_years_experience']
        explanation['detailed_analysis']['Experience'] = {
            'student_experience': student['years_of_experience'],
            'required_experience': job['required_years_experience'],
            'experience_difference': exp_diff,
            'internship_count': student['internship_experience']
        }
        
        if exp_diff >= 0:
            explanation['strengths'].append(f"✓ Sufficient experience: {student['years_of_experience']} years")
        else:
            explanation['gaps'].append(f"✗ Experience gap: {abs(exp_diff)} years below requirement")
        
        # Analyze academics
        gpa_diff = student['gpa'] - job['minimum_gpa']
        explanation['detailed_analysis']['Academic Profile'] = {
            'student_gpa': student['gpa'],
            'required_gpa': job['minimum_gpa'],
            'gpa_difference': gpa_diff,
            'projects_completed': student['projects_count'],
            'is_verified': student['verified']
        }
        
        if gpa_diff >= 0:
            explanation['strengths'].append(f"✓ Meets GPA requirement: {student['gpa']:.2f}")
        else:
            explanation['gaps'].append(f"✗ GPA below requirement: {job['minimum_gpa']:.2f} needed")
        
        # Analyze location
        location_match = (student['location'] == job['location'] or 
                         student['location'] == 'Remote' or 
                         job['location'] == 'Remote')
        explanation['detailed_analysis']['Location'] = {
            'student_location': student['location'],
            'job_location': job['location'],
            'is_match': location_match
        }
        
        if location_match:
            explanation['strengths'].append(f"✓ Location compatible: {student['location']} for {job['location']}")
        else:
            explanation['gaps'].append(f"✗ Location mismatch: {student['location']} vs {job['location']}")
        
        # Add recommendations
        if missing_skills and len(missing_skills) <= 2:
            skills_str = ', '.join(list(missing_skills))
            explanation['recommendations'].append(f"Consider learning: {skills_str}")
        
        if student['projects_count'] < 2:
            explanation['recommendations'].append("Build more portfolio projects to strengthen application")
        
        if exp_diff < 0:
            explanation['recommendations'].append(f"Gain {abs(exp_diff)} more years of experience")
        
        # Add most important features
        if model_importances:
            top_features = sorted(model_importances.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation['top_factors'] = [
                f"{name.replace('_', ' ').title()}" 
                for name, _ in top_features
            ]
        
        return explanation
    
    def _assess_match_quality(self, probability: float) -> str:
        """Assess match quality level"""
        if probability >= 0.85:
            return "Excellent"
        elif probability >= 0.7:
            return "Strong"
        elif probability >= 0.5:
            return "Moderate"
        elif probability >= 0.3:
            return "Weak"
        else:
            return "Poor"
    
    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        """Calculate mean safely"""
        return np.mean(values) if values else 0.0
    
    def format_explanation(self, explanation: Dict[str, Any]) -> str:
        """Format explanation as human-readable text"""
        
        text = []
        text.append("=" * 70)
        text.append(f"MATCH EXPLANATION")
        text.append("=" * 70)
        
        # Summary
        text.append(f"\n📊 OVERALL ASSESSMENT")
        text.append(f"  Match Score: {explanation['match_score']:.1%}")
        text.append(f"  Confidence: {explanation['confidence']:.1%}")
        text.append(f"  Summary: {explanation['summary']}\n")
        
        # Strengths
        if explanation['strengths']:
            text.append("💪 STRENGTHS:")
            for strength in explanation['strengths']:
                text.append(f"  {strength}")
            text.append("")
        
        # Gaps
        if explanation['gaps']:
            text.append("⚠️  GAPS:")
            for gap in explanation['gaps']:
                text.append(f"  {gap}")
            text.append("")
        
        # Detailed Analysis
        text.append("🔍 DETAILED ANALYSIS:\n")
        for category, details in explanation['detailed_analysis'].items():
            text.append(f"  {category}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, list):
                        text.append(f"    • {key}: {', '.join(value) if value else 'None'}")
                    elif isinstance(value, float):
                        text.append(f"    • {key}: {value:.2f}")
                    else:
                        text.append(f"    • {key}: {value}")
            text.append("")
        
        # Recommendations
        if explanation['recommendations']:
            text.append("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(explanation['recommendations'], 1):
                text.append(f"  {i}. {rec}")
            text.append("")
        
        # Top factors
        if 'top_factors' in explanation:
            text.append("⭐ KEY FACTORS IN THIS MATCH:")
            for factor in explanation['top_factors']:
                text.append(f"  • {factor}")
        
        text.append("\n" + "=" * 70)
        
        return "\n".join(text)


class QualityAnalyzer:
    """Analyze and report on item-bank quality"""
    
    def __init__(self):
        self.quality_report = {}

    def analyze_item_quality(self,
                             students: pd.DataFrame,
                             jobs: pd.DataFrame,
                             predictions: np.ndarray,
                             probabilities: np.ndarray,
                             y_true: np.ndarray) -> Dict[str, Any]:
        """
        Analyze quality of items (students and jobs) based on model predictions

        Returns:
            Quality report with flags for weak items
        """

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_students': len(students),
            'total_jobs': len(jobs),
            'weak_students': [],
            'weak_jobs': [],
            'quality_metrics': {}
        }

        # ==========================================
        # FIX SHAPE MISMATCH
        # ==========================================

        n = min(
            len(students),
            len(probabilities),
            len(predictions),
            len(y_true)
        )

        students = students.iloc[:n].reset_index(drop=True)

        predictions = np.array(predictions[:n])
        probabilities = np.array(probabilities[:n])
        y_true = np.array(y_true[:n])

        # ==========================================
        # STUDENT QUALITY ANALYSIS
        # ==========================================

        student_match_scores = {}

        for idx, row in students.iterrows():

            sid = row['student_id']

            if idx < len(probabilities):
                student_match_scores[sid] = float(probabilities[idx])

        if len(student_match_scores) > 0:
            threshold = np.percentile(
                list(student_match_scores.values()),
                25
            )

            report['weak_students'] = [
                {
                    'student_id': sid,
                    'avg_match_score': round(score, 4)
                }
                for sid, score in student_match_scores.items()
                if score < threshold
            ]

        # ==========================================
        # JOB QUALITY ANALYSIS
        # ==========================================

        job_match_scores = {}

        if len(jobs) > 0:

            probs_per_job = max(
                1,
                len(probabilities) // len(jobs)
            )

            for idx, row in jobs.iterrows():

                jid = row['job_id']

                start = idx * probs_per_job
                end = min(
                    start + probs_per_job,
                    len(probabilities)
                )

                job_probs = probabilities[start:end]

                if len(job_probs) > 0:
                    job_match_scores[jid] = float(
                        np.mean(job_probs)
                    )

        if len(job_match_scores) > 0:
            threshold = np.percentile(
                list(job_match_scores.values()),
                25
            )

            report['weak_jobs'] = [
                {
                    'job_id': jid,
                    'avg_match_score': round(score, 4)
                }
                for jid, score in job_match_scores.items()
                if score < threshold
            ]

        # ==========================================
        # OVERALL QUALITY METRICS
        # ==========================================

        report['quality_metrics'] = {
            'avg_match_confidence': float(
                np.mean(probabilities)
            ),

            'std_match_confidence': float(
                np.std(probabilities)
            ),

            'median_match_confidence': float(
                np.median(probabilities)
            ),

            'prediction_distribution': {
                'positive_matches': int(
                    np.sum(predictions)
                ),

                'negative_matches': int(
                    len(predictions) - np.sum(predictions)
                )
            },

            'accuracy': float(
                np.mean(predictions == y_true)
            )
        }

        return report
    
    def flag_items_for_review(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate flags for admin review"""
        
        flags = []
        
        if len(quality_report['weak_students']) > 0:
            flags.append(f"⚠️  {len(quality_report['weak_students'])} students with low match rates - may need profile improvement")
        
        if len(quality_report['weak_jobs']) > 0:
            flags.append(f"⚠️  {len(quality_report['weak_jobs'])} jobs with low match rates - may need better description")
        
        avg_conf = quality_report['quality_metrics']['avg_match_confidence']
        if avg_conf < 0.5:
            flags.append(f"⚠️  Overall match confidence low ({avg_conf:.1%}) - may need data cleaning")
        
        return flags
