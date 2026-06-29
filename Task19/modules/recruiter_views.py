"""
Recruiter Views & Analytics
Provides dashboards and analytics for recruiters and admins
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

class RecruiterDashboard:
    """Generate recruiter analytics and views"""
    
    def __init__(self):
        self.generated_at = datetime.now()

    def job_analytics(self,
                      jobs: pd.DataFrame,
                      predictions: np.ndarray,
                      probabilities: np.ndarray) -> Dict[str, Any]:

        """
        Generate analytics for recruiter dashboard
        """

        analytics = {
            'total_jobs': len(jobs),
            'job_performance': [],
            'top_jobs': [],
            'low_performing_jobs': [],
            'overall_metrics': {}
        }

        # ==========================================
        # FIX SHAPE MISMATCH
        # ==========================================

        probabilities = np.array(probabilities)
        predictions = np.array(predictions)

        n_probs = len(probabilities)

        if n_probs == 0:
            return analytics

        # ==========================================
        # DISTRIBUTE PROBABILITIES ACROSS JOBS
        # ==========================================

        probs_per_job = max(
            1,
            n_probs // len(jobs)
        )

        job_scores = []

        for idx, row in jobs.iterrows():

            jid = row['job_id']

            start = idx * probs_per_job
            end = min(
                start + probs_per_job,
                n_probs
            )

            # Skip if out of bounds
            if start >= n_probs:
                break

            job_probs = probabilities[start:end]

            if len(job_probs) == 0:
                continue

            avg_score = float(np.mean(job_probs))

            positive_matches = int(
                np.sum(job_probs >= 0.5)
            )

            total_matches = len(job_probs)

            performance = {
                'job_id': jid,
                'job_title': row.get(
                    'job_title',
                    f'Job-{jid}'
                ),
                'avg_match_score': round(avg_score, 4),
                'positive_matches': positive_matches,
                'total_candidates': total_matches,
                'success_rate': round(
                    positive_matches / total_matches,
                    4
                )
            }

            analytics['job_performance'].append(
                performance
            )

            job_scores.append(
                (jid, avg_score)
            )

        # ==========================================
        # TOP & LOW PERFORMING JOBS
        # ==========================================

        sorted_jobs = sorted(
            analytics['job_performance'],
            key=lambda x: x['avg_match_score'],
            reverse=True
        )

        analytics['top_jobs'] = sorted_jobs[:5]
        analytics['low_performing_jobs'] = sorted_jobs[-5:]

        # ==========================================
        # OVERALL METRICS
        # ==========================================

        analytics['overall_metrics'] = {
            'average_match_score': float(
                np.mean(probabilities)
            ),

            'highest_match_score': float(
                np.max(probabilities)
            ),

            'lowest_match_score': float(
                np.min(probabilities)
            ),

            'positive_match_rate': float(
                np.mean(predictions)
            )
        }

        return analytics
    
    def student_analytics(self, students: pd.DataFrame, predictions: np.ndarray,
                         probabilities: np.ndarray, matches: pd.DataFrame) -> Dict[str, Any]:
        """Generate analytics for students"""
        
        analytics = {}
        
        for idx, (_, student) in enumerate(students.iterrows()):
            student_id = student['student_id']
            
            # Get predictions for this student
            student_mask = matches['student_id'] == student_id
            student_matches = matches[student_mask]
            
            if len(student_matches) == 0:
                continue
            
            # Get indices for this student
            student_indices = student_matches.index.tolist()
            student_probs = probabilities[student_indices]
            student_preds = predictions[student_indices]
            
            analytics[student_id] = {
                'name': student['name'],
                'college': student['college'],
                'gpa': student['gpa'],
                'location': student['location'],
                'years_experience': student['years_of_experience'],
                'total_matches': len(student_indices),
                'suitable_jobs': int(np.sum(student_preds)),
                'match_rate': float(np.sum(student_preds) / len(student_indices)) if len(student_indices) > 0 else 0,
                'avg_job_quality': float(np.mean(student_probs)),
                'skills_count': len(student['skills']),
                'avg_skill_confidence': float(np.mean(list(student['skills'].values()))) if student['skills'] else 0,
                'projects_completed': student['projects_count'],
                'verified': student['verified'],
                'onboarding_date': str(student['onboarding_date']),
                'days_since_onboard': (datetime.now().date() - student['onboarding_date']).days
            }
        
        return analytics
    
    def _get_top_candidates(self, indices: List[int], probabilities: np.ndarray, 
                           top_k: int = 5) -> List[Dict[str, float]]:
        """Get top candidates for a job"""
        
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_candidates = [
            {
                'index': int(indices[i]),
                'quality_score': float(probabilities[i])
            }
            for i in top_indices
        ]
        return top_candidates
    
    def bulk_onboarding_summary(self, students: pd.DataFrame) -> Dict[str, Any]:
        """Summary of bulk student onboarding"""
        
        # Group by onboarding date
        students['onboarding_date'] = pd.to_datetime(students['onboarding_date'])
        onboarding_by_date = students.groupby('onboarding_date').size()
        
        # Statistics
        summary = {
            'total_students': len(students),
            'verified_students': int(students['verified'].sum()),
            'unverified_students': int((~students['verified']).sum()),
            'verification_rate': float(students['verified'].mean()),
            'avg_gpa': float(students['gpa'].mean()),
            'avg_experience': float(students['years_of_experience'].mean()),
            'colleges_represented': int(students['college'].nunique()),
            'unique_skills': len(set().union(*students['skills'])),
            'onboarding_trend': {
                str(date): int(count)
                for date, count in onboarding_by_date.items()
            },
            'geographic_distribution': students['location'].value_counts().to_dict(),
            'skill_distribution': self._get_skill_distribution(students),
            'recent_onboardings': self._get_recent_onboardings(students, 5)
        }
        
        return summary
    
    def _get_skill_distribution(self, students: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of skills across students"""
        
        skill_counts = {}
        for _, student in students.iterrows():
            for skill in student['skills'].keys():
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Return top 15 skills
        return dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    def _get_recent_onboardings(self, students: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        """Get most recent student onboardings"""
        
        students['onboarding_date'] = pd.to_datetime(students['onboarding_date'])
        recent = students.nlargest(n, 'onboarding_date')[
            ['student_id', 'name', 'college', 'gpa', 'onboarding_date', 'verified']
        ]
        
        return [
            {
                'student_id': row['student_id'],
                'name': row['name'],
                'college': row['college'],
                'gpa': float(row['gpa']),
                'onboarding_date': str(row['onboarding_date'].date()),
                'verified': bool(row['verified'])
            }
            for _, row in recent.iterrows()
        ]
    
    def matching_quality_report(self, predictions: np.ndarray, 
                               probabilities: np.ndarray,
                               y_true: np.ndarray) -> Dict[str, Any]:
        """Generate quality report for matches"""
        
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        report = {
            'total_matches_analyzed': len(predictions),
            'positive_matches': int(np.sum(predictions)),
            'negative_matches': int(len(predictions) - np.sum(predictions)),
            'metrics': {
                'precision': float(precision_score(y_true, predictions, zero_division=0)),
                'recall': float(recall_score(y_true, predictions, zero_division=0)),
                'f1_score': float(f1_score(y_true, predictions, zero_division=0))
            },
            'confidence_distribution': {
                'high_confidence_0.8_1.0': int(np.sum((probabilities >= 0.8) & (predictions == 1))),
                'medium_confidence_0.6_0.8': int(np.sum((probabilities >= 0.6) & (probabilities < 0.8) & (predictions == 1))),
                'low_confidence_0.4_0.6': int(np.sum((probabilities >= 0.4) & (probabilities < 0.6) & (predictions == 1))),
            },
            'avg_match_confidence': float(np.mean(probabilities[predictions == 1])) if np.sum(predictions) > 0 else 0,
            'std_match_confidence': float(np.std(probabilities[predictions == 1])) if np.sum(predictions) > 0 else 0
        }
        
        return report


class AdminPanel:
    """Admin panel for system monitoring and item-bank quality"""
    
    def __init__(self):
        self.alerts = []
        self.recommendations = []
    
    def generate_admin_report(self, 
                             students: pd.DataFrame,
                             jobs: pd.DataFrame,
                             predictions: np.ndarray,
                             probabilities: np.ndarray,
                             weak_students: List[Dict],
                             weak_jobs: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive admin report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_health': self._check_system_health(students, jobs),
            'data_quality': self._assess_data_quality(students, jobs),
            'item_bank_quality': {
                'weak_students_count': len(weak_students),
                'weak_jobs_count': len(weak_jobs),
                'weak_students': weak_students[:10],  # Show top 10
                'weak_jobs': weak_jobs[:10]  # Show top 10
            },
            'recommendations': self._generate_recommendations(weak_students, weak_jobs)
        }
        
        return report
    
    def _check_system_health(self, students: pd.DataFrame, jobs: pd.DataFrame) -> Dict[str, Any]:
        """Check overall system health"""
        
        return {
            'students_registered': len(students),
            'jobs_posted': len(jobs),
            'active_jobs': int(jobs['is_active'].sum()),
            'verified_students': int(students['verified'].sum()),
            'avg_student_quality': float(
                students['gpa'].mean() * 0.5 + 
                students['years_of_experience'].mean() / 5 * 0.5
            ),
            'data_freshness': {
                'oldest_job': (datetime.now().date() - jobs['posted_date'].min()).days,
                'newest_job': (datetime.now().date() - jobs['posted_date'].max()).days,
            }
        }
    
    def _assess_data_quality(self, students: pd.DataFrame, jobs: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality"""
        
        return {
            'student_verification_rate': float(students['verified'].mean()),
            'students_with_skills': int((students['skills'].str.len() > 0).sum()),
            'students_with_experience': int((students['years_of_experience'] > 0).sum()),
            'jobs_with_description': len(jobs),  # Assuming all have description
            'avg_required_skills_per_job': float(jobs['required_skills'].str.len().mean()),
            'data_completeness_score': 0.95  # Example score
        }
    
    def _generate_recommendations(self, weak_students: List[Dict], 
                                 weak_jobs: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        
        recs = []
        
        if len(weak_students) > 5:
            recs.append(
                f"🎯 Contact {len(weak_students)} students with low match rates to help improve their profiles"
            )
        
        if len(weak_jobs) > 5:
            recs.append(
                f"📝 Review {len(weak_jobs)} jobs with low match rates - consider clarifying requirements"
            )
        
        recs.append("📊 Monitor model performance weekly and retrain if drift is detected")
        recs.append("✅ Schedule verification for remaining unverified student profiles")
        
        return recs
