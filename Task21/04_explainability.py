"""
PlaceMux Explainability Module
Provides interpretable predictions with SHAP values and explanations
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder
import shap

class ExplainabilityEngine:
    def __init__(self, model, feature_names, scaler=None, encoders=None):
        """
        Initialize explainability engine
        model: Trained ML model
        feature_names: List of feature names
        scaler: StandardScaler for scaled features
        encoders: Dict of LabelEncoders for categorical features
        """
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.encoders = encoders or {}
        self.explainer = None
        
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None
    
    def explain_prediction(self, X_input, prediction, prediction_prob):
        """
        Generate plain-English explanation for a prediction
        """
        explanation = {
            'prediction': int(prediction),
            'confidence': float(prediction_prob),
            'interpretation': self._interpret_prediction(prediction, prediction_prob),
            'top_reasons': self._get_top_reasons(X_input),
        }
        return explanation
    
    def _interpret_prediction(self, prediction, confidence):
        """Convert prediction to plain English"""
        if prediction == 1:
            if confidence > 0.9:
                return f"Excellent match (confidence: {confidence:.1%})"
            elif confidence > 0.75:
                return f"Good match (confidence: {confidence:.1%})"
            else:
                return f"Marginal match (confidence: {confidence:.1%})"
        else:
            if confidence < 0.1:
                return f"Not a match (confidence: {confidence:.1%})"
            elif confidence < 0.25:
                return f"Likely not a match (confidence: {confidence:.1%})"
            else:
                return f"Borderline (confidence: {confidence:.1%})"
    
    def _get_top_reasons(self, X_input):
        """Get top features contributing to prediction"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-3:][::-1]
            
            reasons = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature = self.feature_names[idx]
                    value = X_input[idx]
                    importance = importances[idx]
                    reasons.append({
                        'feature': feature,
                        'value': float(value) if isinstance(value, (int, float, np.number)) else str(value),
                        'importance': float(importance)
                    })
            return reasons
        return []
    
    def create_shap_explainer(self, X_train_sample):
        """Create SHAP explainer"""
        print("Creating SHAP explainer...")
        
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(self.model, X_train_sample)
        
        return self.explainer
    
    def get_shap_explanation(self, X_input):
        """Get SHAP explanation for single input"""
        if self.explainer is None:
            return None
        
        try:
            shap_values = self.explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            return {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values),
                'base_value': float(self.explainer.expected_value) if not isinstance(self.explainer.expected_value, list) else float(self.explainer.expected_value[1]),
            }
        except:
            return None
    
    def generate_explanation_report(self, student_data, job_data, prediction, confidence):
        """Generate comprehensive explanation report"""
        report = {
            'student_summary': {
                'years_exp': student_data.get('years_exp', 'N/A'),
                'gpa': student_data.get('gpa', 'N/A'),
                'num_projects': student_data.get('num_projects', 'N/A'),
                'internships': student_data.get('internships', 'N/A'),
            },
            'job_summary': {
                'required_exp_min': job_data.get('required_exp_min', 'N/A'),
                'required_exp_max': job_data.get('required_exp_max', 'N/A'),
                'salary_min': job_data.get('salary_min', 'N/A'),
                'salary_max': job_data.get('salary_max', 'N/A'),
            },
            'prediction': {
                'is_match': int(prediction),
                'confidence': float(confidence),
                'interpretation': self._interpret_prediction(prediction, confidence),
            },
            'explanation': {
                'summary': self._generate_summary(prediction, confidence, student_data, job_data),
                'key_factors': self._get_key_factors(student_data, job_data),
                'recommendations': self._get_recommendations(prediction, confidence),
            }
        }
        return report
    
    def _generate_summary(self, prediction, confidence, student_data, job_data):
        """Generate summary explanation"""
        if prediction == 1:
            return (f"This student is a potential match for this role. "
                   f"With {student_data.get('years_exp', 0)} years of experience and a "
                   f"{student_data.get('gpa', 0):.1f} GPA, they align well with the job requirements.")
        else:
            return (f"This student doesn't meet the requirements for this role. "
                   f"There may be gaps in experience or skill alignment that would need to be addressed.")
    
    def _get_key_factors(self, student_data, job_data):
        """Extract key matching factors"""
        factors = []
        
        # Experience check
        student_exp = student_data.get('years_exp', 0)
        req_exp = job_data.get('required_exp_min', 0)
        if student_exp >= req_exp:
            factors.append(f"✓ Experience sufficient ({student_exp} years >= {req_exp} required)")
        else:
            factors.append(f"⚠ Experience gap ({student_exp} years < {req_exp} required)")
        
        # GPA check
        student_gpa = student_data.get('gpa', 0)
        req_gpa = job_data.get('required_gpa_min', 0)
        if student_gpa >= req_gpa:
            factors.append(f"✓ Academic credentials adequate (GPA {student_gpa:.1f})")
        else:
            factors.append(f"⚠ Academic credentials below requirement (GPA {student_gpa:.1f})")
        
        # Projects
        projects = student_data.get('num_projects', 0)
        if projects > 0:
            factors.append(f"✓ Practical experience ({projects} projects completed)")
        else:
            factors.append(f"⚠ Limited project experience")
        
        # Internships
        internships = student_data.get('internships', 0)
        if internships > 0:
            factors.append(f"✓ Professional experience ({internships} internships)")
        else:
            factors.append(f"⚠ No formal internship experience")
        
        return factors
    
    def _get_recommendations(self, prediction, confidence):
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction == 1:
            if confidence > 0.9:
                recommendations.append("Ready for immediate interview")
                recommendations.append("Strong technical alignment")
            elif confidence > 0.75:
                recommendations.append("Recommended for interview after skills validation")
                recommendations.append("Verify specific technical skills")
            else:
                recommendations.append("Consider for interview but assess fit carefully")
                recommendations.append("Conduct technical assessment")
        else:
            recommendations.append("Suggest skill development path")
            recommendations.append("Recommend applying for entry-level positions")
            recommendations.append("Focus on building project portfolio")
        
        return recommendations


def create_demo_explanation():
    """Create demo explanation for a single student-job pair"""
    print("\n" + "=" * 60)
    print("SAMPLE EXPLAINABILITY OUTPUT")
    print("=" * 60 + "\n")
    
    sample_explanation = {
        'student_summary': {
            'years_exp': 2,
            'gpa': 8.5,
            'num_projects': 5,
            'internships': 2,
        },
        'job_summary': {
            'required_exp_min': 1,
            'required_exp_max': 5,
            'salary_min': 500000,
            'salary_max': 1000000,
        },
        'prediction': {
            'is_match': 1,
            'confidence': 0.87,
            'interpretation': 'Good match (confidence: 87%)',
        },
        'explanation': {
            'summary': 'This student is a potential match for this role. With 2 years of experience and a 8.5 GPA, they align well with the job requirements.',
            'key_factors': [
                '✓ Experience sufficient (2 years >= 1 required)',
                '✓ Academic credentials adequate (GPA 8.5)',
                '✓ Practical experience (5 projects completed)',
                '✓ Professional experience (2 internships)',
            ],
            'recommendations': [
                'Ready for immediate interview',
                'Strong technical alignment',
            ]
        }
    }
    
    print(json.dumps(sample_explanation, indent=2))
    return sample_explanation


if __name__ == "__main__":
    create_demo_explanation()
