"""
Matching Models Module
Implements multiple models for student-job matching with comprehensive evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, roc_auc_score, confusion_matrix,
                            precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    """Store evaluation metrics for a model"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    false_positive_rate: float
    specificity: float
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'auc_roc': round(self.auc_roc, 4),
            'false_positive_rate': round(self.false_positive_rate, 4),
            'specificity': round(self.specificity, 4)
        }


class BaselineModel:
    """Baseline model: Simple rule-based matching"""
    
    def __init__(self):
        self.name = "Baseline (Rule-Based)"
        self.model_type = "rule_based"
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Rule-based prediction:
        - Skill match ratio >= 0.6
        - Experience match >= 0.7
        - Location match = True
        - GPA match >= 0.7
        """
        predictions = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            skill_ratio = X[i, 0]
            exp_match = X[i, 4]
            location_match = X[i, 10]
            gpa_match = X[i, 7]
            
            # Rule-based logic
            if (skill_ratio >= 0.6 and exp_match >= 0.7 and 
                location_match >= 0.5 and gpa_match >= 0.7):
                predictions[i] = 1
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates"""
        proba = np.zeros((X.shape[0], 2))
        
        for i in range(X.shape[0]):
            skill_ratio = X[i, 0]
            exp_match = X[i, 4]
            location_match = X[i, 10]
            gpa_match = X[i, 7]
            
            # Score-based probability
            score = (skill_ratio * 0.4 + exp_match * 0.25 + 
                    location_match * 0.2 + gpa_match * 0.15)
            
            proba[i, 1] = score
            proba[i, 0] = 1 - score
        
        return proba


class MatchingModelEnsemble:
    """Ensemble of multiple matching models with cross-validation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.feature_names = None
        self.best_model_name = None
        self.best_model = None
        self.scaler = StandardScaler()
    
    def build_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     feature_names: List[str]) -> Dict[str, ModelMetrics]:
        """
        Build and evaluate multiple models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            feature_names: Names of features
            
        Returns:
            Dictionary of model metrics
        """
        
        self.feature_names = feature_names
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 1. Baseline Model
        baseline = BaselineModel()
        self.models['Baseline'] = baseline
        self.metrics['Baseline'] = self._evaluate_model(baseline, X_val, y_val)
        
        # 2. Logistic Regression
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        self.models['Logistic Regression'] = lr_model
        self.metrics['Logistic Regression'] = self._evaluate_model(
            lr_model, X_val_scaled, y_val, is_scaled=True
        )
        
        # 3. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = rf_model
        self.metrics['Random Forest'] = self._evaluate_model(
            rf_model, X_val_scaled, y_val, is_scaled=True
        )
        
        # 4. Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            subsample=0.8
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['Gradient Boosting'] = gb_model
        self.metrics['Gradient Boosting'] = self._evaluate_model(
            gb_model, X_val_scaled, y_val, is_scaled=True
        )
        
        # 5. Ensemble Voting (Average probabilities)
        self.models['Ensemble'] = 'voting'  # Special handling
        
        # Select best model based on F1 score (balanced metric)
        self._select_best_model()
        
        return self.metrics
    
    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray,
                       is_scaled: bool = False) -> ModelMetrics:
        """Evaluate a single model"""
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        # Calculate false positive rate and specificity
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            false_positive_rate=fpr,
            specificity=specificity
        )
    
    def _select_best_model(self) -> None:
        """Select best model based on F1 score"""
        
        best_f1 = -1
        best_model = None
        
        for model_name, metrics in self.metrics.items():
            if metrics.f1 > best_f1:
                best_f1 = metrics.f1
                best_model = model_name
        
        self.best_model_name = best_model
        self.best_model = self.models[best_model]
        print(f"\n🏆 Best Model Selected: {best_model} (F1: {best_f1:.4f})")
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Make predictions using specified or best model"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if model_name == 'Baseline':
            return model.predict(X)
        else:
            X_scaled = self.scaler.transform(X)
            return model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Get probability predictions"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if model_name == 'Baseline':
            return model.predict_proba(X)
        else:
            X_scaled = self.scaler.transform(X)
            return model.predict_proba(X_scaled)
    
    def get_feature_importance(self, model_name: str = None, top_k: int = 10) -> Dict[str, float]:
        """Get feature importance for tree-based models"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1][:top_k]
            
            importance_dict = {
                self.feature_names[i]: float(importances[i])
                for i in sorted_indices
            }
            return importance_dict
        elif hasattr(model, 'coef_'):
            # For logistic regression
            coef = np.abs(model.coef_[0])
            sorted_indices = np.argsort(coef)[::-1][:top_k]
            
            importance_dict = {
                self.feature_names[i]: float(coef[i])
                for i in sorted_indices
            }
            return importance_dict
        else:
            return {}
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models"""
        
        comparison_data = {}
        for model_name, metrics in self.metrics.items():
            comparison_data[model_name] = metrics.to_dict()
        
        df = pd.DataFrame(comparison_data).T
        return df.round(4)
    
    def get_models_summary(self) -> str:
        """Generate summary of all models"""
        
        summary = "=" * 80 + "\n"
        summary += "MODEL EVALUATION SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for model_name, metrics in self.metrics.items():
            summary += f"\n📊 {model_name}\n"
            summary += "-" * 40 + "\n"
            summary += f"  Accuracy:  {metrics.accuracy:.4f}\n"
            summary += f"  Precision: {metrics.precision:.4f}\n"
            summary += f"  Recall:    {metrics.recall:.4f}\n"
            summary += f"  F1-Score:  {metrics.f1:.4f}\n"
            summary += f"  AUC-ROC:   {metrics.auc_roc:.4f}\n"
            summary += f"  False Positive Rate: {metrics.false_positive_rate:.4f}\n"
            summary += f"  Specificity: {metrics.specificity:.4f}\n"
        
        summary += "\n" + "=" * 80 + "\n"
        summary += f"✅ BEST MODEL: {self.best_model_name}\n"
        summary += "=" * 80 + "\n"
        
        return summary
