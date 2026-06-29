"""
PlaceMux ML Models
Multiple models with explainability for job recommendation ranking
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import json
import pickle
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRecommendationModel:
    """Base class for recommendation models"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        self.feature_importance = None

    def prepare_data(self, df):
        """Prepare feature matrix and target"""
        feature_cols = [
            'tech_overlap', 'soft_overlap', 'student_avg_tech_score',
            'student_avg_soft_score', 'gpa_fit', 'exp_fit', 'student_gpa',
            'job_min_gpa', 'student_years_exp', 'job_exp_required'
        ]

        X = df[feature_cols].fillna(0)
        y = df['is_good_match']
        self.feature_names = feature_cols

        return X, y

    def fit(self, X_train, y_train):
        """Fit the model"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        logger.info(f"{self.model_name} model fitted successfully")

    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For SVM, use decision function
            scores = self.model.decision_function(X_scaled)
            return np.column_stack([1 - scores, scores])

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        self.metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': np.mean(y_pred == y_test),
            'false_positive_rate': np.mean(y_pred[(y_test == 0)] == 1) if sum(y_test == 0) > 0 else 0,
            'false_negative_rate': np.mean(y_pred[(y_test == 1)] == 0) if sum(y_test == 1) > 0 else 0,
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        self.metrics['confusion_matrix'] = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}

        return self.metrics

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(
            self.model, self.scaler.transform(X), y,
            cv=StratifiedKFold(n_splits=cv),
            scoring='roc_auc'
        )
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        return cv_scores

    def save(self, path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'model_name': self.model_name
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.metrics = data['metrics']
        logger.info(f"Model loaded from {path}")


class LogisticRegressionModel(BaseRecommendationModel):
    """Logistic Regression - Baseline model (most explainable)"""

    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def get_feature_importance(self):
        """Get feature coefficients for explainability"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        return importance


class RandomForestModel(BaseRecommendationModel):
    """Random Forest - Good performance with feature importance"""

    def __init__(self, n_estimators=100):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    def get_feature_importance(self):
        """Get feature importance scores"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance


class GradientBoostingModel(BaseRecommendationModel):
    """Gradient Boosting - Best performance"""

    def __init__(self, n_estimators=100):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

    def get_feature_importance(self):
        """Get feature importance scores"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance


class SVMModel(BaseRecommendationModel):
    """Support Vector Machine - Good generalization"""

    def __init__(self):
        super().__init__("SVM (RBF)")
        self.model = SVC(kernel='rbf', probability=True, random_state=42)


class ModelEnsemble:
    """Ensemble of multiple models with voting"""

    def __init__(self, models):
        self.models = {m.model_name: m for m in models}
        self.ensemble_weights = None

    def fit_weights(self, X_val, y_val):
        """Learn optimal weights for ensemble based on validation performance"""
        scores = {}
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores[name] = score

        # Normalize scores to weights
        total = sum(scores.values())
        self.ensemble_weights = {k: v / total for k, v in scores.items()}
        logger.info(f"Ensemble weights: {self.ensemble_weights}")

    def predict_proba(self, X):
        """Predict with weighted ensemble"""
        if not self.ensemble_weights:
            raise ValueError("Call fit_weights first")

        ensemble_proba = np.zeros(len(X))
        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            ensemble_proba += self.ensemble_weights[name] * proba

        return ensemble_proba

    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """Evaluate ensemble"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': np.mean(y_pred == y_test),
        }

        return metrics


class ModelSelector:
    """Select best model based on metrics"""

    @staticmethod
    def compare_models(models, X_test, y_test):
        """Compare all models and return rankings"""
        results = []

        for model in models:
            metrics = model.evaluate(X_test, y_test)
            results.append({
                'model': model.model_name,
                'metrics': metrics
            })

        # Sort by F1 score (balance between precision and recall)
        results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)

        logger.info("\n=== MODEL COMPARISON ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['model']}")
            print(f"   F1 Score: {result['metrics']['f1']:.4f}")
            print(f"   Precision: {result['metrics']['precision']:.4f}")
            print(f"   Recall: {result['metrics']['recall']:.4f}")
            print(f"   ROC-AUC: {result['metrics']['roc_auc']:.4f}")
            print(f"   Accuracy: {result['metrics']['accuracy']:.4f}")

        return results

    @staticmethod
    def get_best_model(results):
        """Get best model from comparison"""
        return results[0]['model']


if __name__ == '__main__':
    print("PlaceMux ML Models - Ready for training")