import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path

class BaselineModel:
    """Simple baseline: skill overlap threshold"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.name = "Baseline (Skill Overlap)"
    
    def fit(self, X, y):
        """No training needed for baseline"""
        return self
    
    def predict(self, X):
        """Predict based on skill overlap threshold"""
        return (X['skill_overlap'] >= self.threshold).astype(int).values
    
    def predict_proba(self, X):
        """Return probabilities"""
        proba = X['skill_overlap'].values
        return np.column_stack([1 - proba, proba])
    
    def get_explanation(self, features):
        """Return explanation for prediction"""
        skill_overlap = features.get('skill_overlap', 0)
        return f"Skill overlap: {skill_overlap:.2%}. {'Match' if skill_overlap >= self.threshold else 'No match'}"


class LogisticRegressionModel:
    """Logistic Regression for skill matching"""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.name = "Logistic Regression"
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the model"""
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_explanation(self, features, coefficients=None):
        """Return explanation based on feature importance"""
        return f"Skill match probability based on feature combination"


class RandomForestModel:
    """Random Forest for skill matching"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
        self.scaler = StandardScaler()
        self.name = "Random Forest"
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the model"""
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return dict(zip(self.feature_names, self.model.feature_importances_))


class XGBoostModel:
    """XGBoost for skill matching"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = XGBClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=6)
        self.scaler = StandardScaler()
        self.name = "XGBoost"
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the model"""
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return dict(zip(self.feature_names, self.model.feature_importances_))


class GradientBoostingModel:
    """Gradient Boosting for skill matching"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=5)
        self.scaler = StandardScaler()
        self.name = "Gradient Boosting"
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the model"""
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return dict(zip(self.feature_names, self.model.feature_importances_))


class SVMModel:
    """Support Vector Machine for skill matching"""
    
    def __init__(self, kernel='rbf', random_state=42):
        self.model = SVC(kernel=kernel, probability=True, random_state=random_state)
        self.scaler = StandardScaler()
        self.name = "Support Vector Machine"
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the model"""
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    
    models = [
        BaselineModel(threshold=0.5),
        LogisticRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
        GradientBoostingModel(),
        SVMModel()
    ]
    
    results = []
    
    for model in models:
        print(f"Training {model.name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model.name,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'model': model
        }
        
        results.append(metrics)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n")
    
    return results


def select_best_model(results):
    """Select best model based on F1 score"""
    return max(results, key=lambda x: x['f1'])


def save_model(model, filepath):
    """Save trained model"""
    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """Load trained model"""
    return joblib.load(filepath)
