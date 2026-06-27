"""
Machine Learning Models for Recommendation System
Includes baseline, classical, and advanced models
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, 
                            roc_auc_score, confusion_matrix, 
                            classification_report, f1_score)
from config import MODELS_DIR

class BaselineModel:
    """Simple baseline model - rule-based matching"""
    
    def __init__(self):
        self.name = "Baseline (Rule-based)"
        
    def fit(self, X_train, y_train):
        """Baseline doesn't need training"""
        self.threshold = 0.60
        return self
    
    def predict(self, X):
        """Predict based on overall match score"""
        if 'overall_match_score' in X.columns:
            predictions = (X['overall_match_score'] >= self.threshold).astype(int).values
        else:
            # Fallback: use average of available scores
            score_cols = [col for col in X.columns if 'score' in col.lower()]
            predictions = (X[score_cols].mean(axis=1) >= self.threshold).astype(int).values
        return predictions
    
    def predict_proba(self, X):
        """Return probability estimates"""
        if 'overall_match_score' in X.columns:
            scores = X['overall_match_score'].values
        else:
            score_cols = [col for col in X.columns if 'score' in col.lower()]
            scores = X[score_cols].mean(axis=1).values
        
        scores = np.clip(scores, 0, 1)
        return np.column_stack([1 - scores, scores])


class LogisticRegressionModel:
    """Logistic Regression model for interpretability"""
    
    def __init__(self):
        self.name = "Logistic Regression"
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        """Train logistic regression model"""
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance from coefficients"""
        importance = np.abs(self.model.coef_[0])
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class RandomForestModel:
    """Random Forest model for better non-linear patterns"""
    
    def __init__(self):
        self.name = "Random Forest"
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        """Train random forest model"""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class GradientBoostingModel:
    """Gradient Boosting model for best performance"""
    
    def __init__(self):
        self.name = "Gradient Boosting"
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        """Train gradient boosting model"""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class NeuralNetworkModel:
    """Simple Neural Network using scikit-learn"""
    
    def __init__(self):
        self.name = "Neural Network"
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        """Train neural network"""
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class ModelEvaluator:
    """Evaluate and compare multiple models"""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
        }
        
        # False Positive Rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Additional metrics
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        
        return metrics
    
    @staticmethod
    def compare_models(models_dict, X_test, y_test):
        """Compare multiple models"""
        results = {}
        
        for model_name, model in models_dict.items():
            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
            results[model_name] = metrics
            
            print(f"\n{'='*50}")
            print(f"Model: {model_name}")
            print(f"{'='*50}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"FPR:       {metrics['fpr']:.4f}")
        
        return pd.DataFrame(results).T.sort_values('f1', ascending=False)


class ModelRegistry:
    """Save and load models"""
    
    @staticmethod
    def save_model(model, model_name):
        """Save model to disk"""
        path = MODELS_DIR / f"{model_name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved model: {model_name}")
        
    @staticmethod
    def load_model(model_name):
        """Load model from disk"""
        path = MODELS_DIR / f"{model_name}.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded model: {model_name}")
        return model


if __name__ == "__main__":
    from data_preparation import DataLoader
    
    # Load data
    students, jobs, matches = DataLoader.load_data()
    data = DataLoader.prepare_features(students, jobs, matches)
    X_train, X_val, X_test, y_train, y_val, y_test = DataLoader.get_train_test_split(data)
    
    # Select numeric features only
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_features]
    X_test = X_test[numeric_features]
    
    # Initialize models
    models = {
        'Baseline': BaselineModel(),
        'Logistic Regression': LogisticRegressionModel(),
        'Random Forest': RandomForestModel(),
        'Gradient Boosting': GradientBoostingModel(),
        'Neural Network': NeuralNetworkModel(),
    }
    
    # Train models
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        ModelRegistry.save_model(model, name.replace(' ', '_').lower())
    
    # Evaluate models
    print("\nEvaluating models...")
    comparison = ModelEvaluator.compare_models(models, X_test, y_test)
    print("\nModel Comparison:")
    print(comparison)
