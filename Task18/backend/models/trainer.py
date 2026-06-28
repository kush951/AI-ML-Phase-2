"""
Model Trainer - Multi-model approach with best model selection
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train and evaluate multiple ML models for job matching
    
    Models included:
    1. Logistic Regression (Baseline) - fast, interpretable
    2. SVM (Support Vector Machine) - good generalization
    3. Random Forest - feature importance insights
    4. Gradient Boosting - best performance potential
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Initialize and train all models
        
        Args:
            X_train: Training features
            y_train: Training labels (0: no match, 1: match)
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features for interpretability
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate baseline (simple skill overlap)
        self.baseline_f1 = self._calculate_baseline()
        logger.info(f"📊 Baseline F1 Score: {self.baseline_f1:.4f}")
        
        # Train all models
        self.trained_models = {}
        self.model_metrics = {}
        self._train_all_models()
        
        # Select best model
        self.best_model_name = self._select_best_model()
        logger.info(f"🏆 Best Model Selected: {self.best_model_name}")
    
    def _calculate_baseline(self):
        """
        Calculate baseline: simple majority class prediction
        (All jobs are good matches)
        """
        baseline_pred = np.ones_like(self.y_test)
        baseline_f1 = f1_score(self.y_test, baseline_pred)
        return baseline_f1
    
    def _train_all_models(self):
        """Train all models and evaluate"""
        
        # 1. Logistic Regression (Baseline Model - Fast & Interpretable)
        logger.info("📚 Training Logistic Regression (Baseline)...")
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=0.1,  # Regularization strength
            solver='lbfgs'
        )
        lr_model.fit(self.X_train_scaled, self.y_train)
        self._evaluate_model(lr_model, "Logistic Regression")
        self.trained_models["Logistic Regression"] = lr_model
        
        # 2. Support Vector Machine
        logger.info("📚 Training SVM...")
        svm_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0,
            gamma='scale'
        )
        svm_model.fit(self.X_train_scaled, self.y_train)
        self._evaluate_model(svm_model, "SVM")
        self.trained_models["SVM"] = svm_model
        
        # 3. Random Forest
        logger.info("📚 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)  # No scaling needed
        self._evaluate_model(rf_model, "Random Forest")
        self.trained_models["Random Forest"] = rf_model
        
        # 4. Gradient Boosting
        logger.info("📚 Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        gb_model.fit(self.X_train, self.y_train)  # No scaling needed
        self._evaluate_model(gb_model, "Gradient Boosting")
        self.trained_models["Gradient Boosting"] = gb_model
    
    def _evaluate_model(self, model, model_name):
        """Evaluate model and store metrics"""
        
        # Use appropriate scaled data based on model type
        if isinstance(model, (LogisticRegression, SVC)):
            X_test_use = self.X_test_scaled
        else:
            X_test_use = self.X_test
        
        # Predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            auc_roc = 0.0
        
        # False positive rate
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "fpr": fpr,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        
        self.model_metrics[model_name] = metrics
        
        logger.info(f"  ✓ {model_name:25s} | Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
    
    def _select_best_model(self):
        """
        Select best model based on F1 score
        (Balanced metric for classification)
        """
        best_model = max(
            self.model_metrics.items(),
            key=lambda x: x[1]["f1_score"]
        )
        return best_model[0]
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance from tree-based models
        """
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_dict = {
                name: float(importance)
                for name, importance in zip(self.feature_names, importances)
            }
            # Sort by importance
            return dict(sorted(
                feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            coefs = np.abs(model.coef_[0])
            feature_coef_dict = {
                name: float(coef)
                for name, coef in zip(self.feature_names, coefs)
            }
            return dict(sorted(
                feature_coef_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        else:
            return {}
    
    def get_model_summary(self):
        """Get summary of all models"""
        summary = {}
        for model_name, metrics in self.model_metrics.items():
            summary[model_name] = {
                "accuracy": f"{metrics['accuracy']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "f1_score": f"{metrics['f1_score']:.4f}",
                "auc_roc": f"{metrics['auc_roc']:.4f}",
                "false_positive_rate": f"{metrics['fpr']:.4f}",
            }
        return summary
