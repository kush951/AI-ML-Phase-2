"""
PlaceMux ML Model Training & Comparison
Trains and compares multiple models for student-job matching prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
import json
from datetime import datetime
import shap

class ModelTrainer:
    def __init__(self, features_df):
        self.features_df = features_df
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare features for modeling"""
        print("Preparing data...")
        
        # Separate features and target
        y = self.features_df['is_match']
        X = self.features_df.drop(['is_match', 'student_id', 'job_id', 'total_score'], axis=1)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'region', 'background', 'company_size', 'urgency_level']
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers['default'] = scaler
        
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Class distribution - Match: {(self.y_train == 1).sum() / len(self.y_train) * 100:.1f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n🔄 Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        metrics = self._evaluate_model(y_pred, y_pred_proba)
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = metrics
        
        return metrics
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("🔄 Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=42, class_weight='balanced', n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._evaluate_model(y_pred, y_pred_proba)
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        return metrics
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("🔄 Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.1,
            random_state=42, scale_pos_weight=1, n_jobs=-1
        )
        model.fit(self.X_train, self.y_train, verbose=0)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._evaluate_model(y_pred, y_pred_proba)
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        return metrics
    
    def train_svm(self):
        """Train SVM model"""
        print("🔄 Training SVM...")
        model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        metrics = self._evaluate_model(y_pred, y_pred_proba)
        self.models['svm'] = model
        self.results['svm'] = metrics
        
        return metrics
    
    def _evaluate_model(self, y_pred, y_pred_proba):
        """Evaluate model performance"""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'fpr': float(fpr),  # False Positive Rate
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        return metrics
    
    def train_all(self):
        """Train all models"""
        self.prepare_data()
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        
        return self.results
    
    def get_best_model(self):
        """Get best performing model"""
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        return best_model[0], best_model[1]
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        if model_name not in ['random_forest', 'xgboost']:
            return None
        
        model = self.models[model_name]
        importances = model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_shap_explanations(self, model_name, n_samples=100):
        """Get SHAP explanations for model predictions"""
        print(f"\nGenerating SHAP explanations for {model_name}...")
        model = self.models[model_name]
        
        # Sample data for SHAP
        X_sample = self.X_test.sample(n=min(n_samples, len(self.X_test)), random_state=42)
        
        if model_name == 'random_forest':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class
        elif model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For linear models
            explainer = shap.LinearExplainer(model, self.X_train_scaled)
            shap_values = explainer.shap_values(self.X_test_scaled)
        
        return shap_values, explainer, X_sample
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            with open(f'{output_dir}/{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers and encoders
        with open(f'{output_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(f'{output_dir}/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save feature names
        with open(f'{output_dir}/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"\n✓ Models saved to {output_dir}/")
    
    def generate_report(self, output_file='model_comparison.json'):
        """Generate model comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(self.features_df),
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'num_features': len(self.feature_names),
                'positive_rate': float((self.features_df['is_match'] == 1).sum() / len(self.features_df)),
            },
            'model_results': self.results,
            'best_model': self.get_best_model()[0],
            'best_metrics': self.get_best_model()[1],
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {output_file}")
        return report


if __name__ == "__main__":
    # Load features
    features = pd.read_csv('data/features.csv')
    
    # Train models
    trainer = ModelTrainer(features)
    results = trainer.train_all()
    
    # Get best model
    best_model, best_metrics = trainer.get_best_model()
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1 Score: {best_metrics['f1']:.4f}")
    
    # Save
    trainer.save_models()
    trainer.generate_report()
    
    # Print feature importance for best model
    if best_model in ['random_forest', 'xgboost']:
        print(f"\n📊 Top Features ({best_model}):")
        importance = trainer.get_feature_importance(best_model)
        print(importance.head(10))
