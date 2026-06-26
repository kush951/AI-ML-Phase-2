"""
PlaceMux Proctoring False-Positive Reduction Pipeline
ML Pipeline with multiple models for FP reduction in verification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

class FPReductionPipeline:
    """
    Pipeline for reducing false positives in proctoring verification
    
    Features:
    - Skill match score (0-1)
    - Time spent in session (minutes)
    - Camera availability (0/1)
    - Environment quality score (0-1)
    - Verification confidence (0-1)
    - Session completion percentage (0-1)
    - Answer consistency (0-1)
    - Device fingerprint stability (0-1)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'skill_match', 'session_duration', 'camera_available',
            'env_quality', 'verification_confidence', 'completion_pct',
            'answer_consistency', 'device_stability'
        ]
        self.baseline_threshold = 0.5
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate realistic synthetic proctoring data"""
        np.random.seed(self.random_state)
        
        data = {
            'skill_match': np.random.uniform(0, 1, n_samples),
            'session_duration': np.random.normal(45, 15, n_samples),  # minutes
            'camera_available': np.random.randint(0, 2, n_samples),
            'env_quality': np.random.uniform(0, 1, n_samples),
            'verification_confidence': np.random.uniform(0, 1, n_samples),
            'completion_pct': np.random.uniform(0, 1, n_samples),
            'answer_consistency': np.random.uniform(0, 1, n_samples),
            'device_stability': np.random.uniform(0, 1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create labels based on realistic business logic
        # High FP rate: incorrectly marked as verified
        fraud_indicators = (
            (df['skill_match'] < 0.4) & (df['verification_confidence'] > 0.7) |  # Suspicious mismatch
            (df['session_duration'] < 10) & (df['completion_pct'] > 0.8) |  # Too fast
            (df['camera_available'] == 0) & (df['env_quality'] > 0.6) |  # Missing camera
            (df['answer_consistency'] < 0.3) & (df['verification_confidence'] > 0.8)  # Inconsistent answers
        )
        
        df['is_fraud'] = fraud_indicators.astype(int)
        
        # Imbalanced data (more legitimate than fraud) - realistic scenario
        true_fraud_pct = 0.15
        fraud_idx = df[df['is_fraud'] == 1].index.tolist()
        legit_idx = df[df['is_fraud'] == 0].index.tolist()
        
        # Keep only a portion of fraud cases
        keep_fraud = int(len(legit_idx) * true_fraud_pct / (1 - true_fraud_pct))
        fraud_idx = np.random.choice(fraud_idx, min(keep_fraud, len(fraud_idx)), replace=False)
        
        df = df.loc[list(fraud_idx) + legit_idx].reset_index(drop=True)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for modeling"""
        X = df[self.feature_names].copy()
        y = df['is_fraud'].copy()
        
        # Handle any NaN values
        X = X.fillna(X.mean())
        
        # Normalize session_duration to 0-1 range for consistency
        X['session_duration'] = (X['session_duration'] - X['session_duration'].min()) / \
                               (X['session_duration'].max() - X['session_duration'].min())
        
        return X, y
    
    def build_baseline(self, X_train, y_train, X_val, y_val):
        """Baseline: Logistic Regression with simple threshold"""
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba, "Baseline (Logistic Regression)")
        
        self.models['baseline'] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return model, metrics
    
    def build_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest for non-linear patterns"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced'  # Handle imbalance
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba, "Random Forest")
        
        self.models['random_forest'] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': self._get_feature_importance(model)
        }
        
        return model, metrics
    
    def build_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Gradient Boosting for superior performance"""
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba, "Gradient Boosting")
        
        self.models['gradient_boosting'] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': self._get_feature_importance(model)
        }
        
        return model, metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """Calculate comprehensive evaluation metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'model_name': model_name,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
        
        return metrics
    
    def _get_feature_importance(self, model):
        """Extract feature importance from tree-based models"""
        importance = model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))
    
    def select_best_model(self):
        """Select best model based on FP reduction (primary) and F1 score (secondary)"""
        best_model = None
        best_metrics = None
        best_name = None
        
        # Primary: Minimize false positives, Secondary: Maximize F1
        for name, model_data in self.models.items():
            metrics = model_data['metrics']
            if best_model is None:
                best_model = model_data['model']
                best_metrics = metrics
                best_name = name
            else:
                # Compare: Lower FP rate is primary, then higher F1
                if (metrics['false_positive_rate'] < best_metrics['false_positive_rate'] or
                    (metrics['false_positive_rate'] == best_metrics['false_positive_rate'] and
                     metrics['f1'] > best_metrics['f1'])):
                    best_model = model_data['model']
                    best_metrics = metrics
                    best_name = name
        
        return best_name, best_model, best_metrics
    
    def train_pipeline(self):
        """Train all models and select the best"""
        print("=" * 70)
        print("PlaceMux Proctoring FP Reduction - Model Training Pipeline")
        print("=" * 70)
        
        # Generate data
        print("\n1. Generating synthetic proctoring data...")
        df = self.generate_synthetic_data(n_samples=2000)
        print(f"   - Total samples: {len(df)}")
        print(f"   - Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
        
        # Prepare data
        print("\n2. Preparing data...")
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        print(f"   - Training set: {len(X_train)} samples")
        print(f"   - Validation set: {len(X_val)} samples")
        print(f"   - Test set: {len(X_test)} samples")
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Build and evaluate models
        print("\n3. Training models...\n")
        
        print("Building Baseline Model (Logistic Regression):")
        self.build_baseline(X_train, y_train, X_val, y_val)
        self._print_metrics(self.models['baseline']['metrics'])
        
        print("\nBuilding Random Forest Model:")
        self.build_random_forest(X_train, y_train, X_val, y_val)
        self._print_metrics(self.models['random_forest']['metrics'])
        
        print("\nBuilding Gradient Boosting Model:")
        self.build_gradient_boosting(X_train, y_train, X_val, y_val)
        self._print_metrics(self.models['gradient_boosting']['metrics'])
        
        # Select best model
        print("\n" + "=" * 70)
        best_name, best_model, best_metrics = self.select_best_model()
        print(f"\nBEST MODEL SELECTED: {best_name.upper()}")
        print("=" * 70)
        print(f"False Positive Rate: {best_metrics['false_positive_rate']:.4f}")
        print(f"False Negatives: {best_metrics['false_negative_rate']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
        
        # Test on held-out test set
        print("\n4. Final evaluation on test set...")
        y_test_pred = best_model.predict(X_test)
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_pred_proba, 
                                               f"Best Model ({best_name}) - Test Set")
        self._print_metrics(test_metrics)
        
        self.best_model_name = best_name
        self.best_model = best_model
        self.best_metrics = best_metrics
        self.test_metrics = test_metrics
        self.X_test = X_test
        self.y_test = y_test
        
        return best_name, best_model, best_metrics
    
    def _print_metrics(self, metrics):
        """Pretty print metrics"""
        print(f"  Model: {metrics['model_name']}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1 Score:     {metrics['f1']:.4f}")
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']} ← (FP Reduction Focus)")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    
    def explain_prediction(self, input_dict):
        """
        Provide human-readable explanation for a prediction
        
        Example input:
        {
            'skill_match': 0.85,
            'session_duration': 0.6,
            'camera_available': 1,
            'env_quality': 0.9,
            'verification_confidence': 0.88,
            'completion_pct': 0.95,
            'answer_consistency': 0.92,
            'device_stability': 0.87
        }
        """
        X_input = np.array([input_dict[f] for f in self.feature_names]).reshape(1, -1)
        X_scaled = self.scaler.transform(X_input)
        
        prediction = self.best_model.predict(X_scaled)[0]
        probability = self.best_model.predict_proba(X_scaled)[0, 1]
        
        # Generate explanation
        explanation = {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2),
            'risk_factors': [],
            'positive_factors': []
        }
        
        # Identify risk and positive factors
        if input_dict['skill_match'] < 0.4:
            explanation['risk_factors'].append('Low skill match score')
        elif input_dict['skill_match'] > 0.8:
            explanation['positive_factors'].append('Strong skill match')
        
        if input_dict['session_duration'] < 0.2:
            explanation['risk_factors'].append('Completed too quickly (suspicious)')
        elif input_dict['session_duration'] > 0.7:
            explanation['positive_factors'].append('Reasonable session duration')
        
        if input_dict['camera_available'] == 0:
            explanation['risk_factors'].append('Camera not available during session')
        else:
            explanation['positive_factors'].append('Camera available throughout')
        
        if input_dict['answer_consistency'] < 0.3:
            explanation['risk_factors'].append('Answers are inconsistent')
        elif input_dict['answer_consistency'] > 0.8:
            explanation['positive_factors'].append('Highly consistent answers')
        
        if input_dict['env_quality'] < 0.4:
            explanation['risk_factors'].append('Poor environment quality')
        elif input_dict['env_quality'] > 0.8:
            explanation['positive_factors'].append('High environment quality')
        
        if input_dict['verification_confidence'] < 0.5:
            explanation['risk_factors'].append('Low verification confidence')
        elif input_dict['verification_confidence'] > 0.85:
            explanation['positive_factors'].append('High verification confidence')
        
        return explanation


if __name__ == "__main__":
    pipeline = FPReductionPipeline()
    best_name, best_model, best_metrics = pipeline.train_pipeline()
    
    # Example prediction with explanation
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION WITH EXPLANATION")
    print("=" * 70)
    
    example_case_1 = {
        'skill_match': 0.85,
        'session_duration': 0.6,
        'camera_available': 1,
        'env_quality': 0.9,
        'verification_confidence': 0.88,
        'completion_pct': 0.95,
        'answer_consistency': 0.92,
        'device_stability': 0.87
    }
    
    explanation = pipeline.explain_prediction(example_case_1)
    print(f"\nCase 1: High-confidence legitimate session")
    print(f"Prediction: {explanation['prediction']}")
    print(f"Fraud Probability: {explanation['fraud_probability']:.4f}")
    print(f"Confidence Level: {explanation['confidence']:.4f}")
    print(f"Positive Factors: {', '.join(explanation['positive_factors'])}")
    
    example_case_2 = {
        'skill_match': 0.3,
        'session_duration': 0.15,
        'camera_available': 0,
        'env_quality': 0.7,
        'verification_confidence': 0.75,
        'completion_pct': 0.85,
        'answer_consistency': 0.25,
        'device_stability': 0.6
    }
    
    explanation = pipeline.explain_prediction(example_case_2)
    print(f"\nCase 2: Suspicious session with fraud indicators")
    print(f"Prediction: {explanation['prediction']}")
    print(f"Fraud Probability: {explanation['fraud_probability']:.4f}")
    print(f"Confidence Level: {explanation['confidence']:.4f}")
    print(f"Risk Factors: {', '.join(explanation['risk_factors'])}")
