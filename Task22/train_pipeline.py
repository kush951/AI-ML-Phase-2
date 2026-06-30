"""
PlaceMux ML Pipeline - Drift Detection & Retraining
Multiple models comparison and selection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """Complete ML pipeline with model comparison and drift detection"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.training_timestamp = None
        
    def generate_synthetic_data(self, n_samples=1000, n_features=10):
        """Generate realistic synthetic data for job-skill matching"""
        np.random.seed(self.random_state)
        
        # Create feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Create realistic labels with some structure
        # Job match = 1 if skill overlap is good
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02, 0.0])
        y = (X @ weights + np.random.randn(n_samples) * 0.5) > 0
        
        feature_names = [
            'skill_overlap_score', 'experience_years', 'certification_match',
            'location_proximity', 'salary_fit', 'education_level',
            'soft_skills_match', 'project_relevance', 'growth_potential', 'cultural_fit'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['match'] = y.astype(int)
        
        return df, feature_names
    
    def create_drift_dataset(self, df, drift_type='gradual', contamination=0.3):
        """Create drifted dataset to test drift detection"""
        df_drifted = df.copy()
        
        if drift_type == 'gradual':
            # Gradual drift - shift feature distributions
            drift_features = df_drifted.columns[:-1][:5]
            for feat in drift_features:
                df_drifted[feat] = df_drifted[feat] + np.random.normal(0.5, 0.2, len(df_drifted))
        
        elif drift_type == 'sudden':
            # Sudden drift - flip some labels
            idx = np.random.choice(len(df_drifted), size=int(len(df_drifted)*contamination), replace=False)
            df_drifted.loc[idx, 'match'] = 1 - df_drifted.loc[idx, 'match']
        
        return df_drifted
    
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df.drop('match', axis=1).values
        y = df['match'].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled, 'y_train': y_train,
            'X_val': X_val_scaled, 'y_val': y_val,
            'X_test': X_test_scaled, 'y_test': y_test,
            'scaler': scaler
        }
    
    def train_baseline(self, data):
        """Baseline model - Logistic Regression"""
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(data['X_train'], data['y_train'])
        
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        metrics = self._calculate_metrics(data['y_test'], y_pred, y_pred_proba)
        
        return model, metrics
    
    def train_random_forest(self, data):
        """Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
        )
        model.fit(data['X_train'], data['y_train'])
        
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        metrics = self._calculate_metrics(data['y_test'], y_pred, y_pred_proba)
        
        return model, metrics
    
    def train_gradient_boosting(self, data):
        """Gradient Boosting model"""
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=self.random_state
        )
        model.fit(data['X_train'], data['y_train'])
        
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        metrics = self._calculate_metrics(data['y_test'], y_pred, y_pred_proba)
        
        return model, metrics
    
    def train_svm(self, data):
        """Support Vector Machine model"""
        model = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        model.fit(data['X_train'], data['y_train'])
        
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        metrics = self._calculate_metrics(data['y_test'], y_pred, y_pred_proba)
        
        return model, metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': np.mean(y_pred == y_true),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
        
        return metrics
    
    def run_full_pipeline(self, n_samples=1000):
        """Run complete training pipeline with all models"""
        print("="*60)
        print("PlaceMux ML Pipeline - Full Training")
        print("="*60)
        
        # Generate data
        print("\n1. Generating synthetic data...")
        df, feature_names = self.generate_synthetic_data(n_samples=n_samples)
        data = self.prepare_data(df)
        print(f"   ✓ Data prepared: {data['X_train'].shape}")
        
        # Train models
        print("\n2. Training multiple models...")
        
        print("   - Training Baseline (Logistic Regression)...")
        baseline_model, baseline_metrics = self.train_baseline(data)
        self.models['Logistic Regression'] = baseline_model
        self.metrics['Logistic Regression'] = baseline_metrics
        self.scalers['Logistic Regression'] = data['scaler']
        
        print("   - Training Random Forest...")
        rf_model, rf_metrics = self.train_random_forest(data)
        self.models['Random Forest'] = rf_model
        self.metrics['Random Forest'] = rf_metrics
        self.scalers['Random Forest'] = data['scaler']
        
        print("   - Training Gradient Boosting...")
        gb_model, gb_metrics = self.train_gradient_boosting(data)
        self.models['Gradient Boosting'] = gb_model
        self.metrics['Gradient Boosting'] = gb_metrics
        self.scalers['Gradient Boosting'] = data['scaler']
        
        print("   - Training SVM...")
        svm_model, svm_metrics = self.train_svm(data)
        self.models['SVM'] = svm_model
        self.metrics['SVM'] = svm_metrics
        self.scalers['SVM'] = data['scaler']
        
        # Select best model
        print("\n3. Model Comparison & Selection...")
        self._select_best_model()
        
        # Print results
        self._print_results()
        
        self.training_timestamp = datetime.now().isoformat()
        
        return self.models, self.metrics, self.best_model_name
    
    def _select_best_model(self):
        """Select best model based on F1 score"""
        f1_scores = {name: metrics['f1_score'] for name, metrics in self.metrics.items()}
        self.best_model_name = max(f1_scores, key=f1_scores.get)
        self.best_model = self.models[self.best_model_name]
    
    def _print_results(self):
        """Print detailed results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        results_df = pd.DataFrame(self.metrics).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'false_positive_rate']]
        
        print("\n", results_df.to_string())
        
        print("\n" + "="*80)
        print(f"BEST MODEL SELECTED: {self.best_model_name}")
        print("="*80)
        best_metrics = self.metrics[self.best_model_name]
        print(f"Accuracy:            {best_metrics['accuracy']:.4f}")
        print(f"Precision:           {best_metrics['precision']:.4f}")
        print(f"Recall:              {best_metrics['recall']:.4f}")
        print(f"F1 Score:            {best_metrics['f1_score']:.4f}")
        print(f"AUC-ROC:             {best_metrics['auc_roc']:.4f}")
        print(f"False Positive Rate:  {best_metrics['false_positive_rate']:.4f}")
        
    def save_models(self, path='./models'):
        """Save all models and scalers"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            with open(f"{path}/{name.replace(' ', '_')}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            with open(f"{path}/{name.replace(' ', '_')}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'timestamp': self.training_timestamp,
            'metrics': self.metrics
        }
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Models saved to {path}")
    
    def load_models(self, path='./models'):
        """Load saved models"""
        import os
        
        model_files = [f for f in os.listdir(path) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
            with open(f"{path}/{model_file}", 'rb') as f:
                self.models[model_name] = pickle.load(f)
            
            scaler_file = model_file.replace('_model.pkl', '_scaler.pkl')
            with open(f"{path}/{scaler_file}", 'rb') as f:
                self.scalers[model_name] = pickle.load(f)
        
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
            self.best_model_name = metadata['best_model']
            self.best_model = self.models[self.best_model_name]
            self.training_timestamp = metadata['timestamp']
            self.metrics = metadata['metrics']
        
        print(f"✓ Models loaded from {path}")
    
    def explain_prediction(self, features, feature_names):
        """Provide explainable prediction"""
        scaler = self.scalers[self.best_model_name]
        features_scaled = scaler.transform([features])
        
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0][1]
        
        # Feature importance explanation
        explanation = {
            'match': bool(prediction),
            'confidence': float(probability),
            'model_used': self.best_model_name,
            'prediction_text': f"{'Match Found' if prediction else 'No Match'} (Confidence: {probability:.2%})"
        }
        
        return explanation


if __name__ == "__main__":
    # Run pipeline
    pipeline = MLPipeline()
    pipeline.run_full_pipeline(n_samples=1000)
    pipeline.save_models()
