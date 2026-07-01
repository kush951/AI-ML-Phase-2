"""
PlaceMux - Model Training
Train and evaluate multiple ML models for job matching
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import os
import json
from datetime import datetime
from data_generator import DataGenerator
from feature_store import FeatureStore


class ModelTrainer:
    """
    Train and evaluate multiple models for job matching
    """
    
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_store = FeatureStore()
        
        # Training data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def prepare_data(self, df: pd.DataFrame, test_size=0.2, val_size=0.2):
        """
        Prepare training, validation, and test splits
        """
        # Get features from feature store
        X, y = self.feature_store.get_features_for_training(df)
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print("✓ Data splits created:")
        print(f"  - Train: {len(self.X_train)} samples")
        print(f"  - Val:   {len(self.X_val)} samples")
        print(f"  - Test:  {len(self.X_test)} samples")
        print(f"  - Positive class: {(self.y_train == 1).sum() / len(self.y_train):.1%}")
    
    def train_logistic_regression(self):
        """Train Logistic Regression baseline"""
        print("\n[1/5] Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        self._evaluate_model('Logistic Regression', model)
    
    def train_random_forest(self):
        """Train Random Forest"""
        print("\n[2/5] Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        self._evaluate_model('Random Forest', model)
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting"""
        print("\n[3/5] Training Gradient Boosting...")
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['Gradient Boosting'] = model
        self._evaluate_model('Gradient Boosting', model)
    
    def train_svm(self):
        """Train Support Vector Machine"""
        print("\n[4/5] Training Support Vector Machine...")
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['SVM'] = model
        self._evaluate_model('SVM', model)
    
    def train_neural_network(self):
        """Train Neural Network"""
        print("\n[5/5] Training Neural Network...")
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['Neural Network'] = model
        self._evaluate_model('Neural Network', model)
    
    def _evaluate_model(self, name: str, model):
        """Evaluate model on validation set"""
        y_pred = model.predict(self.X_val)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        self.results[name] = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_val, y_pred).tolist()
        }
        
        print(f"  Accuracy:  {self.results[name]['accuracy']:.4f}")
        print(f"  Precision: {self.results[name]['precision']:.4f}")
        print(f"  Recall:    {self.results[name]['recall']:.4f}")
        print(f"  F1-Score:  {self.results[name]['f1']:.4f}")
        print(f"  ROC-AUC:   {self.results[name]['roc_auc']:.4f}")
    
    def train_all_models(self, df: pd.DataFrame):
        """Train all models"""
        print("=" * 60)
        print("PHASE 1: PREPARING DATA")
        print("=" * 60)
        self.prepare_data(df)
        
        print("\n" + "=" * 60)
        print("PHASE 2: TRAINING MULTIPLE MODELS")
        print("=" * 60)
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_svm()
        self.train_neural_network()
    
    def select_best_model(self):
        """Select best model based on F1 score"""
        best_f1 = -1
        
        for name, metrics in self.results.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print("\n" + "=" * 60)
        print("PHASE 3: MODEL SELECTION")
        print("=" * 60)
        print(f"\n✓ Best Model Selected: {self.best_model_name}")
        print(f"  F1-Score on Validation Set: {self.results[self.best_model_name]['f1']:.4f}")
    
    def evaluate_on_test_set(self):
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No model selected. Run train_all_models first.")
        
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        test_results = {
            'model_name': self.best_model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        print("\n" + "=" * 60)
        print("PHASE 4: TEST SET EVALUATION")
        print("=" * 60)
        print(f"\nModel: {self.best_model_name}")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1-Score:  {test_results['f1']:.4f}")
        print(f"  ROC-AUC:   {test_results['roc_auc']:.4f}")
        
        return test_results
    
    def save_models(self):
        """Save all models to disk"""
        # Save best model
        with open(f'{self.model_dir}best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save results
        with open(f'{self.model_dir}results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save all models metadata
        models_info = {
            'best_model': self.best_model_name,
            'trained_models': list(self.models.keys()),
            'timestamp': datetime.now().isoformat()
        }
        with open(f'{self.model_dir}models_info.json', 'w') as f:
            json.dump(models_info, f, indent=2)
        
        print(f"\n✓ Models saved to {self.model_dir}")
        self.feature_store.save_feature_store()
    
    def print_model_comparison(self):
        """Print comparison of all models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print("\n{:<20} {:<10} {:<10} {:<10} {:<10}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1-Score"
        ))
        print("-" * 60)
        
        for name in sorted(self.results.keys()):
            metrics = self.results[name]
            print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                name,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1']
            ))


if __name__ == '__main__':
    print("PlaceMux - Model Training Pipeline")
    print("=" * 60)
    
    # Generate data
    print("Generating synthetic data...")
    generator = DataGenerator(n_students=300, n_jobs=150)
    generator.generate_students()
    generator.generate_jobs()
    generator.generate_matches()
    training_data = generator.create_training_data()
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_all_models(training_data)
    trainer.print_model_comparison()
    trainer.select_best_model()
    test_results = trainer.evaluate_on_test_set()
    trainer.save_models()
    
    print("\n✓ Training complete!")
