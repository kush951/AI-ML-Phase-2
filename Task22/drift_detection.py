"""
Drift Detection & Monitoring Module
Detects when model performance degrades and triggers retraining
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    """Monitors data drift and model performance drift"""
    
    def __init__(self, baseline_mean=None, baseline_std=None, threshold=2.0):
        """
        Initialize drift detector
        
        Parameters:
        - baseline_mean: Mean of baseline features
        - baseline_std: Std of baseline features
        - threshold: Z-score threshold for drift detection (default 2.0 = 95% confidence)
        """
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.threshold = threshold
        self.drift_history = []
        self.performance_history = []
        self.last_training_time = datetime.now()
        self.retraining_needed = False
        
    def set_baseline(self, X_train):
        """Set baseline statistics from training data"""
        self.baseline_mean = np.mean(X_train, axis=0)
        self.baseline_std = np.std(X_train, axis=0)
        print(f"✓ Baseline set: mean shape {self.baseline_mean.shape}, std shape {self.baseline_std.shape}")
    
    def detect_statistical_drift(self, X_current, window_size=100):
        """
        Detect statistical drift using Kolmogorov-Smirnov test concept
        """
        if self.baseline_mean is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Calculate z-scores for current data
        z_scores = np.abs((X_current - self.baseline_mean) / (self.baseline_std + 1e-10))
        
        # Calculate percentage of features exceeding threshold
        drift_indices = np.where(np.mean(z_scores > self.threshold, axis=0) > 0.05)[0]
        drift_magnitude = np.mean(z_scores)
        
        drift_detected = len(drift_indices) > 0
        
        drift_record = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'drift_magnitude': float(drift_magnitude),
            'affected_features': int(len(drift_indices)),
            'z_score_threshold': self.threshold
        }
        
        self.drift_history.append(drift_record)
        
        return drift_detected, drift_magnitude, drift_indices
    
    def detect_performance_drift(self, y_true, y_pred, window_size=50):
        """
        Detect performance drift by monitoring key metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Calculate metrics for recent window
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(y_true)
        }
        
        self.performance_history.append(performance_record)
        
        # Detect drift if metrics degraded significantly
        if len(self.performance_history) > 1:
            prev_f1 = self.performance_history[-2]['f1_score']
            f1_drop = prev_f1 - f1
            
            # Drift if F1 drops by more than 5%
            if f1_drop > 0.05:
                performance_record['drift_detected'] = True
                self.retraining_needed = True
                return True, f1_drop
        
        performance_record['drift_detected'] = False
        return False, 0.0
    
    def should_retrain(self, days_since_training=None, force=False):
        """
        Determine if model should be retrained
        Based on: time elapsed, drift detected, or forced flag
        """
        if force:
            return True, "Forced retraining"
        
        if self.retraining_needed:
            return True, "Performance drift detected"
        
        if days_since_training is not None:
            days_elapsed = days_since_training
            if days_elapsed > 7:  # Retrain weekly by default
                return True, f"Weekly retraining: {days_elapsed} days elapsed"
        
        return False, "No retraining needed"
    
    def get_drift_report(self):
        """Generate comprehensive drift report"""
        recent_drift = self.drift_history[-10:] if self.drift_history else []
        recent_performance = self.performance_history[-10:] if self.performance_history else []
        
        report = {
            'total_drift_events': len(self.drift_history),
            'recent_drifts': recent_drift,
            'total_performance_checks': len(self.performance_history),
            'recent_performance': recent_performance,
            'retraining_needed': self.retraining_needed,
            'last_training_time': self.last_training_time.isoformat(),
            'drift_threshold': self.threshold
        }
        
        return report
    
    def reset_drift_flag(self):
        """Reset retraining flag after retraining"""
        self.retraining_needed = False
        self.last_training_time = datetime.now()


class AutoRetrainer:
    """Handles automatic retraining pipeline"""
    
    def __init__(self, pipeline, drift_detector, retrain_interval_days=7):
        """
        Initialize auto-retrainer
        
        Parameters:
        - pipeline: MLPipeline instance
        - drift_detector: DriftDetector instance
        - retrain_interval_days: Days between automatic retrainings
        """
        self.pipeline = pipeline
        self.drift_detector = drift_detector
        self.retrain_interval_days = retrain_interval_days
        self.retrain_history = []
    
    def check_and_retrain(self, new_data, feature_names, force=False):
        """
        Check if retraining is needed and perform if necessary
        """
        print("\n" + "="*60)
        print("AutoRetrainer - Checking if retraining is needed...")
        print("="*60)
        
        should_retrain, reason = self.drift_detector.should_retrain(force=force)
        
        print(f"Retraining needed: {should_retrain} ({reason})")
        
        if not should_retrain:
            return False, "No retraining needed"
        
        print("\nStarting retraining process...")
        
        # Retrain model
        try:
            self.pipeline.run_full_pipeline(n_samples=len(new_data))
            self.drift_detector.reset_drift_flag()
            
            retrain_record = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'samples_used': len(new_data),
                'best_model': self.pipeline.best_model_name,
                'best_f1_score': self.pipeline.metrics[self.pipeline.best_model_name]['f1_score']
            }
            
            self.retrain_history.append(retrain_record)
            
            print(f"✓ Retraining completed successfully")
            print(f"  Best model: {self.pipeline.best_model_name}")
            print(f"  F1 Score: {retrain_record['best_f1_score']:.4f}")
            
            return True, retrain_record
            
        except Exception as e:
            print(f"✗ Retraining failed: {str(e)}")
            return False, str(e)
    
    def get_retrain_history(self):
        """Get retraining history"""
        return self.retrain_history


if __name__ == "__main__":
    print("Drift Detection & Monitoring Module")
    print("Use this module with the MLPipeline for drift detection")
