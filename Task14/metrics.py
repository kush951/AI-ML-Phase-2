import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MetricsEvaluator:
    """Comprehensive metrics evaluation and reporting"""
    
    def __init__(self, model_name, y_true, y_pred, y_proba=None):
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.metrics = {}
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all metrics"""
        
        self.metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['f1'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        self.metrics['true_positives'] = tp
        self.metrics['true_negatives'] = tn
        self.metrics['false_positives'] = fp
        self.metrics['false_negatives'] = fn
        
        # Calculate false positive rate and false negative rate
        self.metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        self.metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC-AUC if probabilities available
        if self.y_proba is not None and len(np.unique(self.y_true)) > 1:
            self.metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba)
        
        # Accuracy
        self.metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Specificity and Sensitivity
        self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def get_metrics_dict(self):
        """Return metrics as dictionary"""
        return self.metrics
    
    def print_report(self):
        """Print detailed metrics report"""
        print(f"\n{'='*60}")
        print(f"METRICS REPORT: {self.model_name}")
        print(f"{'='*60}")
        print(f"\nCore Metrics:")
        print(f"  Precision:          {self.metrics['precision']:.4f} (correct positive predictions)")
        print(f"  Recall:             {self.metrics['recall']:.4f} (coverage of actual positives)")
        print(f"  F1-Score:           {self.metrics['f1']:.4f} (harmonic mean)")
        print(f"  Accuracy:           {self.metrics['accuracy']:.4f} (overall correctness)")
        
        if 'roc_auc' in self.metrics:
            print(f"  ROC-AUC:            {self.metrics['roc_auc']:.4f} (discrimination ability)")
        
        print(f"\nConfusion Matrix Analysis:")
        print(f"  True Positives:     {self.metrics['true_positives']}")
        print(f"  True Negatives:     {self.metrics['true_negatives']}")
        print(f"  False Positives:    {self.metrics['false_positives']}")
        print(f"  False Negatives:    {self.metrics['false_negatives']}")
        
        print(f"\nError Rates:")
        print(f"  False Positive Rate: {self.metrics['false_positive_rate']:.4f} (risk of false matches)")
        print(f"  False Negative Rate: {self.metrics['false_negative_rate']:.4f} (missed matches)")
        print(f"  Specificity:        {self.metrics['specificity']:.4f} (correct negative rate)")
        print(f"  Sensitivity:        {self.metrics['sensitivity']:.4f} (correct positive rate)")
        print(f"{'='*60}\n")
    
    def generate_metrics_df(self):
        """Generate DataFrame of metrics"""
        return pd.DataFrame([self.metrics], index=[self.model_name])


class ComparisonReport:
    """Compare multiple models side-by-side"""
    
    def __init__(self):
        self.evaluators = []
        self.comparison_df = None
    
    def add_evaluator(self, evaluator):
        """Add evaluator to comparison"""
        self.evaluators.append(evaluator)
    
    def generate_comparison(self):
        """Generate comparison dataframe"""
        dfs = [eval.generate_metrics_df() for eval in self.evaluators]
        self.comparison_df = pd.concat(dfs)
        return self.comparison_df
    
    def print_comparison(self):
        """Print comparison table"""
        if self.comparison_df is None:
            self.generate_comparison()
        
        print(f"\n{'='*100}")
        print("MODEL COMPARISON REPORT")
        print(f"{'='*100}")
        print(self.comparison_df[['precision', 'recall', 'f1', 'accuracy', 'false_positive_rate', 'roc_auc']].to_string())
        print(f"{'='*100}\n")
    
    def plot_comparison(self, filepath='reports/model_comparison.png'):
        """Plot model comparison"""
        if self.comparison_df is None:
            self.generate_comparison()
        
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        comparison_subset = self.comparison_df[metrics]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot
        comparison_subset.T.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Metrics Comparison')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Metrics')
        axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1))
        axes[0].grid(axis='y', alpha=0.3)
        
        # Heatmap
        sns.heatmap(comparison_subset.T, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[1], 
                    cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
        axes[1].set_title('Model Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {filepath}")


def plot_confusion_matrix(y_true, y_pred, model_name, filepath='reports/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {filepath}")


def plot_roc_curve(y_true, y_proba, model_name, filepath='reports/roc_curve.png'):
    """Plot ROC curve"""
    if len(np.unique(y_true)) < 2:
        print("⚠ Skipping ROC curve (need both classes in data)")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {filepath}")


def plot_precision_recall_curve(y_true, y_proba, model_name, filepath='reports/pr_curve.png'):
    """Plot Precision-Recall curve"""
    if len(np.unique(y_true)) < 2:
        print("⚠ Skipping PR curve (need both classes in data)")
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ PR curve saved to {filepath}")


def create_baseline_comparison(baseline_metrics, other_metrics):
    """Compare metrics against baseline"""
    comparison = {}
    
    for key in baseline_metrics:
        if key in other_metrics:
            diff = other_metrics[key] - baseline_metrics[key]
            pct_change = (diff / baseline_metrics[key] * 100) if baseline_metrics[key] != 0 else 0
            comparison[key] = {
                'baseline': baseline_metrics[key],
                'improved': other_metrics[key],
                'difference': diff,
                'pct_change': pct_change
            }
    
    return comparison
