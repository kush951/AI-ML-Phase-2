"""
PlaceMux Fairness Audit
Detects bias and disparate impact across demographic groups (gender, region, background)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import json
from datetime import datetime

class FairnessAudit:
    def __init__(self, features_df, predictions_df):
        """
        Initialize fairness audit
        features_df: Original feature dataframe with demographic info
        predictions_df: Predictions with demographic breakdowns
        """
        self.features = features_df
        self.predictions = predictions_df
        self.audit_results = {}
        
    def compute_group_metrics(self, group_col):
        """Compute metrics for each demographic group"""
        print(f"\n📊 Analyzing {group_col}...")
        
        groups = self.predictions[group_col].unique()
        group_metrics = {}
        
        for group in sorted(groups):
            group_data = self.predictions[self.predictions[group_col] == group]
            
            if len(group_data) == 0:
                continue
            
            y_true = group_data['y_true']
            y_pred = group_data['y_pred']
            
            # Compute metrics
            accuracy = (y_true == y_pred).sum() / len(y_true)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # False negative rate and false positive rate
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Positive prediction rate (proportion recommended)
            ppr = (y_pred == 1).sum() / len(y_pred)
            
            group_metrics[str(group)] = {
                'count': len(group_data),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'fnr': float(fnr),  # False Negative Rate
                'fpr': float(fpr),  # False Positive Rate
                'ppr': float(ppr),  # Positive Prediction Rate
                'positive_actual': int((y_true == 1).sum()),
                'positive_predicted': int((y_pred == 1).sum()),
            }
            
            print(f"  {group_col}={group}: N={len(group_data)}, Accuracy={accuracy:.3f}, Recall={recall:.3f}, PPR={ppr:.3f}")
        
        return group_metrics
    
    def compute_disparate_impact(self, group_col, reference_group=None):
        """
        Compute disparate impact (4/5 rule)
        Disparate impact exists if: selection_rate_min / selection_rate_max < 0.8
        """
        print(f"\n⚖️  Computing Disparate Impact for {group_col}...")
        
        groups = self.predictions[group_col].unique()
        disparate_impact = {}
        
        selection_rates = {}
        for group in sorted(groups):
            group_data = self.predictions[self.predictions[group_col] == group]
            sr = (group_data['y_pred'] == 1).sum() / len(group_data) if len(group_data) > 0 else 0
            selection_rates[str(group)] = sr
        
        if len(selection_rates) < 2:
            return disparate_impact
        
        min_sr = min(selection_rates.values())
        max_sr = max(selection_rates.values())
        di_ratio = min_sr / max_sr if max_sr > 0 else 1.0
        
        disparate_impact['selection_rates'] = {k: float(v) for k, v in selection_rates.items()}
        disparate_impact['di_ratio'] = float(di_ratio)
        disparate_impact['has_disparate_impact'] = di_ratio < 0.8
        disparate_impact['interpretation'] = (
            f"Disparate Impact detected (ratio={di_ratio:.3f})" if di_ratio < 0.8 
            else f"No significant disparate impact (ratio={di_ratio:.3f})"
        )
        
        print(f"  {disparate_impact['interpretation']}")
        
        return disparate_impact
    
    def compute_predictive_parity(self, group_col):
        """Check if positive prediction rate is equal across groups"""
        print(f"\n🔄 Checking Predictive Parity for {group_col}...")
        
        groups = self.predictions[group_col].unique()
        pprs = {}
        
        for group in sorted(groups):
            group_data = self.predictions[self.predictions[group_col] == group]
            ppr = (group_data['y_pred'] == 1).sum() / len(group_data) if len(group_data) > 0 else 0
            pprs[str(group)] = ppr
        
        ppr_variance = np.var(list(pprs.values()))
        ppr_std = np.std(list(pprs.values()))
        
        result = {
            'pprs': {k: float(v) for k, v in pprs.items()},
            'variance': float(ppr_variance),
            'std_dev': float(ppr_std),
            'parity_ok': ppr_std < 0.1,  # Less than 10% variance is acceptable
        }
        
        print(f"  PPR Std Dev: {ppr_std:.4f} - {'✓ PASS' if result['parity_ok'] else '⚠ FAIL'}")
        
        return result
    
    def compute_equalized_odds(self, group_col):
        """Check if FPR and FNR are equal across groups"""
        print(f"\n⚗️  Checking Equalized Odds for {group_col}...")
        
        groups = self.predictions[group_col].unique()
        fprs = {}
        fnrs = {}
        
        for group in sorted(groups):
            group_data = self.predictions[self.predictions[group_col] == group]
            y_true = group_data['y_true']
            y_pred = group_data['y_pred']
            
            if len(np.unique(y_true)) < 2:
                continue
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            fprs[str(group)] = fpr
            fnrs[str(group)] = fnr
        
        fpr_std = np.std(list(fprs.values()))
        fnr_std = np.std(list(fnrs.values()))
        
        result = {
            'fprs': {k: float(v) for k, v in fprs.items()},
            'fnrs': {k: float(v) for k, v in fnrs.items()},
            'fpr_std_dev': float(fpr_std),
            'fnr_std_dev': float(fnr_std),
            'equalized_ok': fpr_std < 0.1 and fnr_std < 0.1,
        }
        
        print(f"  FPR Std Dev: {fpr_std:.4f} | FNR Std Dev: {fnr_std:.4f} - {'✓ PASS' if result['equalized_ok'] else '⚠ FAIL'}")
        
        return result
    
    def run_audit(self):
        """Run complete fairness audit"""
        print("=" * 60)
        print("PlaceMux FAIRNESS AUDIT - Starting...")
        print("=" * 60)
        
        demographic_attrs = ['gender', 'region', 'background']
        
        for attr in demographic_attrs:
            if attr not in self.predictions.columns:
                print(f"⚠ Skipping {attr} - not found in data")
                continue
            
            self.audit_results[attr] = {
                'group_metrics': self.compute_group_metrics(attr),
                'disparate_impact': self.compute_disparate_impact(attr),
                'predictive_parity': self.compute_predictive_parity(attr),
                'equalized_odds': self.compute_equalized_odds(attr),
            }
        
        return self.audit_results
    
    def generate_fairness_score(self):
        """Generate overall fairness score (0-100)"""
        print("\n" + "=" * 60)
        print("FAIRNESS SCORE CALCULATION")
        print("=" * 60)
        
        scores = []
        
        # Check disparate impact (25% weight)
        di_pass = 0
        for attr in self.audit_results:
            if self.audit_results[attr]['disparate_impact']['has_disparate_impact']:
                di_pass += 0
            else:
                di_pass += 1
        di_score = (di_pass / len(self.audit_results)) * 100 if self.audit_results else 0
        print(f"Disparate Impact Score: {di_score:.1f}/100 (25% weight = {di_score * 0.25:.1f})")
        
        # Check predictive parity (25% weight)
        pp_pass = 0
        for attr in self.audit_results:
            if self.audit_results[attr]['predictive_parity']['parity_ok']:
                pp_pass += 1
        pp_score = (pp_pass / len(self.audit_results)) * 100 if self.audit_results else 0
        print(f"Predictive Parity Score: {pp_score:.1f}/100 (25% weight = {pp_score * 0.25:.1f})")
        
        # Check equalized odds (25% weight)
        eo_pass = 0
        for attr in self.audit_results:
            if self.audit_results[attr]['equalized_odds']['equalized_ok']:
                eo_pass += 1
        eo_score = (eo_pass / len(self.audit_results)) * 100 if self.audit_results else 0
        print(f"Equalized Odds Score: {eo_score:.1f}/100 (25% weight = {eo_score * 0.25:.1f})")
        
        # Model accuracy (25% weight)
        accuracy = (self.predictions['y_true'] == self.predictions['y_pred']).sum() / len(self.predictions)
        acc_score = accuracy * 100
        print(f"Model Accuracy Score: {acc_score:.1f}/100 (25% weight = {acc_score * 0.25:.1f})")
        
        overall_score = (di_score * 0.25 + pp_score * 0.25 + eo_score * 0.25 + acc_score * 0.25)
        
        print(f"\n🎯 OVERALL FAIRNESS SCORE: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            status = "✓ EXCELLENT - Ready for deployment"
        elif overall_score >= 70:
            status = "⚠ GOOD - Monitor in production"
        elif overall_score >= 60:
            status = "⚠ FAIR - Significant bias detected, needs review"
        else:
            status = "✗ POOR - Do not deploy, major bias issues"
        
        print(f"   Status: {status}\n")
        
        return {
            'overall_score': float(overall_score),
            'di_score': float(di_score),
            'pp_score': float(pp_score),
            'eo_score': float(eo_score),
            'acc_score': float(acc_score),
            'status': status,
        }
    
    def save_report(self, output_file='fairness_audit_report.json'):
        """Save fairness audit report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'audit_results': self.audit_results,
            'fairness_score': self.generate_fairness_score(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Fairness audit report saved to {output_file}")
        return report


def run_fairness_audit_example():
    """Run fairness audit on example data"""
    # Load features
    features = pd.read_csv('data/features.csv')
    
    # Load or generate predictions
    import pickle
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prepare test data
    X_test = features.drop(['is_match', 'student_id', 'job_id', 'total_score'], axis=1)
    
    # Encode categorical
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = ['gender', 'region', 'background', 'company_size', 'urgency_level']
    for col in categorical_cols:
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create predictions dataframe
    predictions_df = features[['gender', 'region', 'background']].copy()
    predictions_df['y_true'] = features['is_match'].values
    predictions_df['y_pred'] = y_pred
    
    # Run audit
    audit = FairnessAudit(features, predictions_df)
    audit.run_audit()
    audit.save_report()


if __name__ == "__main__":
    print("Run fairness audit after training models")
