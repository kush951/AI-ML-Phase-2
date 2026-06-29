"""
PlaceMux ML Pipeline
Main orchestration script for student-job matching intelligence
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Import custom modules
from data_generator import DataGenerator
from feature_engineering import FeatureEngineer
from matching_models import MatchingModelEnsemble
from explainability import MatchExplainer, QualityAnalyzer
from recruiter_views import RecruiterDashboard, AdminPanel

class PlaceMuxPipeline:
    """End-to-end ML pipeline for PlaceMux"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.students = None
        self.jobs = None
        self.matches = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.feature_engineer = None
        self.model_ensemble = None
        self.explainer = None
        self.quality_analyzer = None
        self.recruiter_dashboard = None
        self.admin_panel = None
        
        self.results = {}
    
    def run_pipeline(self, n_students: int = 150, n_jobs: int = 60, 
                    n_matches: int = 300) -> Dict:
        """Run complete ML pipeline"""
        
        print("\n" + "=" * 80)
        print("PlaceMux AI/ML Pipeline - Bulk Onboarding & Recruiter Views")
        print("=" * 80)
        
        # Stage A: Data Generation
        print("\n📦 STAGE A: DATA GENERATION")
        print("-" * 80)
        self._stage_data_generation(n_students, n_jobs, n_matches)
        
        # Stage B: Feature Engineering
        print("\n🔧 STAGE B: FEATURE ENGINEERING")
        print("-" * 80)
        self._stage_feature_engineering()
        
        # Stage C: Model Building & Selection
        print("\n🤖 STAGE C: MODEL BUILDING & SELECTION")
        print("-" * 80)
        self._stage_model_building()
        
        # Stage D: Explainability & Quality Analysis
        print("\n🔍 STAGE D: QUALITY ANALYSIS & EXPLAINABILITY")
        print("-" * 80)
        self._stage_quality_analysis()
        
        # Stage E: Recruiter Views & Analytics
        print("\n📊 STAGE E: RECRUITER VIEWS & ANALYTICS")
        print("-" * 80)
        self._stage_recruiter_views()
        
        # Stage F: Admin Panel & Recommendations
        print("\n👨‍💼 STAGE F: ADMIN PANEL & RECOMMENDATIONS")
        print("-" * 80)
        self._stage_admin_panel()
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return self.results
    
    def _stage_data_generation(self, n_students: int, n_jobs: int, n_matches: int):
        """Generate synthetic but realistic data"""
        
        print(f"Generating {n_students} students, {n_jobs} jobs, {n_matches} matches...")
        
        self.students, self.jobs, self.matches = DataGenerator.generate_full_dataset(
            n_students=n_students,
            n_jobs=n_jobs,
            n_matches=n_matches
        )
        
        print(f"✓ Generated data:")
        print(f"  • Students: {len(self.students)}")
        print(f"  • Jobs: {len(self.jobs)}")
        print(f"  • Matches: {len(self.matches)}")
        print(f"  • Positive matches: {self.matches['is_good_match'].sum()} ({self.matches['is_good_match'].mean():.1%})")
        
        self.results['data_generation'] = {
            'students_count': len(self.students),
            'jobs_count': len(self.jobs),
            'matches_count': len(self.matches),
            'positive_ratio': float(self.matches['is_good_match'].mean())
        }
    
    def _stage_feature_engineering(self):
        """Extract and engineer features"""
        
        print("Engineering features...")
        
        self.feature_engineer = FeatureEngineer()
        self.X, self.y, feature_names = self.feature_engineer.extract_features(
            self.students, self.jobs, self.matches
        )
        
        print(f"✓ Extracted {self.X.shape[1]} features from {self.X.shape[0]} samples")
        print(f"  Features: {', '.join(feature_names[:5])}... (showing first 5)")
        
        # Train-val-test split (60-20-20)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"✓ Data split: Train {len(self.X_train)} | Val {len(self.X_val)} | Test {len(self.X_test)}")
        
        self.results['feature_engineering'] = {
            'total_features': self.X.shape[1],
            'total_samples': self.X.shape[0],
            'feature_names': feature_names
        }
    
    def _stage_model_building(self):
        """Build and evaluate multiple models"""
        
        print("Building and evaluating multiple models...")
        
        self.model_ensemble = MatchingModelEnsemble(random_state=self.random_state)
        metrics = self.model_ensemble.build_models(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_train.shape[1]
        )
        
        # Print model evaluation summary
        print(self.model_ensemble.get_models_summary())
        
        # Test set evaluation
        print("\n📈 TEST SET EVALUATION")
        print("-" * 80)
        y_pred_test = self.model_ensemble.predict(self.X_test)
        y_pred_proba_test = self.model_ensemble.predict_proba(self.X_test)[:, 1]
        
        test_metrics = self.model_ensemble._evaluate_model(
            self.model_ensemble.best_model,
            self.X_test,
            self.y_test,
            is_scaled=True if self.model_ensemble.best_model_name != 'Baseline' else False
        )
        
        print(f"\n{self.model_ensemble.best_model_name} - Test Performance:")
        print(f"  • Accuracy:  {test_metrics.accuracy:.4f}")
        print(f"  • Precision: {test_metrics.precision:.4f}")
        print(f"  • Recall:    {test_metrics.recall:.4f}")
        print(f"  • F1-Score:  {test_metrics.f1:.4f}")
        print(f"  • AUC-ROC:   {test_metrics.auc_roc:.4f}")
        print(f"  • False Positive Rate: {test_metrics.false_positive_rate:.4f}")
        
        # Feature importance
        if self.model_ensemble.best_model_name != 'Baseline':
            importances = self.model_ensemble.get_feature_importance()
            print(f"\n🎯 Top Features ({self.model_ensemble.best_model_name}):")
            for feature, importance in list(importances.items())[:5]:
                print(f"  • {feature}: {importance:.4f}")
        
        self.results['model_evaluation'] = {
            'best_model': self.model_ensemble.best_model_name,
            'test_metrics': test_metrics.to_dict(),
            'model_comparison': self.model_ensemble.get_model_comparison().to_dict()
        }
    
    def _stage_quality_analysis(self):
        """Analyze match quality and generate explanations"""
        
        print("Analyzing match quality and generating explanations...")
        
        # Get test predictions
        y_pred_test = self.model_ensemble.predict(self.X_test)
        y_pred_proba_test = self.model_ensemble.predict_proba(self.X_test)[:, 1]
        
        # Quality analysis
        self.quality_analyzer = QualityAnalyzer()
        quality_report = self.quality_analyzer.analyze_item_quality(
            self.students, self.jobs,
            y_pred_test, y_pred_proba_test, self.y_test
        )
        
        print(f"\n📋 QUALITY REPORT:")
        print(f"  • Weak students identified: {len(quality_report['weak_students'])}")
        print(f"  • Weak jobs identified: {len(quality_report['weak_jobs'])}")
        print(f"  • Average match confidence: {quality_report['quality_metrics']['avg_match_confidence']:.1%}")
        print(f"  • Median match confidence: {quality_report['quality_metrics']['median_match_confidence']:.1%}")
        
        flags = self.quality_analyzer.flag_items_for_review(quality_report)
        if flags:
            print(f"\n⚠️  QUALITY FLAGS:")
            for flag in flags:
                print(f"  {flag}")
        
        self.results['quality_analysis'] = {
            'weak_students_count': len(quality_report['weak_students']),
            'weak_jobs_count': len(quality_report['weak_jobs']),
            'quality_metrics': quality_report['quality_metrics'],
            'flags': flags
        }
    
    def _stage_recruiter_views(self):
        """Generate recruiter analytics and dashboards"""
        
        print("Generating recruiter views...")
        
        y_pred_test = self.model_ensemble.predict(self.X_test)
        y_pred_proba_test = self.model_ensemble.predict_proba(self.X_test)[:, 1]
        
        self.recruiter_dashboard = RecruiterDashboard()
        
        # Job analytics
        job_analytics = self.recruiter_dashboard.job_analytics(
            self.jobs, y_pred_test, y_pred_proba_test, self.matches
        )
        
        # Student analytics
        student_analytics = self.recruiter_dashboard.student_analytics(
            self.students, y_pred_test, y_pred_proba_test, self.matches
        )
        
        # Onboarding summary
        onboarding_summary = self.recruiter_dashboard.bulk_onboarding_summary(
            self.students
        )
        
        print(f"✓ Job analytics generated for {len(job_analytics)} jobs")
        print(f"✓ Student analytics generated for {len(student_analytics)} students")
        print(f"✓ Onboarding summary:")
        print(f"  • Total students: {onboarding_summary['total_students']}")
        print(f"  • Verified: {onboarding_summary['verified_students']} ({onboarding_summary['verification_rate']:.1%})")
        print(f"  • Avg GPA: {onboarding_summary['avg_gpa']:.2f}")
        print(f"  • Colleges: {onboarding_summary['colleges_represented']}")
        print(f"  • Unique skills: {onboarding_summary['unique_skills']}")
        
        self.results['recruiter_analytics'] = {
            'jobs_analyzed': len(job_analytics),
            'students_analyzed': len(student_analytics),
            'onboarding_summary': onboarding_summary
        }
    
    def _stage_admin_panel(self):
        """Generate admin panel and recommendations"""
        
        print("Generating admin recommendations...")
        
        y_pred_test = self.model_ensemble.predict(self.X_test)
        y_pred_proba_test = self.model_ensemble.predict_proba(self.X_test)[:, 1]
        
        self.admin_panel = AdminPanel()
        
        # Get weak items
        quality_analyzer = QualityAnalyzer()
        quality_report = quality_analyzer.analyze_item_quality(
            self.students, self.jobs,
            y_pred_test, y_pred_proba_test, self.y_test
        )
        
        admin_report = self.admin_panel.generate_admin_report(
            self.students, self.jobs,
            y_pred_test, y_pred_proba_test,
            quality_report['weak_students'],
            quality_report['weak_jobs']
        )
        
        print(f"\n✓ System Health:")
        print(f"  • Registered: {admin_report['system_health']['students_registered']}")
        print(f"  • Active Jobs: {admin_report['system_health']['active_jobs']}")
        print(f"  • Verified Students: {admin_report['system_health']['verified_students']}")
        
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(admin_report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        self.results['admin_panel'] = {
            'system_health': admin_report['system_health'],
            'data_quality': admin_report['data_quality'],
            'weak_items': {
                'weak_students': len(admin_report['item_bank_quality']['weak_students']),
                'weak_jobs': len(admin_report['item_bank_quality']['weak_jobs'])
            },
            'recommendations': admin_report['recommendations']
        }


# Example usage
if __name__ == "__main__":
    pipeline = PlaceMuxPipeline(random_state=42)
    results = pipeline.run_pipeline(
        n_students=150,
        n_jobs=60,
        n_matches=300
    )
    
    # Save results
    with open('pipeline_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    print("\n💾 Results saved to pipeline_results.json")
