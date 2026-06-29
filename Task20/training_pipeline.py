"""
PlaceMux Training Pipeline
Train, evaluate, and select best recommendation model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import logging

from data_generator import PlaceMuxDataGenerator
from models import (
    LogisticRegressionModel, RandomForestModel, GradientBoostingModel,
    SVMModel, ModelEnsemble, ModelSelector
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline"""

    def __init__(self, data_dir='data', model_dir='models', seed=42):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.seed = seed
        self.results = {}

        os.makedirs(model_dir, exist_ok=True)

    def load_or_generate_data(self, regenerate=False):
        """Load existing data or generate new data"""
        if not regenerate and os.path.exists(f'{self.data_dir}/matches.csv'):
            logger.info("Loading existing data...")
            self.matches_df = pd.read_csv(f'{self.data_dir}/matches.csv')
        else:
            logger.info("Generating new data...")
            generator = PlaceMuxDataGenerator(seed=self.seed)
            students = generator.generate_students(n_students=200)
            jobs = generator.generate_jobs(n_jobs=150)
            matches = generator.generate_matches(students, jobs, n_samples=800)
            _, _, self.matches_df = generator.save_data(students, jobs, matches)

        logger.info(f"Loaded {len(self.matches_df)} match records")
        logger.info(f"Class distribution:\n{self.matches_df['is_good_match'].value_counts()}")

        return self.matches_df

    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.matches_df.drop('is_good_match', axis=1),
            self.matches_df['is_good_match'],
            test_size=test_size,
            random_state=self.seed,
            stratify=self.matches_df['is_good_match']
        )

        # Second split: train vs val from temp
        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_split,
            random_state=self.seed,
            stratify=y_temp
        )

        logger.info(f"Train set: {len(X_train)}")
        logger.info(f"Val set: {len(X_val)}")
        logger.info(f"Test set: {len(X_test)}")

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_baseline_model(self):
        """Train simple baseline model"""
        logger.info("\n" + "=" * 50)
        logger.info("BASELINE MODEL: Simple Skill Overlap")
        logger.info("=" * 50)

        # Baseline: just use tech overlap as predictor
        y_pred = (self.X_test['tech_overlap'] > 0.5).astype(int)
        y_pred_proba = self.X_test['tech_overlap'].values

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        baseline_metrics = {
            'model': 'Baseline (Tech Overlap > 0.5)',
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'accuracy': np.mean(y_pred == self.y_test),
        }

        logger.info(f"Baseline F1 Score: {baseline_metrics['f1']:.4f}")

        return baseline_metrics

    def train_all_models(self):
        """Train all models"""
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 50)

        models = [
            LogisticRegressionModel(),
            RandomForestModel(n_estimators=100),
            GradientBoostingModel(n_estimators=100),
            SVMModel()
        ]

        for model in models:
            logger.info(f"\nTraining {model.model_name}...")
            model.fit(self.X_train, self.y_train)
            metrics = model.evaluate(self.X_test, self.y_test)
            model.metrics = metrics

            # Save model
            model.save(f'{self.model_dir}/{model.model_name.replace(" ", "_")}.pkl')

        self.models = models
        return models

    def compare_models(self):
        """Compare all trained models"""
        logger.info("\n" + "=" * 50)
        logger.info("MODEL COMPARISON & SELECTION")
        logger.info("=" * 50)

        results = ModelSelector.compare_models(self.models, self.X_test, self.y_test)
        self.results['model_comparison'] = results

        # Get best model
        best_model_name = ModelSelector.get_best_model(results)
        self.best_model = next(m for m in self.models if m.model_name == best_model_name)

        logger.info(f"\n✓ BEST MODEL: {self.best_model.model_name}")
        logger.info(f"  F1 Score: {self.best_model.metrics['f1']:.4f}")

        return results

    def create_ensemble(self):
        """Create weighted ensemble from all models"""
        logger.info("\n" + "=" * 50)
        logger.info("CREATING MODEL ENSEMBLE")
        logger.info("=" * 50)

        ensemble = ModelEnsemble(self.models)
        ensemble.fit_weights(self.X_val, self.y_val)

        ensemble_metrics = ensemble.evaluate(self.X_test, self.y_test)
        logger.info(f"Ensemble F1 Score: {ensemble_metrics['f1']:.4f}")

        self.ensemble = ensemble
        return ensemble_metrics

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("\n" + "=" * 50)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("=" * 50)

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'total_samples': len(self.matches_df),
                'train_size': len(self.X_train),
                'val_size': len(self.X_val),
                'test_size': len(self.X_test),
                'positive_class': int(self.y_test.sum()),
                'negative_class': int((1 - self.y_test).sum()),
            },
            'models': {}
        }

        # Add baseline
        baseline = self.train_baseline_model()
        report['baseline'] = baseline

        # Add individual models
        for model in self.models:
            report['models'][model.model_name] = {
                'metrics': model.metrics,
                'feature_importance': None
            }

            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                report['models'][model.model_name]['feature_importance'] = importance.to_dict('records')

        # Add ensemble
        report['ensemble'] = {
            'model': 'Weighted Ensemble',
            'weights': self.ensemble.ensemble_weights,
            'metrics': {
                'precision': float(self.ensemble.evaluate(self.X_test, self.y_test)['precision']),
                'recall': float(self.ensemble.evaluate(self.X_test, self.y_test)['recall']),
                'f1': float(self.ensemble.evaluate(self.X_test, self.y_test)['f1']),
                'roc_auc': float(self.ensemble.evaluate(self.X_test, self.y_test)['roc_auc']),
                'accuracy': float(self.ensemble.evaluate(self.X_test, self.y_test)['accuracy']),
            }
        }

        # Add best model
        report['best_model'] = {
            'name': self.best_model.model_name,
            'metrics': {k: float(v) if isinstance(v, np.number) else v
                        for k, v in self.best_model.metrics.items()}
        }

        # Save report
        report_path = f'{self.model_dir}/evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        return report

    def run(self, regenerate_data=False):
        """Run complete training pipeline"""
        logger.info("Starting PlaceMux Training Pipeline...")

        # Load data
        self.load_or_generate_data(regenerate=regenerate_data)

        # Split data
        self.split_data()

        # Train models
        self.train_all_models()

        # Compare models
        self.compare_models()

        # Create ensemble
        self.create_ensemble()

        # Generate report
        report = self.generate_report()

        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 50)

        return report


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    report = pipeline.run(regenerate_data=True)

    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ Best model: {report['best_model']['name']}")
    print(f"✓ F1 Score: {report['best_model']['metrics']['f1']:.4f}")