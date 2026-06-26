import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime

from data_generator import generate_sample_data
from feature_engineering import prepare_training_data
from models import (
    train_and_evaluate_models,
    select_best_model,
    save_model,
    load_model
)

from metrics import (
    MetricsEvaluator,
    ComparisonReport,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)


def main():
    """Main training pipeline"""

    print("=" * 70)
    print("PLACEMUX TASK 14: End-to-End Status Tracking & Parsing")
    print("Building AI/ML Pipeline for Skill Matching")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Create directories
    # ------------------------------------------------------------------

    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 - Generate Sample Data
    # ------------------------------------------------------------------

    print("\n[1/6] Generating sample data...")

    resumes_df, jobs_df, matches_df = generate_sample_data()

    print("✓ Sample data generated successfully")
    print(f"✓ Generated {len(resumes_df)} resumes and {len(jobs_df)} job postings")

    # ------------------------------------------------------------------
    # DEBUG MATCHES
    # ------------------------------------------------------------------

    print("\nChecking matches dataframe...")

    if "is_match" not in matches_df.columns:
        raise ValueError(
            "'is_match' column missing in matches_df.\n"
            "Please verify generate_sample_data() function."
        )

    print(matches_df.head())

    print("\nMissing values in matches_df:")
    print(matches_df.isnull().sum())

    # Fix NaN in is_match
    matches_df["is_match"] = matches_df["is_match"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # STEP 2 - Feature Engineering
    # ------------------------------------------------------------------

    print("\n[2/6] Engineering features...")

    training_data, feature_engineer = prepare_training_data(
        resumes_df,
        jobs_df,
        matches_df
    )

    print(
        f"✓ Created {len(training_data)} training samples "
        f"with {len(training_data.columns) - 4} features"
    )

    print(f"Features: {list(training_data.columns[:-4])}")

    # ------------------------------------------------------------------
    # CLEAN DATA
    # ------------------------------------------------------------------

    print("\nCleaning training data...")

    # Remove rows where target is NaN
    training_data = training_data.dropna(subset=["is_match"])

    # Replace NaN in features
    training_data = training_data.fillna(0)

    print("\nMissing values after cleaning:")
    print(training_data.isnull().sum())

    # ------------------------------------------------------------------
    # STEP 3 - Train Test Split
    # ------------------------------------------------------------------

    print("\n[3/6] Splitting data...")

    feature_cols = [
        col
        for col in training_data.columns
        if col not in [
            "candidate_id",
            "job_id",
            "is_match",
            "match_score"
        ]
    ]

    X = training_data[feature_cols]
    y = training_data["is_match"].astype(int)

    print("\nTarget Distribution:")
    print(y.value_counts(dropna=False))

    # Safety check
    if y.nunique() < 2:
        raise ValueError(
            "Target column contains only one class.\n"
            "Need both 0 and 1 labels for training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")

    print(
        f"✓ Positive Matches: "
        f"{y.sum()}/{len(y)} "
        f"({(y.sum()/len(y))*100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # STEP 4 - Train Models
    # ------------------------------------------------------------------

    print("\n[4/6] Training and evaluating multiple models...")
    print("-" * 70)

    results = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ------------------------------------------------------------------
    # STEP 5 - Select Best Model
    # ------------------------------------------------------------------

    print("\n[5/6] Selecting best model...")

    best_result = select_best_model(results)

    best_model = best_result["model"]

    print(f"✓ Best Model: {best_model.name}")
    print(f"  - Precision: {best_result['precision']:.4f}")
    print(f"  - Recall: {best_result['recall']:.4f}")
    print(f"  - F1 Score: {best_result['f1']:.4f}")

    # ------------------------------------------------------------------
    # BEST MODEL PREDICTIONS
    # ------------------------------------------------------------------

    y_pred_best = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):
        y_proba_best = best_model.predict_proba(X_test)[:, 1]
    else:
        y_proba_best = np.zeros(len(y_test))

    evaluator = MetricsEvaluator(
        best_model.name,
        y_test,
        y_pred_best,
        y_proba_best
    )

    evaluator.print_report()

    # ------------------------------------------------------------------
    # STEP 6 - Reports & Visualizations
    # ------------------------------------------------------------------

    print("\n[6/6] Generating reports and visualizations...")

    comparison_report = ComparisonReport()

    for result in results:

        model = result["model"]

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = np.zeros(len(y_test))

        eval_obj = MetricsEvaluator(
            model.name,
            y_test,
            y_pred,
            y_proba
        )

        comparison_report.add_evaluator(eval_obj)

    # Print comparison
    comparison_report.print_comparison()

    # Save comparison chart
    comparison_report.plot_comparison(
        "reports/model_comparison.png"
    )

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    plot_confusion_matrix(
        y_test,
        y_pred_best,
        best_model.name,
        "reports/confusion_matrix.png"
    )

    plot_roc_curve(
        y_test,
        y_proba_best,
        best_model.name,
        "reports/roc_curve.png"
    )

    plot_precision_recall_curve(
        y_test,
        y_proba_best,
        best_model.name,
        "reports/precision_recall_curve.png"
    )

    # ------------------------------------------------------------------
    # Save Model
    # ------------------------------------------------------------------

    save_model(best_model, "models/best_model.pkl")

    # Save feature names
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ------------------------------------------------------------------
    # Save Summary
    # ------------------------------------------------------------------

    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model.name,
        "metrics": evaluator.get_metrics_dict(),
        "all_models": [
            {
                "name": r["model"].name,
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1"]),
                "roc_auc": float(r.get("roc_auc", 0))
            }
            for r in results
        ],
        "feature_columns": feature_cols,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open("reports/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # SUCCESS MESSAGE
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nArtifacts Generated:")
    print("  ✓ models/best_model.pkl")
    print("  ✓ models/feature_names.json")
    print("  ✓ reports/training_summary.json")
    print("  ✓ reports/confusion_matrix.png")
    print("  ✓ reports/roc_curve.png")
    print("  ✓ reports/precision_recall_curve.png")
    print("  ✓ reports/model_comparison.png")

    print("\n" + "=" * 70)

    return {
        "best_model": best_model,
        "training_data": training_data,
        "feature_engineer": feature_engineer,
        "summary": summary
    }


if __name__ == "__main__":
    main()