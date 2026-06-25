"""
PlaceMux · Training Pipeline
Run this to train all models, pick the best, and save artefacts.

Usage:
    python src/train.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    BioBioVectorizer, build_feature_matrix, FEATURE_NAMES
)
from src.models import (
    train_and_compare, select_best, get_feature_importances, evaluate,
    baseline_predict_proba
)

# ─── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    with open(ROOT / "data" / "dataset.json") as f:
        data = json.load(f)
    students_by_id = {s["id"]: s for s in data["students"]}
    jobs_by_id = {j["id"]: j for j in data["jobs"]}
    return data["pairs"], students_by_id, jobs_by_id, data["students"], data["jobs"]


def main():
    print("=" * 70)
    print("PlaceMux · Matching Validation · Training Pipeline")
    print("=" * 70)

    pairs, students_by_id, jobs_by_id, students, jobs = load_data()
    print(f"\nDataset: {len(pairs)} pairs  |  "
          f"Positive rate: {sum(p['label'] for p in pairs) / len(pairs):.2%}\n")

    # ── Fit TF-IDF vectorizer on all texts ──────────────────────────────────
    print("Fitting text vectorizer...")
    vec = BioBioVectorizer()
    vec.fit(students, jobs)

    # ── Build feature matrix ─────────────────────────────────────────────────
    print("Building feature matrix...")
    X, y = build_feature_matrix(pairs, students_by_id, jobs_by_id, vec)
    print(f"  X shape: {X.shape}  |  Features: {FEATURE_NAMES}\n")

    # ── Train / test split (stratified) ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(y_train)} samples  |  Test: {len(y_test)} samples\n")

    # ── Train & compare all models ───────────────────────────────────────────
    print("Training & evaluating all models:")
    print("-" * 70)
    results, trained_models = train_and_compare(
        X_train, y_train, X_test, y_test, FEATURE_NAMES
    )

    # ── Select best ─────────────────────────────────────────────────────────
    best_name = select_best(results)
    best_model = trained_models[best_name]
    best_metrics = results[best_name]
    baseline_metrics = results["Baseline_SkillOverlap"]

    print("\n" + "=" * 70)
    print(f"✅  BEST MODEL: {best_name}")
    print(f"    F1:       {best_metrics['f1']:.4f}  (baseline: {baseline_metrics['f1']:.4f}, "
          f"+{best_metrics['f1'] - baseline_metrics['f1']:.4f})")
    print(f"    AUC-ROC:  {best_metrics['roc_auc']:.4f}  (baseline: {baseline_metrics['roc_auc']:.4f})")
    print(f"    Precision:{best_metrics['precision']:.4f}")
    print(f"    Recall:   {best_metrics['recall']:.4f}")
    print(f"    FPR:      {best_metrics['fpr']:.4f}")
    print(f"    NDCG@5:   {best_metrics['ndcg_5']:.4f}")
    print(f"    MAP:      {best_metrics['map']:.4f}")

    # ── Feature importances ─────────────────────────────────────────────────
    importances = get_feature_importances(best_model, FEATURE_NAMES)
    if importances:
        print(f"\nTop feature importances for {best_name}:")
        for feat, imp in list(importances.items())[:6]:
            bar = "█" * int(imp * 40)
            print(f"  {feat:<30} {bar} {imp:.4f}")

    # ── Save artefacts ───────────────────────────────────────────────────────
    exp_dir = ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)

    # Experiment log
    log = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "all_results": results,
        "feature_importances": importances,
        "feature_names": FEATURE_NAMES,
        "dataset_stats": {
            "total_pairs": len(pairs),
            "positive_rate": round(y.mean(), 4),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
        }
    }
    with open(exp_dir / "experiment_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Best models + vectorizer
    artefacts = {
        "models": best_model,
        "vectorizer": vec,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances,
    }
    with open(exp_dir / "best_model.pkl", "wb") as f:
        pickle.dump(artefacts, f)

    print(f"\nArtefacts saved to experiments/")
    print("  • experiments/experiment_log.json")
    print("  • experiments/best_model.pkl")
    print("=" * 70)

    return log


if __name__ == "__main__":
    main()
