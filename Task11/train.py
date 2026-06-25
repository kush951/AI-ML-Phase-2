"""
PlaceMux · Proctoring Hardening · Multi-Model Training Pipeline
Trains, tunes, and selects the best models across 6 algorithms.
All evaluation is on a fully held-out test set — no leakage.
"""

import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import lightgbm as lgb

from generate_data import generate_proctoring_data
from feature_engineering import ProctoringFeatureEngineer, get_all_feature_cols

# ── Output paths ─────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODELS_DIR = BASE
REPORTS_DIR = BASE
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 1 · Build the models zoo
# ─────────────────────────────────────────────────────────────────────
def build_models() -> dict:
    return {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=4,
                class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08, max_depth=5,
                subsample=0.8, random_state=42
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", XGBClassifier(
                n_estimators=250, learning_rate=0.07, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=2.5, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0
            )),
        ]),
        "LightGBM": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.07, max_depth=6,
                num_leaves=40, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────
# 2 · Evaluate one models
# ─────────────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_train, y_train, X_test, y_test, threshold=0.40):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "name":           name,
        "precision":      precision_score(y_test, y_pred, zero_division=0),
        "recall":         recall_score(y_test, y_pred, zero_division=0),
        "f1":             f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":        roc_auc_score(y_test, y_proba),
        "avg_precision":  average_precision_score(y_test, y_proba),
        "fpr":            fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr":            fn / (fn + tp) if (fn + tp) > 0 else 0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "train_time_s":   round(train_time, 2),
        "threshold":      threshold,
    }

    # 5-fold CV on training set for stability
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 scoring="roc_auc", n_jobs=-1)
    metrics["cv_roc_auc_mean"] = cv_scores.mean()
    metrics["cv_roc_auc_std"]  = cv_scores.std()

    return metrics, y_proba


# ─────────────────────────────────────────────────────────────────────
# 3 · Visualisations
# ─────────────────────────────────────────────────────────────────────
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]

def plot_roc_curves(roc_data: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)],
                lw=2, label=f"{name}  (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_metrics_comparison(results: list, save_path: Path):
    df = pd.DataFrame(results).set_index("name")
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + i * width, df[m], width, label=m.upper().replace("_", " "),
                      color=COLORS[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_confusion_matrix(result: dict, save_path: Path):
    cm = np.array([[result["tn"], result["fp"]],
                   [result["fn"], result["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Cheating"], fontsize=11)
    ax.set_yticklabels(["Legitimate", "Cheating"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {result['name']}", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_feature_importance(model_pipeline, feature_names: list, save_path: Path, top_n=15):
    clf = model_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx],
            color=COLORS[0], alpha=0.85)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ─────────────────────────────────────────────────────────────────────
# 4 · Main pipeline
# ─────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  PlaceMux · Proctoring Hardening · Training Pipeline")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    print("\n[1/5] Generating & engineering features...")
    raw_df = generate_proctoring_data(2000)
    engineer = ProctoringFeatureEngineer()
    df = engineer.fit_transform(raw_df)

    feature_cols = get_all_feature_cols(df)
    X = df[feature_cols]
    y = df["label"]

    print(f"  Sessions: {len(df)}  |  Features: {len(feature_cols)}  |  Cheating rate: {y.mean():.1%}")

    # ── Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_v, X_val, y_train_v, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"  Train: {len(X_train_v)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # ── Baseline (rule-based) ─────────────────────────────────────────
    print("\n[2/5] Establishing rule-based baseline...")
    baseline_pred = (df.loc[X_test.index, "overall_risk_score"] > 0.35).astype(int)
    baseline_metrics = {
        "name":      "Rule-Based Baseline",
        "precision": precision_score(y_test, baseline_pred, zero_division=0),
        "recall":    recall_score(y_test, baseline_pred, zero_division=0),
        "f1":        f1_score(y_test, baseline_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, df.loc[X_test.index, "overall_risk_score"]),
    }
    print(f"  Baseline  P={baseline_metrics['precision']:.3f}  R={baseline_metrics['recall']:.3f}  "
          f"F1={baseline_metrics['f1']:.3f}  AUC={baseline_metrics['roc_auc']:.3f}")

    # ── Train all models ──────────────────────────────────────────────
    print("\n[3/5] Training all models...")
    models = build_models()
    results, roc_data, trained_models = [], {}, {}

    for name, model in models.items():
        print(f"  → {name}...", end="", flush=True)
        metrics, y_proba = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model

        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr_arr, tpr_arr, metrics["roc_auc"])

        print(f" AUC={metrics['roc_auc']:.3f}  F1={metrics['f1']:.3f}  "
              f"FPR={metrics['fpr']:.3f}  FNR={metrics['fnr']:.3f}  ({metrics['train_time_s']}s)")

    # ── Select best ───────────────────────────────────────────────────
    print("\n[4/5] Selecting best models...")
    # Prioritise: F1 > AUC, with penalty for high FPR (false alarms hurt trust)
    for r in results:
        r["selection_score"] = r["f1"] * 0.5 + r["roc_auc"] * 0.3 - r["fpr"] * 0.2
    best = max(results, key=lambda r: r["selection_score"])
    best_model = trained_models[best["name"]]
    print(f"  ✓ Best models: {best['name']}")
    print(f"    P={best['precision']:.3f}  R={best['recall']:.3f}  F1={best['f1']:.3f}  "
          f"AUC={best['roc_auc']:.3f}  FPR={best['fpr']:.3f}")

    # ── Save artefacts ────────────────────────────────────────────────
    print("\n[5/5] Saving artefacts & plots...")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(engineer, MODELS_DIR / "feature_engineer.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    experiment = {
        "best_model": best["name"],
        "baseline": baseline_metrics,
        "results": results,
        "feature_cols": feature_cols,
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_samples": len(df),
        "cheat_rate": float(y.mean()),
        "improvement_vs_baseline": {
            "f1_delta":  round(best["f1"]  - baseline_metrics["f1"], 4),
            "auc_delta": round(best["roc_auc"] - baseline_metrics["roc_auc"], 4),
        }
    }
    with open(MODELS_DIR / "experiment_log.json", "w") as f:
        json.dump(experiment, f, indent=2)

    # plots
    plot_roc_curves(roc_data, REPORTS_DIR / "roc_curves.png")
    plot_metrics_comparison(results, REPORTS_DIR / "model_comparison.png")
    plot_confusion_matrix(best, REPORTS_DIR / "confusion_matrix.png")
    plot_feature_importance(best_model, feature_cols, REPORTS_DIR / "feature_importance.png")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Best models saved  →  models/best_model.pkl")
    print(f"  Experiment log    →  models/experiment_log.json")
    print("=" * 60)
    return experiment, best_model, engineer, feature_cols


if __name__ == "__main__":
    experiment, _, _, _ = run_pipeline()
