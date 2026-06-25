"""
PlaceMux · Multi-Model Training & Selection Pipeline
Trains 6 models, evaluates on held-out test set, selects best, saves artefacts.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
for d in (MODEL_DIR, REPORT_DIR):
    d.mkdir(exist_ok=True)

SKILLS = [
    "python", "sql", "machine_learning", "data_analysis", "deep_learning",
    "javascript", "react", "node_js", "java", "aws", "docker", "kubernetes",
    "communication", "teamwork", "problem_solving", "project_management",
    "excel", "tableau", "spark", "statistics",
]

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_features():
    df = pd.read_csv(DATA_DIR / "matching_pairs.csv")
    feature_cols = (
        ["skill_overlap_ratio", "skill_vector_distance", "student_experience", "student_gpa", "job_salary_lpa"]
        + [f"gap_{s}" for s in SKILLS]
    )
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols, df

# ── Models ────────────────────────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)),
        ]),
        "Extra Trees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
        ]),
    }

# ── Baseline ──────────────────────────────────────────────────────────────────
def baseline_predictions(df, threshold=0.55):
    """Dumb baseline: predict match if skill_overlap_ratio >= threshold."""
    return (df["skill_overlap_ratio"] >= threshold).astype(int).values

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "models": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

# ── Plot helpers ──────────────────────────────────────────────────────────────
PALETTE = ["#1B4F8A", "#E07B39", "#2CA87F", "#9B59B6", "#E74C3C", "#3498DB", "#7F8C8D"]

def plot_model_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PlaceMux · Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    metrics = ["f1", "roc_auc", "avg_precision"]
    titles  = ["F1 Score", "ROC-AUC", "Avg Precision (PR-AUC)"]
    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.barh(results_df["models"], results_df[metric], color=PALETTE[:len(results_df)])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.axvline(x=results_df[results_df["models"] == "Baseline"][metric].values[0],
                   color="gray", linestyle="--", linewidth=1.5, label="Baseline")
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → model_comparison.png saved")

def plot_roc_pr(models_dict, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PlaceMux · ROC & PR Curves", fontsize=13, fontweight="bold")
    for (name, model), color in zip(models_dict.items(), PALETTE):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax1.legend(fontsize=7)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → roc_pr_curves.png saved")

def plot_feature_importance(best_model, feature_cols):
    clf = best_model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_cols[i] for i in idx[::-1]], importances[idx[::-1]], color="#1B4F8A")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top-20 Feature Importances (Best Model)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → feature_importance.png saved")

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Match", "Match"],
                yticklabels=["No Match", "Match"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix · {model_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → confusion_matrix.png saved")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PlaceMux · Multi-Model Training Pipeline")
    print("=" * 60)

    X, y, feature_cols, df = load_features()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    df_test = df.iloc[idx_test].copy()
    print(f"\nDataset: {len(X)} samples  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Positive class balance: {y.mean():.1%}")

    # --- Baseline ---
    print("\n[Baseline]")
    y_base_pred = baseline_predictions(df_test)
    y_base_prob = df_test["skill_overlap_ratio"].values
    base_result = evaluate("Baseline", y_test, y_base_pred, y_base_prob)
    print(f"  F1={base_result['f1']}  AUC={base_result['roc_auc']}  FPR={base_result['false_positive_rate']}")

    # --- Train models ---
    models = build_models()
    results = [base_result]
    trained_models = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[Training Models]")
    for name, pipeline in models.items():
        t0 = time.time()
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        result = evaluate(name, y_test, y_pred, y_prob)
        result["cv_auc_mean"] = round(cv_auc.mean(), 4)
        result["cv_auc_std"]  = round(cv_auc.std(), 4)
        result["train_time_s"] = round(time.time() - t0, 2)
        results.append(result)
        trained_models[name] = pipeline
        print(f"  {name:25s}  F1={result['f1']:.4f}  AUC={result['roc_auc']:.4f}  "
              f"FPR={result['false_positive_rate']:.4f}  CV={cv_auc.mean():.4f}±{cv_auc.std():.4f}  ({result['train_time_s']}s)")

    # --- Select best ---
    results_df = pd.DataFrame(results)
    ml_results = results_df[results_df["models"] != "Baseline"].copy()
    best_row = ml_results.loc[ml_results["roc_auc"].idxmax()]
    best_name = best_row["models"]
    best_model = trained_models[best_name]
    print(f"\n✓ Best models: {best_name}  (AUC={best_row['roc_auc']:.4f}  F1={best_row['f1']:.4f})")

    # --- Plots ---
    print("\n[Generating Plots]")
    plot_model_comparison(results_df)
    plot_roc_pr(trained_models, X_test, y_test)
    plot_feature_importance(best_model, feature_cols)
    y_best_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_best_pred, best_name)

    # --- Save artefacts ---
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
    results_df.to_csv(REPORT_DIR / "model_results.csv", index=False)

    # Save experiment log
    experiment_log = {
        "best_model": best_name,
        "best_metrics": best_row.to_dict(),
        "baseline_metrics": base_result,
        "all_results": results_df.to_dict(orient="records"),
        "feature_count": len(feature_cols),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    with open(REPORT_DIR / "experiment_log.json", "w") as f:
        json.dump(experiment_log, f, indent=2, default=str)

    print(f"\nAll artefacts saved to {MODEL_DIR} and {REPORT_DIR}")
    print("\n" + classification_report(y_test, y_best_pred, target_names=["No Match", "Match"]))

    return results_df, best_model, best_name, feature_cols, X_test, y_test

if __name__ == "__main__":
    main()
