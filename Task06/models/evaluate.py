"""
PlaceMux · Evaluation Module
Computes the complete metric suite on the held-out test set:
  Precision, Recall, F1, FPR, ROC-AUC, PR-AUC,
  NDCG@5, NDCG@10, MRR, MAP@10
— broken down by domain segment.
"""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

MODELS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    r = np.asarray(relevance[:k], dtype=float)
    if r.size == 0:
        return 0.0
    gains = r / np.log2(np.arange(2, r.size + 2))
    return float(gains.sum())


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k averaged over all queries (treat each row as a query)."""
    order  = np.argsort(y_score)[::-1]
    ideal  = np.sort(y_true)[::-1]
    dcg    = dcg_at_k(y_true[order], k)
    idcg   = dcg_at_k(ideal, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """MRR — reciprocal rank of first relevant document."""
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """AP@k for a single query."""
    order = np.argsort(y_score)[::-1][:k]
    hits, score = 0, 0.0
    for i, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            hits += 1
            score += hits / i
    return score / max(np.sum(y_true), 1)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   threshold: float = 0.5, df_full: pd.DataFrame = None) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    yt = y_test.values
    cm = confusion_matrix(yt, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)

    # Ranking metrics (treat whole test set as one ranked list)
    ndcg5  = ndcg_at_k(yt, y_prob, k=5)
    ndcg10 = ndcg_at_k(yt, y_prob, k=10)
    mrr    = mean_reciprocal_rank(yt, y_prob)
    map10  = average_precision_at_k(yt, y_prob, k=10)

    metrics = {
        "threshold":  threshold,
        "precision":  round(precision_score(yt, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(yt, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(yt, y_pred, zero_division=0), 4),
        "fpr":        round(float(fpr), 4),
        "roc_auc":    round(roc_auc_score(yt, y_prob), 4),
        "pr_auc":     round(average_precision_score(yt, y_prob), 4),
        "ndcg_at_5":  round(ndcg5, 4),
        "ndcg_at_10": round(ndcg10, 4),
        "mrr":        round(mrr, 4),
        "map_at_10":  round(map10, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "test_size":  len(yt),
    }

    # Segment breakdown (by domain if available)
    if df_full is not None and "domain" in df_full.columns:
        idx = X_test.index
        domains = df_full.loc[idx, "domain"] if idx.isin(df_full.index).all() else None
        if domains is not None:
            seg_metrics = {}
            for dom in domains.unique():
                mask = domains == dom
                if mask.sum() < 5:
                    continue
                yp  = y_pred[mask.values]
                yt_ = yt[mask.values]
                seg_metrics[dom] = {
                    "n":          int(mask.sum()),
                    "precision":  round(precision_score(yt_, yp, zero_division=0), 4),
                    "recall":     round(recall_score(yt_, yp, zero_division=0), 4),
                    "f1":         round(f1_score(yt_, yp, zero_division=0), 4),
                }
            metrics["segment_breakdown"] = seg_metrics

    return metrics


def print_report(metrics: dict, model_name: str = "Best Model"):
    print(f"\n{'='*55}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*55}")
    core = ["precision","recall","f1","fpr","roc_auc","pr_auc",
            "ndcg_at_5","ndcg_at_10","mrr","map_at_10"]
    for k in core:
        guardrail = ""
        if k == "fpr"        and metrics[k] <= 0.12: guardrail = "  ✅ ≤ 0.12"
        if k == "ndcg_at_10" and metrics[k] >= 0.75: guardrail = "  ✅ ≥ 0.75"
        print(f"  {k:<18}: {metrics[k]:.4f}{guardrail}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TN={metrics['tn']}")
    if "segment_breakdown" in metrics:
        print(f"\n  Segment Breakdown:")
        for dom, sm in metrics["segment_breakdown"].items():
            print(f"    {dom:<28}: F1={sm['f1']:.3f}  P={sm['precision']:.3f}  R={sm['recall']:.3f}  (n={sm['n']})")


if __name__ == "__main__":
    from utils.features import FEATURE_COLS
    from models.baseline import SkillOverlapBaseline

    print("Loading test data and best model...")
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    X_test     = pd.read_csv(MODELS_DIR / "X_test.csv")
    y_test     = pd.read_csv(MODELS_DIR / "y_test.csv").squeeze()

    # Load threshold if saved, else default 0.5
    thr_path = MODELS_DIR / "threshold.json"
    threshold = 0.5
    if thr_path.exists():
        threshold = json.load(open(thr_path))["optimal_threshold"]

    # Baseline
    baseline   = SkillOverlapBaseline()
    b_metrics  = baseline.evaluate(X_test[FEATURE_COLS], y_test)
    print_report(b_metrics, "Baseline (skill overlap)")

    # Best model
    metrics = evaluate_model(best_model, X_test[FEATURE_COLS], y_test, threshold=threshold)
    print_report(metrics, "Best Model (XGBoost)")

    # Save results
    results = {"baseline": b_metrics, "best_model": metrics, "threshold": threshold}
    out = MODELS_DIR / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
