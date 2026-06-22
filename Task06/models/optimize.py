"""
PlaceMux · Threshold Optimization
Sweeps classification threshold, finds optimal via Youden's J,
enforces the FPR ≤ 0.12 guardrail, and records the result.
"""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve

MODELS_DIR = Path(__file__).parent
FPR_GUARDRAIL = 0.12   # Max tolerated false-positive rate
NDCG_FLOOR    = 0.75   # Min NDCG@10 (checked in evaluate.py)


def youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Optimal threshold by Youden's J statistic: max(TPR - FPR).
    If the optimal threshold violates the FPR guardrail, we tighten it.
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr_arr - fpr_arr
    best_idx  = int(np.argmax(j_scores))
    optimal   = float(thresholds[best_idx])
    optimal_fpr = float(fpr_arr[best_idx])

    print(f"\nYouden's J optimal threshold: {optimal:.3f}  "
          f"(TPR={tpr_arr[best_idx]:.3f}, FPR={optimal_fpr:.3f})")

    if optimal_fpr > FPR_GUARDRAIL:
        # Tighten threshold until FPR ≤ guardrail
        for fpr_val, tpr_val, thr in zip(fpr_arr, tpr_arr, thresholds):
            if fpr_val <= FPR_GUARDRAIL:
                final = float(thr)
                print(f"  FPR {optimal_fpr:.3f} > guardrail {FPR_GUARDRAIL} — "
                      f"tightened to threshold {final:.3f} (FPR={fpr_val:.3f})")
                return final
        return 0.5  # fallback
    return optimal


def pr_curve_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return dict of {threshold: {precision, recall, f1}} for the full sweep."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    results = {}
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 2 * p * r / max(p + r, 1e-9)
        results[round(float(t), 2)] = {
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1":        round(float(f1), 4),
        }
    return results


def optimize_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Full threshold optimization pipeline."""
    y_prob = model.predict_proba(X_val)[:, 1]
    optimal = youden_j_threshold(y_val.values, y_prob)

    sweep = pr_curve_sweep(y_val.values, y_prob)
    best_f1_thr = max(sweep.items(), key=lambda x: x[1]["f1"])
    print(f"  Best F1 at threshold {best_f1_thr[0]}: {best_f1_thr[1]}")

    return optimal


def apply_guardrails(metrics: dict) -> dict:
    """
    Check all guardrail metrics and annotate with pass/fail.
    Returns metrics dict with added 'guardrails' key.
    """
    guardrails = {
        "fpr_guardrail":   {"threshold": FPR_GUARDRAIL, "value": metrics.get("fpr", 1.0),
                            "pass": metrics.get("fpr", 1.0) <= FPR_GUARDRAIL},
        "ndcg_floor":      {"threshold": NDCG_FLOOR,    "value": metrics.get("ndcg_at_10", 0.0),
                            "pass": metrics.get("ndcg_at_10", 0.0) >= NDCG_FLOOR},
        "latency_guardrail":{"threshold": 20,            "value": metrics.get("latency_p95_ms", 99),
                             "pass": metrics.get("latency_p95_ms", 99) <= 20},
    }
    metrics["guardrails"] = guardrails
    all_pass = all(v["pass"] for v in guardrails.values())
    metrics["all_guardrails_pass"] = all_pass
    if all_pass:
        print("\n✅ All guardrails PASS")
    else:
        failed = [k for k, v in guardrails.items() if not v["pass"]]
        print(f"\n⚠️  Guardrail failures: {failed}")
    return metrics


if __name__ == "__main__":
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    X_val      = pd.read_csv(MODELS_DIR.parent / "data" / "sample_data.csv")
    from utils.features import FEATURE_COLS
    y_val = X_val["label"]
    X_val = X_val[FEATURE_COLS]

    # Use 20% as proxy val set
    from sklearn.model_selection import train_test_split
    _, X_v, _, y_v = train_test_split(X_val, y_val, test_size=0.2,
                                      stratify=y_val, random_state=99)
    optimal_thr = optimize_threshold(best_model, X_v, y_v)

    result = {"optimal_threshold": optimal_thr}
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOptimal threshold saved: {optimal_thr:.3f}")
