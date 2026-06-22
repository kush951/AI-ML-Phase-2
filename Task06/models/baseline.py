"""
PlaceMux · Dumb Baseline
Rank by skill-overlap score only — no ML, no features beyond F1.
Every later model must beat this on all metrics.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.features import FEATURE_COLS


class SkillOverlapBaseline:
    """
    Threshold-based classifier using only skill_overlap (F1).
    Threshold tuned on validation set.
    """

    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = X["skill_overlap"].values
        probs = np.column_stack([1 - scores, scores])
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["skill_overlap"].values >= self.threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]

        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        tn = ((y_pred == 0) & (y == 0)).sum()
        fpr = fp / max(fp + tn, 1)

        return {
            "model":     "Baseline (skill overlap only)",
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
            "fpr":       round(fpr, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }


if __name__ == "__main__":
    from data.generate_data import generate_dataset
    from sklearn.model_selection import train_test_split

    df = generate_dataset(n_candidates=600, n_jobs=200, pairs_per_candidate=4)
    X = df[FEATURE_COLS]
    y = df["label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)
    baseline = SkillOverlapBaseline()
    results = baseline.evaluate(X_test, y_test)
    print("\nBaseline results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
