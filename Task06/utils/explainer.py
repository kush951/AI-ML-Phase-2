"""
PlaceMux · SHAP Explainability Wrapper
Provides per-prediction feature attribution and global importance.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from utils.features import FEATURE_COLS


FEATURE_LABELS = {
    "skill_overlap":        "Skill overlap (Jaccard)",
    "exp_match_norm":       "Experience match",
    "semantic_similarity":  "Semantic profile similarity",
    "verified_score_norm":  "Verified test score",
    "loc_salary_match":     "Location & salary fit",
    "domain_match":         "Same domain",
}


class MatchExplainer:
    def __init__(self, model, X_background: pd.DataFrame):
        self.model = model
        self.X_background = X_background[FEATURE_COLS]
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            if SHAP_AVAILABLE:
                try:
                    self._explainer = shap.TreeExplainer(self.model)
                except Exception:
                    self._explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(self.X_background, 50)
                    )
        return self._explainer

    def explain_single(self, row: pd.DataFrame) -> Dict[str, Any]:
        """
        Return per-feature SHAP values for a single prediction row.
        Falls back to feature*weight heuristic if SHAP unavailable.
        """
        x = row[FEATURE_COLS]
        score = float(self.model.predict_proba(x)[:, 1][0])

        if SHAP_AVAILABLE:
            try:
                explainer = self._get_explainer()
                sv = explainer.shap_values(x)
                # For binary classifiers, sv may be [neg_class, pos_class]
                if isinstance(sv, list):
                    sv = sv[1]
                shap_vals = sv[0] if sv.ndim == 2 else sv
                attribution = {
                    FEATURE_LABELS[f]: round(float(v), 4)
                    for f, v in zip(FEATURE_COLS, shap_vals)
                }
            except Exception:
                attribution = _fallback_attribution(x)
        else:
            attribution = _fallback_attribution(x)

        sorted_attr = dict(
            sorted(attribution.items(), key=lambda item: abs(item[1]), reverse=True)
        )

        verdict = (
            "Strong match — recommend application" if score >= 0.75
            else "Partial match — review manually" if score >= 0.50
            else "Weak match — likely not suitable"
        )

        return {
            "match_score": round(score, 4),
            "verdict": verdict,
            "shap_reasons": sorted_attr,
            "plain_english": _plain_english(sorted_attr, score),
        }

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Mean absolute SHAP value per feature across the dataset.
        Returns a DataFrame sorted by importance descending.
        """
        x = X[FEATURE_COLS]
        if SHAP_AVAILABLE:
            try:
                explainer = self._get_explainer()
                sv = explainer.shap_values(x)
                if isinstance(sv, list):
                    sv = sv[1]
                mean_abs = np.abs(sv).mean(axis=0)
            except Exception:
                mean_abs = x.mean().values
        else:
            mean_abs = x.mean().values

        return pd.DataFrame({
            "feature": [FEATURE_LABELS[f] for f in FEATURE_COLS],
            "importance": np.round(mean_abs, 4),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


def _fallback_attribution(x: pd.DataFrame) -> Dict[str, float]:
    """Simple feature-value heuristic when SHAP is unavailable."""
    weights = [0.38, 0.10, 0.22, 0.30, 0.08, 0.05]
    return {
        FEATURE_LABELS[f]: round(float(x[f].values[0]) * w, 4)
        for f, w in zip(FEATURE_COLS, weights)
    }


def _plain_english(attribution: Dict[str, float], score: float) -> str:
    """Generate a human-readable explanation of the top drivers."""
    top = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    parts = [f"{name} (contribution: {val:+.3f})" for name, val in top]
    return (
        f"Match score {score:.2f} driven mainly by: {'; '.join(parts)}. "
        f"{'All key criteria met.' if score >= 0.75 else 'Some gaps exist — manual review advised.'}"
    )
