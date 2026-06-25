"""
PlaceMux — Explainability Task
Stage B.4 — Make it explainable & demoable.

Turns a models's numeric prediction into a plain-English "why" using SHAP
(TreeExplainer for tree models, LinearExplainer for linear models, KernelExplainer
fallback for anything else). This is the core deliverable: every match score
must ship with an explanation payload, never a bare number.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = str(ROOT / "models")
DATA_PATH = str(ROOT / "data" / "match_pairs.csv")

FRIENDLY_NAMES = {
    "domain_match": "Field match (student's domain vs job's domain)",
    "required_overlap_count": "Number of required skills the student has",
    "required_overlap_ratio": "Share of required skills the student has",
    "nice_overlap_count": "Number of nice-to-have skills the student has",
    "nice_overlap_ratio": "Share of nice-to-have skills the student has",
    "avg_required_skill_score": "Average verified score on required skills",
    "min_required_skill_score": "Weakest verified score among required skills",
    "meets_min_score_threshold": "Meets the job's minimum verified-score bar",
    "years_experience": "Student's years of experience",
    "min_years_required": "Job's minimum years required",
    "years_gap": "Experience surplus/deficit vs job requirement",
    "verified_skill_breadth": "Total number of verified skills on profile",
}


class MatchExplainer:
    def __init__(self):
        self.model = joblib.load(f"{MODELS_DIR}/best_model.joblib")
        self.scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
        with open(f"{MODELS_DIR}/model_meta.json") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        self.needs_scaling = self.meta["needs_scaling"]

        bg = pd.read_csv(DATA_PATH)[self.feature_cols].sample(
            n=200, random_state=42
        )
        bg_arr = self.scaler.transform(bg.values) if self.needs_scaling else bg.values

        model_type = self.meta["best_model_name"]
        if model_type in ("random_forest", "gradient_boosting"):
            self.explainer = shap.TreeExplainer(self.model)
            self.mode = "tree"
        elif model_type == "logistic_regression":
            self.explainer = shap.LinearExplainer(self.model, bg_arr)
            self.mode = "linear"
        else:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_arr)
            self.mode = "kernel"

    def _shap_values_for_positive_class(self, x_row_arr):
        sv = self.explainer.shap_values(x_row_arr)
        if isinstance(sv, list):
            return sv[1][0]
        arr = np.array(sv)
        if arr.ndim == 3:
            return arr[0, :, 1]
        return arr[0]

    def explain_batch(self, feature_rows: list, top_k: int = 3):
        """Vectorised explain for many rows at once (e.g. ranking a shortlist) —
        far cheaper than calling explain() once per row."""
        x_df = pd.DataFrame(feature_rows)[self.feature_cols]
        x_arr = self.scaler.transform(x_df.values) if self.needs_scaling else x_df.values
        probas = self.model.predict_proba(x_arr)[:, 1]

        sv = self.explainer.shap_values(x_arr)
        if isinstance(sv, list):
            sv_pos = sv[1]
        else:
            arr = np.array(sv)
            sv_pos = arr[:, :, 1] if arr.ndim == 3 else arr

        out = []
        for i in range(len(feature_rows)):
            proba = float(probas[i])
            pred = int(proba >= 0.5)
            contributions = [
                {
                    "feature": col,
                    "label": FRIENDLY_NAMES.get(col, col),
                    "value": float(x_df.iloc[i][col]),
                    "impact": round(float(v), 4),
                    "direction": "increases match likelihood" if v > 0 else "decreases match likelihood",
                }
                for col, v in zip(self.feature_cols, sv_pos[i])
            ]
            contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
            top = contributions[:top_k]
            out.append({
                "match_score": round(proba * 100, 1),
                "prediction": "GOOD MATCH" if pred == 1 else "WEAK MATCH",
                "top_factors": top,
                "explanation": self._to_sentence(pred, proba, top),
                "model_used": self.meta["best_model_name"],
            })
        return out

    def explain(self, feature_dict: dict, top_k: int = 4):
        x_row = pd.DataFrame([feature_dict])[self.feature_cols]
        x_arr = self.scaler.transform(x_row.values) if self.needs_scaling else x_row.values

        proba = float(self.model.predict_proba(x_arr)[0, 1])
        pred = int(proba >= 0.5)

        shap_vals = self._shap_values_for_positive_class(x_arr)
        contributions = [
            {
                "feature": col,
                "label": FRIENDLY_NAMES.get(col, col),
                "value": float(x_row.iloc[0][col]),
                "impact": round(float(v), 4),
                "direction": "increases match likelihood" if v > 0 else "decreases match likelihood",
            }
            for col, v in zip(self.feature_cols, shap_vals)
        ]
        contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
        top = contributions[:top_k]

        sentence = self._to_sentence(pred, proba, top)

        return {
            "match_score": round(proba * 100, 1),
            "prediction": "GOOD MATCH" if pred == 1 else "WEAK MATCH",
            "top_factors": top,
            "all_factors": contributions,
            "explanation": sentence,
            "model_used": self.meta["best_model_name"],
        }

    @staticmethod
    def _to_sentence(pred, proba, top):
        helping = [c for c in top if c["impact"] > 0]
        hurting = [c for c in top if c["impact"] <= 0]
        verdict = "a strong match" if proba >= 0.7 else ("a possible match" if proba >= 0.5 else "not a strong match")
        parts = [f"This candidate looks like {verdict} ({proba*100:.0f}/100)."]
        if helping:
            reasons = "; ".join(f"{c['label'].lower()} ({c['value']:g})" for c in helping[:3])
            parts.append(f"Main reasons in favour: {reasons}.")
        if hurting:
            reasons = "; ".join(f"{c['label'].lower()} ({c['value']:g})" for c in hurting[:2])
            parts.append(f"Held back by: {reasons}.")
        return " ".join(parts)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    explainer = MatchExplainer()
    sample = df.sample(3, random_state=7).to_dict(orient="records")
    for row in sample:
        result = explainer.explain(row)
        print(f"\n{row['student_id']} <-> {row['job_id']} (true label={row['is_good_match']})")
        print(json.dumps(result, indent=2))
