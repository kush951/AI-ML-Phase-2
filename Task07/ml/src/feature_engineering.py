"""
PlaceMux · AI/ML · Feature Engineering
Builds interaction features from student-job pairs for the matching models.
"""

import numpy as np
import pandas as pd
from typing import List

SKILLS = [
    "Python", "JavaScript", "React", "SQL", "Machine Learning",
    "Data Analysis", "Communication", "Leadership", "Project Management",
    "Node.js", "AWS", "Docker", "Java", "C++", "Product Design",
    "Excel", "Tableau", "Statistics", "NLP", "Deep Learning",
    "Git", "Agile", "Problem Solving", "FastAPI", "MongoDB"
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer matching-specific features from raw student+job columns.
    Returns a DataFrame of numeric features for models input.
    """
    features = pd.DataFrame(index=df.index)

    # ── Skill Coverage Features ──────────────────────────────────────────
    req_cols = [f"req_skill_{s.replace(' ', '_').lower()}" for s in SKILLS]
    stu_cols = [f"skill_{s.replace(' ', '_').lower()}" for s in SKILLS]

    req_matrix = df[req_cols].values
    stu_matrix = df[stu_cols].values

    # Hard coverage: student meets ≥75% of required threshold per skill
    hard_met = np.where(
        req_matrix > 0,
        (stu_matrix >= req_matrix * 0.75).astype(float),
        np.nan
    )
    features["skill_coverage_hard"] = np.nanmean(hard_met, axis=1)

    # Soft coverage: ratio of student score to required score
    soft_ratio = np.where(
        req_matrix > 0,
        np.minimum(stu_matrix / (req_matrix + 1e-9), 1.5),
        np.nan
    )
    features["skill_coverage_soft"] = np.nanmean(soft_ratio, axis=1)

    # Number of required skills student exceeds fully
    features["skills_exceeded"] = np.nansum(
        np.where(req_matrix > 0, (stu_matrix > req_matrix).astype(float), np.nan), axis=1
    )

    # Total deficit: sum of gaps for unmet required skills
    deficit = np.where(req_matrix > 0, np.maximum(req_matrix - stu_matrix, 0), 0)
    features["total_skill_deficit"] = deficit.sum(axis=1)

    # Number of required skills (job complexity)
    features["n_required_skills"] = (req_matrix > 0).sum(axis=1)

    # Number of student skills above 60
    features["n_strong_skills"] = (stu_matrix > 60).sum(axis=1)

    # Average required skill threshold
    features["avg_required_threshold"] = np.where(
        req_matrix > 0, req_matrix, np.nan
    ).mean(axis=1) if req_matrix.any() else 0
    features["avg_required_threshold"] = np.nanmean(
        np.where(req_matrix > 0, req_matrix, np.nan), axis=1
    )

    # ── Experience & Academic Features ───────────────────────────────────
    features["exp_gap"] = (df["years_experience"] - df["min_experience"]).clip(lower=-5)
    features["exp_meets"] = (df["years_experience"] >= df["min_experience"]).astype(int)
    features["cgpa_gap"] = (df["cgpa"] - df["min_cgpa"]).clip(lower=-5)
    features["cgpa_meets"] = (df["cgpa"] >= df["min_cgpa"]).astype(int)
    features["cgpa"] = df["cgpa"]
    features["years_experience"] = df["years_experience"]

    # ── Composite Score ───────────────────────────────────────────────────
    features["composite_score"] = (
        features["skill_coverage_hard"] * 0.5
        + features["exp_meets"] * 0.25
        + features["cgpa_meets"] * 0.25
    )

    return features.fillna(0)


def feature_names() -> List[str]:
    sample = pd.DataFrame(np.zeros((1, 1)))
    return [
        "skill_coverage_hard", "skill_coverage_soft", "skills_exceeded",
        "total_skill_deficit", "n_required_skills", "n_strong_skills",
        "avg_required_threshold", "exp_gap", "exp_meets",
        "cgpa_gap", "cgpa_meets", "cgpa", "years_experience", "composite_score"
    ]
