"""
PlaceMux · Feature Engineering (F1–F5)
All feature computation logic lives here to keep train/inference consistent.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


FEATURE_COLS = [
    "skill_overlap",        # F1 — Jaccard similarity of skill sets
    "exp_match_norm",       # F2 — Experience delta normalised [0,1]
    "semantic_similarity",  # F3 — TF-IDF cosine of profile vs JD text
    "verified_score_norm",  # F4 — Verified test score / 100
    "loc_salary_match",     # F5 — Location + salary band alignment
    "domain_match",         # Bonus binary: same domain?
]


def skill_overlap(candidate_skills: List[str], required_skills: List[str]) -> float:
    """Jaccard similarity between candidate skills and JD requirements."""
    c = set(s.lower() for s in candidate_skills)
    r = set(s.lower() for s in required_skills)
    if not r:
        return 0.0
    return len(c & r) / len(c | r)


def experience_match(candidate_yoe: int, min_yoe_required: int,
                     clip: float = 5.0) -> float:
    """
    Normalised experience delta.
    Returns value in [0, 1]; 0.5 = exact match, >0.5 = over-qualified, <0.5 = under.
    """
    delta = np.clip(candidate_yoe - min_yoe_required, -clip, clip)
    return round((delta + clip) / (2 * clip), 4)


def semantic_similarity(profile_text: str, jd_text: str) -> float:
    """TF-IDF cosine similarity between candidate profile and job description."""
    try:
        vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        tfidf = vec.fit_transform([profile_text, jd_text])
        return round(float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]), 4)
    except Exception:
        return 0.0


def verified_score_norm(avg_score: float) -> float:
    """Normalise verified skill-test score to [0, 1]."""
    return round(np.clip(avg_score, 0, 100) / 100.0, 4)


def loc_salary_match(candidate_city_tier: int, job_city_tier_accepted: int,
                     candidate_salary_exp: float, job_salary_min: float,
                     job_salary_max: float) -> float:
    """Binary location match + salary band overlap combined into [0, 1]."""
    loc = int(candidate_city_tier <= job_city_tier_accepted)
    sal = int(job_salary_min <= candidate_salary_exp <= job_salary_max * 1.10)
    return round((loc + sal) / 2.0, 4)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all FEATURE_COLS exist in df.
    If the df already has them (pre-computed), just return the subset.
    """
    return df[FEATURE_COLS].copy()


def feature_names() -> List[str]:
    return FEATURE_COLS.copy()
