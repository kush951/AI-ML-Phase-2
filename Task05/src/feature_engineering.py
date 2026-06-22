"""
PlaceMux · Feature Engineering
Transforms raw student/job pairs into ML-ready feature vectors.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


# ─── Individual feature extractors ─────────────────────────────────────────────

def feat_skill_match_ratio(student: dict, job: dict) -> float:
    """Fraction of required skills that the student has."""
    req = set(job["required_skills"])
    if not req:
        return 0.0
    matched = req & set(student["skill_list"])
    return len(matched) / len(req)


def feat_qualified_skill_ratio(student: dict, job: dict) -> float:
    """Fraction of required skills met above the job's min threshold."""
    req = set(job["required_skills"])
    if not req:
        return 0.0
    threshold = job["min_skill_threshold"]
    qualified = sum(
        1 for s in req
        if student["verified_skills"].get(s, 0) >= threshold
    )
    return qualified / len(req)


def feat_avg_verified_score_on_required(student: dict, job: dict) -> float:
    """Average verified score on required skills (0 if not present)."""
    req = job["required_skills"]
    if not req:
        return 0.0
    scores = [student["verified_skills"].get(s, 0.0) for s in req]
    return np.mean(scores) / 100.0


def feat_preferred_skill_ratio(student: dict, job: dict) -> float:
    """Fraction of preferred skills the student has."""
    pref = set(job.get("preferred_skills", []))
    if not pref:
        return 0.0
    return len(pref & set(student["skill_list"])) / len(pref)


def feat_skill_coverage(student: dict, job: dict) -> float:
    """Coverage: how much of the student's skill set is relevant to the job."""
    all_job_skills = set(job["required_skills"] + job.get("preferred_skills", []))
    if not all_job_skills or not student["skill_list"]:
        return 0.0
    return len(all_job_skills & set(student["skill_list"])) / len(all_job_skills)


def feat_experience_match(student: dict, job: dict) -> float:
    """Normalised experience fit. Positive = over-qualified (capped), negative = under."""
    gap = student["experience_years"] - job["min_experience_years"]
    if gap >= 0:
        # Slight over-experience is fine; very over-qualified is less ideal
        return 1.0 - min(gap / 10.0, 0.3)
    else:
        return max(0.0, 1.0 + gap / 5.0)


def feat_seniority_gap(student: dict, job: dict) -> float:
    """Normalised seniority delta (student - job); clipped [-1, 1]."""
    gap = student["seniority_idx"] - job["seniority_required_idx"]
    return np.clip(gap / 4.0, -1.0, 1.0)


def feat_education_match(student: dict, job: dict) -> float:
    """1 if student meets or exceeds required education, else partial."""
    gap = student["education_level_idx"] - job["education_required_idx"]
    if gap >= 0:
        return 1.0
    return max(0.0, 1.0 + gap / 4.0)


def feat_location_fit(student: dict, job: dict) -> float:
    """1 = same city, 0.8 = remote ok, 0 = mismatch."""
    if student["city"] == job["city"]:
        return 1.0
    if job["remote_ok"] or student["open_to_remote"]:
        return 0.8
    return 0.0


def feat_salary_fit(student: dict, job: dict) -> float:
    """How well student expectation fits within budget."""
    budget = job["salary_budget_lpa"]
    expect = student["salary_expectation_lpa"]
    if expect <= budget:
        return 1.0
    ratio = expect / max(budget, 1.0)
    return max(0.0, 1.0 - (ratio - 1.0))


def feat_domain_match(student: dict, job: dict) -> float:
    """Exact domain match."""
    return 1.0 if student["domain"] == job["domain"] else 0.0


def feat_total_verified_skills(student: dict, _job: dict) -> float:
    """Number of verified skills student has (normalised by 20)."""
    return min(len(student["skill_list"]) / 20.0, 1.0)


# ─── Text similarity ────────────────────────────────────────────────────────────

class BioBioVectorizer:
    """Fits TF-IDF on all bios/descriptions; produces cosine sim at inference."""

    def __init__(self):
        self._vec = TfidfVectorizer(max_features=200, stop_words="english")
        self._fitted = False

    def fit(self, students: list[dict], jobs: list[dict]):
        texts = [s["bio"] for s in students] + [j["description"] for j in jobs]
        self._vec.fit(texts)
        self._fitted = True

    def transform(self, student: dict, job: dict) -> float:
        if not self._fitted:
            return 0.0
        sv = self._vec.transform([student["bio"]])
        jv = self._vec.transform([job["description"]])
        return float(cosine_similarity(sv, jv)[0, 0])


# ─── Full feature vector ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "skill_match_ratio",
    "qualified_skill_ratio",
    "avg_verified_score",
    "preferred_skill_ratio",
    "skill_coverage",
    "experience_match",
    "seniority_gap",
    "education_match",
    "location_fit",
    "salary_fit",
    "domain_match",
    "total_verified_skills",
    "text_similarity",
]


def extract_features(
    student: dict,
    job: dict,
    vectorizer: Optional[BioBioVectorizer] = None,
) -> np.ndarray:
    """Return a 1D numpy feature vector for a (student, job) pair."""
    return np.array([
        feat_skill_match_ratio(student, job),
        feat_qualified_skill_ratio(student, job),
        feat_avg_verified_score_on_required(student, job),
        feat_preferred_skill_ratio(student, job),
        feat_skill_coverage(student, job),
        feat_experience_match(student, job),
        feat_seniority_gap(student, job),
        feat_education_match(student, job),
        feat_location_fit(student, job),
        feat_salary_fit(student, job),
        feat_domain_match(student, job),
        feat_total_verified_skills(student, job),
        vectorizer.transform(student, job) if vectorizer else 0.0,
    ], dtype=float)


def build_feature_matrix(
    pairs: list[dict],
    students_by_id: dict,
    jobs_by_id: dict,
    vectorizer: Optional[BioBioVectorizer] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the full pairs list."""
    X_rows, y = [], []
    for p in pairs:
        s = students_by_id[p["student_id"]]
        j = jobs_by_id[p["job_id"]]
        X_rows.append(extract_features(s, j, vectorizer))
        y.append(p["label"])
    return np.vstack(X_rows), np.array(y, dtype=int)


def explain_features(student: dict, job: dict, vectorizer=None) -> dict:
    """Human-readable breakdown of every feature for a given pair."""
    feats = extract_features(student, job, vectorizer)
    return {
        name: round(float(val), 4)
        for name, val in zip(FEATURE_NAMES, feats)
    }
