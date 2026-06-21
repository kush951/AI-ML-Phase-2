"""
PlaceMux - Feature engineering
================================
Single source of truth for turning a (student, job) pair into a feature vector.
Used identically by training, evaluation, and live inference so there is no
train/serve skew. Every feature here is also given a plain-English label so
the API can produce an explanation, not just a number.
"""
from dataclasses import dataclass
import numpy as np


FEATURE_NAMES = [
    "weighted_skill_coverage",   # fraction of required (weight-adjusted) skills the student clears
    "avg_required_skill_score",  # how strong the student's verified scores are on required skills, 0-1
    "experience_fit",            # 1.0 if student meets/exceeds min experience, decays otherwise
    "n_overlapping_skills",      # raw count of overlapping skills (normalised)
    "skill_count_ratio",         # student's total verified skills / job's required skill count
]


@dataclass
class FeatureResult:
    vector: np.ndarray
    reasons: list  # plain-English explanation strings, ordered by contribution


def compute_features(student_skills: dict, student_experience: float,
                      job_requirements: list) -> FeatureResult:
    """
    student_skills: {skill_name: verified_score (0-100)}
    student_experience: years (float)
    job_requirements: list of (skill_name, weight, min_required_score)
    """
    if not job_requirements:
        vec = np.zeros(len(FEATURE_NAMES))
        return FeatureResult(vec, ["This job has no listed skill requirements yet."])

    total_weight = sum(w for _, w, _ in job_requirements)
    covered_weight = 0.0
    overlap_count = 0
    score_sum = 0.0
    met_skills, missing_skills = [], []

    for skill, weight, min_req in job_requirements:
        score = student_skills.get(skill, 0.0)
        score_sum += score / 100.0
        if score >= min_req:
            covered_weight += weight
            overlap_count += 1
            met_skills.append((skill, score, min_req))
        else:
            missing_skills.append((skill, score, min_req))

    weighted_skill_coverage = covered_weight / total_weight if total_weight else 0.0
    avg_required_skill_score = score_sum / len(job_requirements)
    n_overlapping_skills = overlap_count / len(job_requirements)
    skill_count_ratio = min(len(student_skills) / max(len(job_requirements), 1), 3.0) / 3.0

    j_min_exp = 0.0
    for_exp = job_requirements  # experience handled separately by caller; default neutral
    experience_fit = 1.0  # overwritten by caller if min_experience supplied (see compute_features_full)

    vector = np.array([
        weighted_skill_coverage,
        avg_required_skill_score,
        experience_fit,
        n_overlapping_skills,
        skill_count_ratio,
    ])

    reasons = []
    if met_skills:
        top = sorted(met_skills, key=lambda x: -x[1])[:3]
        reasons.append(
            "Meets requirement on: " + ", ".join(f"{s} ({sc:.0f}/100, needs {m})" for s, sc, m in top)
        )
    if missing_skills:
        worst = sorted(missing_skills, key=lambda x: x[1])[:2]
        reasons.append(
            "Below requirement on: " + ", ".join(f"{s} ({sc:.0f}/100, needs {m})" for s, sc, m in worst)
        )
    reasons.append(f"Covers {weighted_skill_coverage*100:.0f}% of weighted required skills.")

    return FeatureResult(vector, reasons)


def compute_features_full(student_skills: dict, student_experience: float,
                           job_requirements: list, min_experience: float) -> FeatureResult:
    """Full version including the experience-fit feature and its explanation."""
    result = compute_features(student_skills, student_experience, job_requirements)
    if min_experience is not None:
        if student_experience >= min_experience:
            experience_fit = 1.0
            exp_reason = f"Meets the {min_experience:.1f}yr experience requirement ({student_experience:.1f}yr)."
        else:
            gap = min_experience - student_experience
            experience_fit = max(0.0, 1 - gap / 3)
            exp_reason = (f"Short of the {min_experience:.1f}yr experience requirement by "
                          f"{gap:.1f}yr (has {student_experience:.1f}yr).")
        result.vector[2] = experience_fit
        result.reasons.append(exp_reason)
    return result
