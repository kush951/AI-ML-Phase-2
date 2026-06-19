"""
The student<->job feature space.

Design principle from the study guide: a feature is a measurable signal
(verified skill score, JD requirement, years of exposure). Garbage or
leaky features beat any fancy model, so this module is deliberately the
single source of truth for every feature used downstream — both by the
baseline and by the trained model. Nothing in matching.py invents a
feature that isn't defined here.

All features are computed only from information that is legitimately
available at scoring time (no leakage of the outcome label).
"""
from typing import Dict, List

FEATURE_NAMES = [
    "skill_overlap_ratio",
    "skill_coverage_weighted",
    "avg_skill_gap",
    "missing_skills_ratio",
    "extra_skills_count",
    "experience_delta",
    "experience_met",
    "location_match",
    "education_score",
]


def compute_features(student: Dict, job: Dict) -> Dict[str, float]:
    """
    student: {years_experience, education_level, verified_skills: {skill: score}, location, remote_ok}
    job:     {min_experience, role_level, required_skills: {skill: min_score}, location, remote_ok}
    Returns a flat dict of numeric features — the row a model is trained/scored on.
    """
    required: Dict[str, float] = job.get("required_skills", {}) or {}
    have: Dict[str, float] = student.get("verified_skills", {}) or {}

    required_skills = set(required.keys())
    have_skills = set(have.keys())

    matched = required_skills & have_skills
    # a skill only "counts" if the student's verified score clears the job's bar
    cleared = {s for s in matched if have.get(s, 0) >= required[s]}

    n_required = max(len(required_skills), 1)

    skill_overlap_ratio = len(cleared) / n_required

    # weighted coverage: how far above/below the bar, capped so one skill can't dominate
    coverage_terms = []
    gap_terms = []
    for s in required_skills:
        req_score = required[s]
        got_score = have.get(s, 0.0)
        coverage_terms.append(min(got_score / max(req_score, 1e-6), 1.2))
        if s in matched:
            gap_terms.append(got_score - req_score)
    skill_coverage_weighted = sum(coverage_terms) / n_required if coverage_terms else 0.0
    avg_skill_gap = (sum(gap_terms) / len(gap_terms)) if gap_terms else -50.0  # no overlap -> big penalty
    avg_skill_gap = max(min(avg_skill_gap, 100.0), -100.0) / 100.0  # normalize to [-1, 1]

    missing = required_skills - cleared
    missing_skills_ratio = len(missing) / n_required

    extra_skills_count = len(have_skills - required_skills)

    experience_delta = student.get("years_experience", 0.0) - job.get("min_experience", 0.0)
    experience_delta_clipped = max(min(experience_delta, 10.0), -5.0) / 10.0  # normalize
    experience_met = 1.0 if experience_delta >= 0 else 0.0

    loc_a = (student.get("location") or "Remote").strip().lower()
    loc_b = (job.get("location") or "Remote").strip().lower()
    remote_ok = bool(student.get("remote_ok", True)) and bool(job.get("remote_ok", True))
    location_match = 1.0 if (loc_a == loc_b or remote_ok) else 0.0

    education_score = min(student.get("education_level", 2), 4) / 4.0

    return {
        "skill_overlap_ratio": skill_overlap_ratio,
        "skill_coverage_weighted": skill_coverage_weighted,
        "avg_skill_gap": avg_skill_gap,
        "missing_skills_ratio": missing_skills_ratio,
        "extra_skills_count": float(extra_skills_count),
        "experience_delta": experience_delta_clipped,
        "experience_met": experience_met,
        "location_match": location_match,
        "education_score": education_score,
    }


def feature_vector(features: Dict[str, float]) -> List[float]:
    return [features[name] for name in FEATURE_NAMES]
