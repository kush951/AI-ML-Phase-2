"""
Stage B deliverables:
  1. "Implement match vectors"  -> build_match_vector() / FEATURE_NAMES
  2. "Threshold validation"     -> validate_thresholds() / competency_band()

Everything here is deliberately simple and explainable (per the study guide's
"explainability" and "baseline first" principles) -- the feature vector is a
small set of human-interpretable numbers, not an opaque embedding.
"""
from typing import Dict, List, Tuple

from data import SKILLS

# Competency bands used for plain-English explanations and for validating
# that a threshold maps to a sane competency level.
COMPETENCY_BANDS = [
    (0, 20, "Novice"),
    (20, 40, "Beginner"),
    (40, 60, "Intermediate"),
    (60, 80, "Advanced"),
    (80, 101, "Expert"),
]

FEATURE_NAMES = [
    "coverage_ratio",   # fraction of required skills the student meets
    "avg_excess",       # avg (score - threshold) over MET skills (how comfortably they clear the bar)
    "deficit_sum",       # total shortfall across skills they DON'T meet (normalized 0-1)
    "weighted_score",    # coverage weighted by how demanding the job is
    "n_required",        # how many skills this job gates on (context feature)
]


def competency_band(level: int) -> str:
    for lo, hi, name in COMPETENCY_BANDS:
        if lo <= level < hi:
            return name
    return "Expert"


def validate_thresholds(thresholds: Dict[str, int]) -> List[str]:
    """Threshold validation (Stage C deliverable).

    Returns a list of human-readable error strings; empty list = valid.
    This is the gate that stops a company posting a nonsensical job
    (unknown skill, level outside L1-L100, etc).
    """
    errors = []
    if not thresholds:
        errors.append("A job must gate on at least one skill threshold.")
    for skill, level in thresholds.items():
        if skill not in SKILLS:
            errors.append(f"Unknown skill '{skill}'. Must be one of: {', '.join(SKILLS)}")
        if not isinstance(level, int):
            errors.append(f"Threshold for '{skill}' must be an integer level (L1-L100), got {level!r}")
        elif not (1 <= level <= 100):
            errors.append(f"Threshold for '{skill}' = L{level} is out of range. Must be between L1 and L100.")
    return errors


def build_match_vector(student_scores: Dict[str, int], thresholds: Dict[str, int]) -> Dict[str, float]:
    """Implement match vectors (Stage B deliverable).

    Turns a (student, job) pair into a small, explainable feature vector.
    """
    n_required = len(thresholds)
    if n_required == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    met = 0
    excess_total = 0.0
    deficit_total = 0.0
    difficulty_total = 0.0  # sum of thresholds, used to weight by how demanding the job is

    for skill, threshold in thresholds.items():
        score = student_scores.get(skill, 0)
        difficulty_total += threshold
        if score >= threshold:
            met += 1
            excess_total += (score - threshold)
        else:
            # normalize shortfall by threshold so a miss on a high bar
            # counts more than a miss on a low bar
            deficit_total += (threshold - score) / threshold

    coverage_ratio = met / n_required
    avg_excess = (excess_total / met) if met else 0.0
    deficit_sum = deficit_total / n_required
    avg_difficulty = difficulty_total / n_required
    weighted_score = coverage_ratio * (avg_difficulty / 100.0)

    return {
        "coverage_ratio": coverage_ratio,
        "avg_excess": avg_excess,
        "deficit_sum": deficit_sum,
        "weighted_score": weighted_score,
        "n_required": float(n_required),
    }


def baseline_score(student_scores: Dict[str, int], thresholds: Dict[str, int]) -> float:
    """Dumb baseline: plain overlap of required vs verified skills.

    Every other number in this project is judged relative to this.
    """
    if not thresholds:
        return 0.0
    met = sum(1 for s, t in thresholds.items() if student_scores.get(s, 0) >= t)
    return met / len(thresholds)


def explain_match(student_scores: Dict[str, int], thresholds: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """Plain-English 'why' for a match -- the explainability requirement.

    Returns (reasons_met, reasons_missing).
    """
    met_reasons, missing_reasons = [], []
    for skill, threshold in thresholds.items():
        score = student_scores.get(skill, 0)
        band = competency_band(score)
        if score >= threshold:
            met_reasons.append(
                f"{skill}: verified at L{score} ({band}), clears the L{threshold} bar by {score - threshold} levels."
            )
        else:
            missing_reasons.append(
                f"{skill}: verified at L{score} ({band}), falls short of the L{threshold} threshold by {threshold - score} levels."
            )
    return met_reasons, missing_reasons
