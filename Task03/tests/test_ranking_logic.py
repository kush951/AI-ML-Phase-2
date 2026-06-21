"""
Unit tests for the feature engineering and baseline ranking logic.
Run: pytest tests/test_ranking_logic.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import compute_features, compute_features_full, FEATURE_NAMES
from ml.baseline import baseline_score


def test_baseline_perfect_match():
    """Student clears every required skill -> baseline score is exactly 1.0."""
    student = {"Python": 90, "SQL": 80}
    reqs = [("Python", 1.0, 50), ("SQL", 1.0, 50)]
    assert baseline_score(student, reqs) == 1.0


def test_baseline_zero_match():
    """Student has none of the required skills -> baseline score is exactly 0.0."""
    student = {"Java": 90}
    reqs = [("Python", 1.0, 50), ("SQL", 1.0, 50)]
    assert baseline_score(student, reqs) == 0.0


def test_baseline_partial_match():
    """1 of 2 required skills cleared -> baseline is exactly 0.5, unweighted by design."""
    student = {"Python": 90, "SQL": 10}  # SQL score too low
    reqs = [("Python", 1.0, 50), ("SQL", 1.0, 50)]
    assert baseline_score(student, reqs) == 0.5


def test_baseline_no_requirements_returns_zero():
    """Edge case: a job with zero listed requirements must not crash or divide by zero."""
    assert baseline_score({"Python": 90}, []) == 0.0


def test_features_no_requirements_is_handled_gracefully():
    """A job with no skill requirements should return a zero vector and an explanatory
    reason, not raise an exception (this is the kind of missing-data edge case the
    study guide calls out as a common failure mode)."""
    result = compute_features({}, 0.0, [])
    assert result.vector.shape == (len(FEATURE_NAMES),)
    assert all(v == 0 for v in result.vector)
    assert "no listed skill requirements" in result.reasons[0]


def test_features_weighted_coverage_respects_weights():
    """A heavily-weighted skill the student misses should pull weighted_skill_coverage
    down more than a lightly-weighted one."""
    reqs_heavy_missing = [("Python", 3.0, 50), ("SQL", 1.0, 50)]
    student = {"SQL": 90}  # misses the heavily-weighted Python
    r1 = compute_features(student, 1.0, reqs_heavy_missing)

    reqs_light_missing = [("Python", 1.0, 50), ("SQL", 3.0, 50)]
    r2 = compute_features(student, 1.0, reqs_light_missing)  # same student, misses lightly-weighted Python

    coverage_idx = FEATURE_NAMES.index("weighted_skill_coverage")
    assert r2.vector[coverage_idx] > r1.vector[coverage_idx]


def test_features_full_experience_fit_meets_requirement():
    reqs = [("Python", 1.0, 50)]
    result = compute_features_full({"Python": 80}, student_experience=3.0, job_requirements=reqs,
                                    min_experience=2.0)
    exp_idx = FEATURE_NAMES.index("experience_fit")
    assert result.vector[exp_idx] == 1.0
    assert any("Meets the" in r for r in result.reasons)


def test_features_full_experience_fit_decays_below_requirement():
    """A student short on experience should get a partial (not zero, not full) score,
    and an explanation that states the gap."""
    reqs = [("Python", 1.0, 50)]
    result = compute_features_full({"Python": 80}, student_experience=0.0, job_requirements=reqs,
                                    min_experience=3.0)
    exp_idx = FEATURE_NAMES.index("experience_fit")
    assert 0.0 <= result.vector[exp_idx] < 1.0
    assert any("Short of" in r for r in result.reasons)


def test_features_reasons_mention_both_met_and_missing_skills():
    """Explainability requirement: when a student partially matches, the explanation
    must surface both what they cleared and what they're missing — not just one side."""
    reqs = [("Python", 1.0, 50), ("SQL", 1.0, 50)]
    student = {"Python": 90, "SQL": 10}
    result = compute_features(student, 1.0, reqs)
    joined = " ".join(result.reasons)
    assert "Meets requirement" in joined
    assert "Below requirement" in joined


def test_features_empty_student_profile_does_not_crash():
    """A brand-new student with zero verified skills should still produce a valid
    (all-low) feature vector rather than raising."""
    reqs = [("Python", 1.0, 50), ("SQL", 1.0, 50)]
    result = compute_features({}, 0.0, reqs)
    assert result.vector.shape == (len(FEATURE_NAMES),)
    coverage_idx = FEATURE_NAMES.index("weighted_skill_coverage")
    assert result.vector[coverage_idx] == 0.0
