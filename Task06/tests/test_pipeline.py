"""
PlaceMux · Unit Tests
Tests for feature engineering, baseline, and API endpoint.
Run with: pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from utils.features import (
    skill_overlap, experience_match, verified_score_norm,
    loc_salary_match, FEATURE_COLS,
)
from models.baseline import SkillOverlapBaseline
from models.evaluate import ndcg_at_k, mean_reciprocal_rank, average_precision_at_k


# ---- Feature tests ----

class TestSkillOverlap:
    def test_perfect_overlap(self):
        assert skill_overlap(["Python", "SQL"], ["Python", "SQL"]) == 1.0

    def test_no_overlap(self):
        assert skill_overlap(["Python"], ["Java"]) == 0.0

    def test_partial_overlap(self):
        result = skill_overlap(["Python", "SQL", "React"], ["Python", "Java"])
        assert 0 < result < 1

    def test_empty_required(self):
        assert skill_overlap(["Python"], []) == 0.0

    def test_case_insensitive(self):
        assert skill_overlap(["python"], ["Python"]) == 1.0


class TestExperienceMatch:
    def test_exact_match(self):
        result = experience_match(3, 3)
        assert result == 0.5  # delta=0, midpoint

    def test_over_qualified(self):
        result = experience_match(8, 3)
        assert result > 0.5

    def test_under_qualified(self):
        result = experience_match(0, 5)
        assert result < 0.5

    def test_range(self):
        for yoe in range(0, 10):
            r = experience_match(yoe, 3)
            assert 0.0 <= r <= 1.0


class TestVerifiedScore:
    def test_perfect_score(self):
        assert verified_score_norm(100) == 1.0

    def test_zero_score(self):
        assert verified_score_norm(0) == 0.0

    def test_midrange(self):
        assert verified_score_norm(75) == pytest.approx(0.75, abs=0.01)

    def test_clipping(self):
        assert verified_score_norm(120) == 1.0
        assert verified_score_norm(-10) == 0.0


class TestLocSalaryMatch:
    def test_full_match(self):
        assert loc_salary_match(1, 1, 600000, 500000, 800000) == 1.0

    def test_no_match(self):
        assert loc_salary_match(3, 1, 1200000, 400000, 600000) == 0.0

    def test_half_match_location_only(self):
        result = loc_salary_match(1, 1, 1200000, 400000, 600000)
        assert result == 0.5


# ---- Baseline tests ----

class TestBaseline:
    def _make_data(self):
        np.random.seed(0)
        n = 100
        X = pd.DataFrame({
            "skill_overlap":        np.random.uniform(0, 1, n),
            "exp_match_norm":       np.random.uniform(0, 1, n),
            "semantic_similarity":  np.random.uniform(0, 1, n),
            "verified_score_norm":  np.random.uniform(0, 1, n),
            "loc_salary_match":     np.random.uniform(0, 1, n),
            "domain_match":         np.random.randint(0, 2, n),
        })
        y = pd.Series(np.random.randint(0, 2, n))
        return X, y

    def test_predict_shape(self):
        baseline = SkillOverlapBaseline()
        X, y = self._make_data()
        preds = baseline.predict(X)
        assert len(preds) == len(y)

    def test_predict_binary(self):
        baseline = SkillOverlapBaseline()
        X, y = self._make_data()
        preds = baseline.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self):
        baseline = SkillOverlapBaseline()
        X, y = self._make_data()
        proba = baseline.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_proba_sums_to_one(self):
        baseline = SkillOverlapBaseline()
        X, _ = self._make_data()
        proba = baseline.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_keys(self):
        baseline = SkillOverlapBaseline()
        X, y = self._make_data()
        m = baseline.evaluate(X, y)
        for key in ["precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]:
            assert key in m


# ---- Ranking metric tests ----

class TestRankingMetrics:
    def test_ndcg_perfect(self):
        y_true  = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert ndcg_at_k(y_true, y_score, k=4) == pytest.approx(1.0, abs=0.01)

    def test_ndcg_worst(self):
        y_true  = np.array([1, 0, 0, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.7])  # relevant at bottom
        assert ndcg_at_k(y_true, y_score, k=4) < 1.0

    def test_mrr_first_hit(self):
        y_true  = np.array([1, 0, 0])
        y_score = np.array([0.9, 0.5, 0.3])
        assert mean_reciprocal_rank(y_true, y_score) == pytest.approx(1.0, abs=0.01)

    def test_mrr_second_hit(self):
        y_true  = np.array([0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.1])
        assert mean_reciprocal_rank(y_true, y_score) == pytest.approx(0.5, abs=0.01)

    def test_map_at_k(self):
        y_true  = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.1])
        ap = average_precision_at_k(y_true, y_score, k=4)
        assert 0.0 <= ap <= 1.0


# ---- Feature column contract ----

class TestFeatureContract:
    def test_feature_cols_length(self):
        assert len(FEATURE_COLS) == 6

    def test_feature_cols_names(self):
        expected = {
            "skill_overlap", "exp_match_norm", "semantic_similarity",
            "verified_score_norm", "loc_salary_match", "domain_match",
        }
        assert set(FEATURE_COLS) == expected
