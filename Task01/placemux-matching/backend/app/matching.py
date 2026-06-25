"""
The matching engine: baseline ranker, trained classifier, explainability,
and held-out evaluation. Per the study guide — "build a dumb baseline
before any clever models", "report precision/recall/false-positive rate
on real sample data", "every match needs a plain-English why".
"""
import json
import os
import random
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score,
)

from .features import compute_features, feature_vector, FEATURE_NAMES
from .seed_data import generate_students, generate_jobs, generate_companies, generate_labeled_pairs

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "match_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
EXPERIMENT_LOG_PATH = os.path.join(MODEL_DIR, "experiment_log.jsonl")

BASELINE_THRESHOLD = 0.6   # skill_overlap_ratio >= this -> baseline says "match"
MODEL_THRESHOLD = 0.5      # predicted probability >= this -> models says "match"


# ---------------------------------------------------------------- baseline
def baseline_score(features: Dict[str, float]) -> float:
    """Dumb baseline: rank purely by overlap of required vs verified skills."""
    return features["skill_overlap_ratio"]


# ---------------------------------------------------------------- training
def _build_dataset(pairs_per_job: int = 25, n_students: int = 300, n_jobs: int = 60):
    companies = generate_companies()
    students = generate_students(n_students)
    jobs = generate_jobs(list(range(len(companies))), n_jobs)
    labeled_pairs = generate_labeled_pairs(students, jobs, pairs_per_job)

    X, y = [], []
    for student, job, label in labeled_pairs:
        f = compute_features(student, job)
        X.append(feature_vector(f))
        y.append(label)
    return np.array(X), np.array(y), len(labeled_pairs)


def _best_f1_threshold(y_true, scores) -> float:
    """Pick the operating point that maximises F1 on this scorer's own ROC/PR curve,
    so baseline and models are each evaluated at their best realistic threshold
    rather than an arbitrary fixed cutoff."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
    return float(thresholds[int(np.argmax(f1s))])


def _metrics_at_threshold(y_true, scores, threshold) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan")
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "false_positive_rate": round(float(fpr), 4),
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(auc), 4),
        "n": int(len(y_true)),
    }


def train_and_evaluate(pairs_per_job: int = 25) -> Dict:
    X, y, n_pairs = _build_dataset(pairs_per_job=pairs_per_job)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # carve a validation slice out of train to pick operating thresholds,
    # so the test set stays purely held-out for the final reported numbers
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    val_model_probs = clf.predict_proba(X_val)[:, 1]
    val_baseline_scores = X_val[:, FEATURE_NAMES.index("skill_overlap_ratio")]
    model_threshold = _best_f1_threshold(y_val, val_model_probs)
    baseline_threshold = _best_f1_threshold(y_val, val_baseline_scores)

    # refit on the full training split (train+val) for the models we actually deploy
    clf.fit(X_train, y_train)
    model_probs = clf.predict_proba(X_test)[:, 1]
    baseline_scores_test = X_test[:, FEATURE_NAMES.index("skill_overlap_ratio")]

    model_metrics = _metrics_at_threshold(y_test, model_probs, model_threshold)
    baseline_metrics = _metrics_at_threshold(y_test, baseline_scores_test, baseline_threshold)
    model_metrics["threshold"] = round(model_threshold, 4)
    baseline_metrics["threshold"] = round(baseline_threshold, 4)

    lift = 0.0
    if baseline_metrics["precision"] > 0:
        lift = round(
            100 * (model_metrics["precision"] - baseline_metrics["precision"]) / baseline_metrics["precision"], 2
        )

    coefs = dict(zip(FEATURE_NAMES, clf.coef_[0].round(4).tolist()))

    result = {
        "model_version": "logreg_v1",
        "trained_on_pairs": int(n_pairs),
        "test_set_size": int(len(y_test)),
        "baseline": baseline_metrics,
        "models": model_metrics,
        "lift_over_baseline_pct": lift,
        "feature_coefficients": coefs,
    }

    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    with open(EXPERIMENT_LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def load_model() -> LogisticRegression:
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    return joblib.load(MODEL_PATH)


def load_metrics() -> Dict:
    if not os.path.exists(METRICS_PATH):
        return train_and_evaluate()
    with open(METRICS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------- inference
def score_pair(clf: LogisticRegression, student: Dict, job: Dict) -> Tuple[float, float, Dict]:
    f = compute_features(student, job)
    vec = np.array(feature_vector(f)).reshape(1, -1)
    model_prob = float(clf.predict_proba(vec)[0, 1])
    base = baseline_score(f)
    return model_prob, base, f


def explain(student: Dict, job: Dict, features: Dict[str, float]) -> Dict:
    required = set((job.get("required_skills") or {}).keys())
    have = student.get("verified_skills") or {}
    have_skills = set(have.keys())

    matched = sorted([s for s in required & have_skills if have[s] >= job["required_skills"][s]])
    missing = sorted([s for s in required if s not in have_skills or have[s] < job["required_skills"][s]])
    extra = sorted(list(have_skills - required))[:5]

    exp_delta = student.get("years_experience", 0) - job.get("min_experience", 0)
    experience_fit = (
        f"meets the {job.get('min_experience', 0)}yr requirement "
        f"({student.get('years_experience', 0)}yr on record)"
        if exp_delta >= 0 else
        f"short of the {job.get('min_experience', 0)}yr requirement by {abs(round(exp_delta,1))}yr"
    )
    location_fit = "remote/location compatible" if features["location_match"] >= 1.0 else "location mismatch, not remote-eligible"

    if matched and not missing:
        plain = f"Cleared every required skill ({', '.join(matched)}) and {experience_fit}."
    elif matched:
        plain = (
            f"Cleared {len(matched)}/{len(required)} required skills ({', '.join(matched)}); "
            f"missing or below bar on {', '.join(missing)}. Candidate {experience_fit}."
        )
    else:
        plain = f"No required skills cleared the verification bar yet; missing {', '.join(missing) or 'none on file'}."

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "experience_fit": experience_fit,
        "location_fit": location_fit,
        "plain_english": plain,
    }
