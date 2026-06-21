"""
PlaceMux - Train the ranking model
=====================================
Trains a calibrated logistic regression on the engineered feature vector
(see ml/features.py) to predict P(good_match) for a (student, job) pair.
That probability is the ranking score used for both:
  - Job ranking for students   (rank jobs for a fixed student)
  - Candidate ranking for companies (rank students for a fixed job)

Model: GradientBoostingClassifier, calibrated via Platt scaling (sigmoid).
We moved from a linear model to gradient-boosted trees because the engineered
feature set includes step-function and interaction signals (per-skill
threshold coverage, coverage x strength, a near-miss gap) that a linear model
can only combine additively -- trees capture the nonlinear interactions
between them directly. Explainability is preserved two ways: (1) the model
still scores a small, named, auditable feature set (not raw text or
embeddings), and (2) we report permutation/impurity-based feature importances
from the trained trees as the global explainability artifact, alongside the
existing per-prediction plain-English reasons (see ml/features.py) which are
independent of model type.

Every run appends to reports/experiment_log.csv so results are reproducible
and comparable across iterations (the "experiment log" the study guide asks for).
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.features import compute_features_full, FEATURE_NAMES
from ml.baseline import baseline_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "ml" / "artifacts"
MODELS.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)


def load_lookups():
    students = pd.read_csv(DATA / "students.csv")
    student_skills = pd.read_csv(DATA / "student_skills.csv")
    jobs = pd.read_csv(DATA / "jobs.csv")
    job_skills = pd.read_csv(DATA / "job_skills.csv")
    applications = pd.read_csv(DATA / "applications.csv")

    student_skill_lookup = {
        sid: dict(zip(g.skill_name, g.verified_score))
        for sid, g in student_skills.groupby("student_id")
    }
    job_skill_lookup = {
        jid: list(zip(g.skill_name, g.weight, g.min_required_score))
        for jid, g in job_skills.groupby("job_id")
    }
    student_meta = students.set_index("student_id").to_dict("index")
    job_meta = jobs.set_index("job_id").to_dict("index")
    return student_skill_lookup, job_skill_lookup, student_meta, job_meta, applications, jobs, students


def build_dataset(student_skill_lookup, job_skill_lookup, student_meta, job_meta, applications):
    X, y, baseline_scores, job_ids, student_ids, categories = [], [], [], [], [], []
    for row in applications.itertuples():
        sid, jid, label = row.student_id, row.job_id, row.good_match
        s_skills = student_skill_lookup.get(sid, {})
        s_exp = student_meta[sid]["experience_years"]
        reqs = job_skill_lookup.get(jid, [])
        min_exp = job_meta[jid]["min_experience_years"]

        feat = compute_features_full(s_skills, s_exp, reqs, min_exp)
        X.append(feat.vector)
        y.append(label)
        baseline_scores.append(baseline_score(s_skills, reqs))
        job_ids.append(jid)
        student_ids.append(sid)
        categories.append(job_meta[jid]["category"])

    return (np.array(X), np.array(y), np.array(baseline_scores),
            np.array(job_ids), np.array(student_ids), np.array(categories))


def main():
    t0 = time.time()
    (student_skill_lookup, job_skill_lookup, student_meta, job_meta,
     applications, jobs_df, students_df) = load_lookups()

    X, y, baseline_scores, job_ids, student_ids, categories = build_dataset(
        student_skill_lookup, job_skill_lookup, student_meta, job_meta, applications
    )

    # 60/20/20 train/val/test, stratified on label so each split keeps the real class balance
    (X_train, X_temp, y_train, y_temp, base_train, base_temp, cat_train, cat_temp,
     sid_train, sid_temp, jid_train, jid_temp) = train_test_split(
        X, y, baseline_scores, categories, student_ids, job_ids,
        test_size=0.4, random_state=7, stratify=y
    )
    (X_val, X_test, y_val, y_test, base_val, base_test, cat_val, cat_test,
     sid_val, sid_test, jid_val, jid_test) = train_test_split(
        X_temp, y_temp, base_temp, cat_temp, sid_temp, jid_temp,
        test_size=0.5, random_state=7, stratify=y_temp
    )

    # Small hyperparameter search, selected on the VALIDATION split (never the test split).
    # We optimise for PR-AUC specifically because precision-at-reasonable-recall is the
    # metric the baseline was already strong on -- this is what we are trying to beat.
    from sklearn.metrics import average_precision_score
    candidate_configs = [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.10, "subsample": 1.0},
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.05, "subsample": 1.0},
    ]
    best_config, best_val_pr_auc, best_clf = None, -1, None
    search_log = []
    for cfg in candidate_configs:
        trial = CalibratedClassifierCV(
            GradientBoostingClassifier(random_state=7, **cfg), method="sigmoid", cv=3
        )
        trial.fit(X_train, y_train)
        val_scores = trial.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_scores)
        search_log.append({**cfg, "val_pr_auc": round(val_pr_auc, 4)})
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc, best_config, best_clf = val_pr_auc, cfg, trial

    # refit the winning config with the full cv=5 calibration for the final artifact
    clf = CalibratedClassifierCV(
        GradientBoostingClassifier(random_state=7, **best_config), method="sigmoid", cv=5
    )
    clf.fit(X_train, y_train)

    # quick val check (used only for model selection, never for the reported test numbers)
    val_acc = clf.score(X_val, y_val)
    val_pr_auc_final = average_precision_score(y_val, clf.predict_proba(X_val)[:, 1])

    # --- Operating threshold selection (also on validation only) ---
    # The default 0.5 cutoff is arbitrary. We instead pick the threshold that gives the
    # model the BEST PRECISION it can reach while matching or beating the baseline's
    # recall at its own 0.5 cutoff -- a fair apples-to-apples operating point, not a
    # cherry-picked one. Falls back to 0.5 if no threshold clears the recall bar.
    from sklearn.metrics import precision_recall_curve, precision_score, recall_score
    baseline_val_pred = (base_val >= 0.5).astype(int)
    baseline_val_recall = recall_score(y_val, baseline_val_pred)
    baseline_val_precision = precision_score(y_val, baseline_val_pred, zero_division=0)

    model_val_scores = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, model_val_scores)
    eligible = [(p, t) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if r >= baseline_val_recall]
    if eligible:
        chosen_threshold = max(eligible, key=lambda pt: pt[0])[1]
    else:
        chosen_threshold = 0.5
    chosen_threshold = float(round(chosen_threshold, 4))

    with open(MODELS / "threshold.json", "w") as f:
        json.dump({
            "threshold": chosen_threshold,
            "selection_method": "max precision on validation subject to recall >= baseline recall on validation",
            "baseline_val_precision": round(float(baseline_val_precision), 4),
            "baseline_val_recall": round(float(baseline_val_recall), 4),
        }, f, indent=2)

    joblib.dump(clf, MODELS / "ranking_model.joblib")
    np.save(MODELS / "X_test.npy", X_test)
    np.save(MODELS / "y_test.npy", y_test)
    np.save(MODELS / "base_test.npy", base_test)
    np.save(MODELS / "cat_test.npy", cat_test)
    np.save(MODELS / "sid_test.npy", sid_test)
    np.save(MODELS / "jid_test.npy", jid_test)

    # learned feature importances -> explainability artifact (averaged across the 5 calibration folds)
    importances = np.mean([c.estimator.feature_importances_ for c in clf.calibrated_classifiers_], axis=0)
    coef_report = dict(zip(FEATURE_NAMES, importances.round(4).tolist()))

    run = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "positive_rate_train": float(y_train.mean()),
        "val_accuracy": round(float(val_acc), 4),
        "val_pr_auc": round(float(val_pr_auc_final), 4),
        "chosen_config": best_config,
        "chosen_threshold": chosen_threshold,
        "hyperparameter_search_log": search_log,
        "feature_importances": coef_report,
        "train_seconds": round(time.time() - t0, 2),
    }

    log_path = REPORTS / "experiment_log.csv"
    log_row = pd.DataFrame([{
        "timestamp": run["timestamp"], "n_train": run["n_train"], "n_val": run["n_val"],
        "n_test": run["n_test"], "positive_rate_train": run["positive_rate_train"],
        "val_accuracy": run["val_accuracy"], "val_pr_auc": run["val_pr_auc"],
        "chosen_config": str(best_config), "train_seconds": run["train_seconds"],
        **{f"importance_{k}": v for k, v in coef_report.items()},
    }])
    if log_path.exists():
        log_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_row.to_csv(log_path, index=False)

    with open(REPORTS / "last_train_run.json", "w") as f:
        json.dump(run, f, indent=2)

    print(json.dumps(run, indent=2))
    print(f"\nModel saved to {MODELS / 'ranking_model.joblib'}")
    print(f"Experiment log appended to {log_path}")


if __name__ == "__main__":
    main()