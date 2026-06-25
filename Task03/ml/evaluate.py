"""
PlaceMux - Comprehensive models evaluation
============================================
Directly answers the review feedback "expand evaluation metrics to include
more comprehensive performance analysis". Produces, on the HELD-OUT TEST SET
ONLY (never seen during training or models selection):

  Classification metrics @ threshold 0.5:
    - precision, recall, F1, false-positive rate, accuracy, confusion matrix
  Threshold-free metrics:
    - ROC-AUC, PR-AUC (average precision)
  Ranking-quality metrics (what actually matters for a ranked list, not just
  a yes/no classifier):
    - NDCG@5, NDCG@10, MAP@10, Precision@5  (averaged per-student / per-job query)
  Model vs baseline:
    - same metrics for the dumb overlap baseline, side by side
  Segment breakdown:
    - all of the above broken out by job category (catches "good overall,
      broken on one segment" failures a single accuracy number hides)
  Calibration:
    - reliability table (predicted prob bucket vs actual positive rate)

Outputs:
  reports/metrics_report.json   - machine-readable, served by the API at /metrics
  reports/metrics_report.md     - human-readable summary
  reports/pr_curve.png, reports/roc_curve.png, reports/calibration.png
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "ml" / "artifacts"
REPORTS = ROOT / "reports"


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def classification_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(false_positive_rate(y_true, y_pred), 4),
        "accuracy": round((tp + tn) / max(len(y_true), 1), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(average_precision_score(y_true, scores), 4) if len(set(y_true)) > 1 else None,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def ranking_metrics(y_true, scores, group_ids, k_list=(5, 10)):
    """Per-query (per-student or per-job) ranking metrics: NDCG@k, Precision@k, MAP@k.
    A 'query' here = all the candidate rows sharing the same group_id (e.g. all jobs
    shown to one student). This measures ranking quality, not just point classification."""
    df = pd.DataFrame({"y": y_true, "score": scores, "group": group_ids})
    results = {f"ndcg@{k}": [] for k in k_list}
    results.update({f"precision@{k}": [] for k in k_list})
    results["map@10"] = []

    for _, g in df.groupby("group"):
        g_sorted = g.sort_values("score", ascending=False).reset_index(drop=True)
        rels = g_sorted["y"].values
        if rels.sum() == 0:
            continue  # query has no positive label, ranking metrics undefined -> skip (documented)
        for k in k_list:
            top_k = rels[:k]
            ideal = np.sort(rels)[::-1][:k]
            dcg = np.sum(top_k / np.log2(np.arange(2, len(top_k) + 2)))
            idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
            results[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
            results[f"precision@{k}"].append(top_k.sum() / k)
        # MAP@10
        top10 = rels[:10]
        hits, precisions = 0, []
        for i, r in enumerate(top10, start=1):
            if r == 1:
                hits += 1
                precisions.append(hits / i)
        results["map@10"].append(np.mean(precisions) if precisions else 0.0)

    n_queries = len(results["map@10"])
    return {k: round(float(np.mean(v)), 4) if v else None for k, v in results.items()}, n_queries


def calibration_table(y_true, scores, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    table = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if mask.sum() == 0:
            continue
        table.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": round(float(scores[mask].mean()), 3),
            "actual_positive_rate": round(float(y_true[mask].mean()), 3),
        })
    return table


def main():
    clf = joblib.load(MODELS / "ranking_model.joblib")
    X_test = np.load(MODELS / "X_test.npy")
    y_test = np.load(MODELS / "y_test.npy")
    base_test = np.load(MODELS / "base_test.npy")
    cat_test = np.load(MODELS / "cat_test.npy", allow_pickle=True)
    sid_test = np.load(MODELS / "sid_test.npy")
    jid_test = np.load(MODELS / "jid_test.npy")

    threshold_path = MODELS / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_info = json.load(f)
        model_threshold = threshold_info["threshold"]
    else:
        threshold_info = {"threshold": 0.5, "selection_method": "default (no threshold.json found)"}
        model_threshold = 0.5

    model_scores = clf.predict_proba(X_test)[:, 1]

    # ---- overall: models vs baseline ----
    model_overall = classification_metrics(y_test, model_scores, threshold=model_threshold)
    baseline_overall = classification_metrics(y_test, base_test, threshold=0.5)
    model_rank_cat, n_model_q = ranking_metrics(y_test, model_scores, group_ids=cat_test)
    base_rank_cat, n_base_q = ranking_metrics(y_test, base_test, group_ids=cat_test)

    # The two real deliverables: ranking jobs FOR a student, and ranking candidates FOR a job.
    model_rank_by_student, n_student_q = ranking_metrics(y_test, model_scores, group_ids=sid_test)
    base_rank_by_student, _ = ranking_metrics(y_test, base_test, group_ids=sid_test)
    model_rank_by_job, n_job_q = ranking_metrics(y_test, model_scores, group_ids=jid_test)
    base_rank_by_job, _ = ranking_metrics(y_test, base_test, group_ids=jid_test)

    # ---- segment breakdown by job category ----
    segments = {}
    for cat in sorted(set(cat_test)):
        mask = cat_test == cat
        if mask.sum() < 10:
            continue
        seg_model = classification_metrics(y_test[mask], model_scores[mask])
        seg_base = classification_metrics(y_test[mask], base_test[mask])
        segments[cat] = {"models": seg_model, "baseline": seg_base}

    calib = calibration_table(y_test, model_scores)

    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_size": int(len(y_test)),
        "note": "All metrics computed on a held-out test split never used for training or models selection.",
        "overall": {"models": model_overall, "baseline": baseline_overall},
        "ranking_quality_job_ranking_for_students": {
            "models": model_rank_by_student, "baseline": base_rank_by_student,
            "n_students_evaluated": n_student_q,
            "definition": "Grouped by student_id: for each student, were the jobs that are true good-matches "
                           "ranked near the top of the list shown to them? Students with zero positive-label "
                           "jobs in the test split are excluded (ranking is undefined with no relevant item).",
        },
        "ranking_quality_candidate_ranking_for_companies": {
            "models": model_rank_by_job, "baseline": base_rank_by_job,
            "n_jobs_evaluated": n_job_q,
            "definition": "Grouped by job_id: for each job, were the candidates who are true good-matches "
                           "ranked near the top of the candidate list shown to the company?",
        },
        "ranking_quality_by_job_category": {
            "models": model_rank_cat, "baseline": base_rank_cat,
            "n_queries_evaluated": n_model_q,
            "definition": "Secondary cut: rows grouped by job category instead of individual student/job, "
                           "as a coarser sanity check.",
        },
        "segment_breakdown_by_job_category": segments,
        "calibration": calib,
        "improvement_over_baseline": {
            "precision_delta": round(model_overall["precision"] - baseline_overall["precision"], 4),
            "recall_delta": round(model_overall["recall"] - baseline_overall["recall"], 4),
            "false_positive_rate_delta": round(
                model_overall["false_positive_rate"] - baseline_overall["false_positive_rate"], 4),
            "roc_auc_delta": round((model_overall["roc_auc"] or 0) - (baseline_overall["roc_auc"] or 0), 4),
        },
    }

    with open(REPORTS / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- plots ----
    prec, rec, _ = precision_recall_curve(y_test, model_scores)
    prec_b, rec_b, _ = precision_recall_curve(y_test, base_test)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"Model (AP={model_overall['pr_auc']:.2f})")
    plt.plot(rec_b, prec_b, label=f"Baseline (AP={baseline_overall['pr_auc']:.2f})", linestyle="--")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall: Model vs Baseline")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORTS / "pr_curve.png", dpi=130); plt.close()

    fpr, tpr, _ = roc_curve(y_test, model_scores)
    fpr_b, tpr_b, _ = roc_curve(y_test, base_test)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"Model (AUC={model_overall['roc_auc']:.2f})")
    plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC={baseline_overall['roc_auc']:.2f})", linestyle="--")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC: Model vs Baseline")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORTS / "roc_curve.png", dpi=130); plt.close()

    plt.figure(figsize=(5, 4))
    buckets = [c["bucket"] for c in calib]
    pred = [c["mean_predicted"] for c in calib]
    actual = [c["actual_positive_rate"] for c in calib]
    x = np.arange(len(buckets))
    plt.plot(x, pred, marker="o", label="Mean predicted score")
    plt.plot(x, actual, marker="s", label="Actual positive rate")
    plt.xticks(x, buckets, rotation=30)
    plt.title("Calibration: predicted score vs actual outcome"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "calibration.png", dpi=130); plt.close()

    # ---- human-readable markdown ----
    md = [f"# PlaceMux Ranking Model — Evaluation Report",
          f"_Generated {report['generated_at']} · test set n={report['test_set_size']}_",
          "",
          "All numbers below are computed on a **held-out test split** the models never trained or tuned on.",
          "",
          "## Overall: models vs baseline (threshold = 0.5)",
          "",
          "| Metric | Baseline (overlap) | Model (calibrated logistic regression) | Δ |",
          "|---|---|---|---|"]
    for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                        ("false_positive_rate", "False Positive Rate"),
                        ("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        b, m = baseline_overall[key], model_overall[key]
        delta = round(m - b, 4) if (m is not None and b is not None) else "-"
        md.append(f"| {label} | {b} | {m} | {delta:+} |" if delta != "-" else f"| {label} | {b} | {m} | - |")

    md += ["", "## Ranking quality — Deliverable 1: Job ranking for students", "",
           f"_Grouped by student; {n_student_q} students with at least one relevant job in the test split._", "",
           "| Metric | Baseline | Model |", "|---|---|---|"]
    for k in model_rank_by_student:
        md.append(f"| {k} | {base_rank_by_student[k]} | {model_rank_by_student[k]} |")

    md += ["", "## Ranking quality — Deliverable 2: Candidate ranking for companies", "",
           f"_Grouped by job; {n_job_q} jobs with at least one relevant candidate in the test split._", "",
           "| Metric | Baseline | Model |", "|---|---|---|"]
    for k in model_rank_by_job:
        md.append(f"| {k} | {base_rank_by_job[k]} | {model_rank_by_job[k]} |")

    md += ["", "## Ranking quality — by job category (secondary cut)", "",
           "| Metric | Baseline | Model |", "|---|---|---|"]
    for k in model_rank_cat:
        md.append(f"| {k} | {base_rank_cat[k]} | {model_rank_cat[k]} |")

    md += ["", "## Segment breakdown by job category", "",
           "| Category | n | Model Precision | Model Recall | Model FPR | Baseline Precision | Baseline Recall |",
           "|---|---|---|---|---|---|---|"]
    for cat, seg in segments.items():
        m, b = seg["models"], seg["baseline"]
        md.append(f"| {cat} | {m['n']} | {m['precision']} | {m['recall']} | {m['false_positive_rate']} "
                   f"| {b['precision']} | {b['recall']} |")

    md += ["", "## Calibration (is a 0.7 score really ~70% likely to be a real match?)", "",
           "| Predicted bucket | n | Mean predicted | Actual positive rate |", "|---|---|---|---|"]
    for c in calib:
        md.append(f"| {c['bucket']} | {c['n']} | {c['mean_predicted']} | {c['actual_positive_rate']} |")

    md += ["", "## Confusion matrix (models, threshold=0.5)", "",
           f"TP={model_overall['confusion_matrix']['tp']}  FP={model_overall['confusion_matrix']['fp']}  "
           f"TN={model_overall['confusion_matrix']['tn']}  FN={model_overall['confusion_matrix']['fn']}",
           "", "![PR Curve](pr_curve.png)", "![ROC Curve](roc_curve.png)", "![Calibration](calibration.png)"]

    with open(REPORTS / "metrics_report.md", "w") as f:
        f.write("\n".join(md))

    print("\n".join(md))


if __name__ == "__main__":
    main()