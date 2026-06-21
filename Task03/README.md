# PlaceMux — Search & Discovery (Task 3, AI/ML Engineer, Week 2 Phase 2)

End-to-end implementation of **Job ranking for students** and **Candidate ranking
for companies (v1)**: data → features → trained model → database → API → frontend,
with a comprehensive, held-out evaluation report.

This revision specifically addresses the previous review's feedback:

| Feedback | Where it's addressed |
|---|---|
| Expand evaluation metrics | `ml/evaluate.py` / `reports/metrics_report.md` — precision, recall, F1, FPR, ROC-AUC, PR-AUC, NDCG@5/10, Precision@k, MAP@10 computed **separately for both deliverables**, plus an 8-category segment breakdown and a calibration table |
| More detailed testing & validation evidence | `tests/` — 29 automated tests (unit + API integration), `reports/test_run_output.txt` is a saved passing run |
| Database integration | `db/` — real SQLAlchemy/SQLite schema, every ranking call persists to `match_scores` (see `test_ranking_calls_persist_to_database`) |

## Architecture

```
data/        synthetic-but-realistic sample data generator (students, jobs, skills, labelled applications)
ml/          features.py (shared feature engineering), baseline.py (dumb overlap ranker),
             train.py (calibrated logistic regression + experiment log), evaluate.py (full metrics suite)
db/          SQLAlchemy models + seed script -> placemux.db (SQLite)
api/         FastAPI service: ranking endpoints, persistence, error handling, serves the frontend
frontend/    single-page UI: student job search, company candidate search, live model report
tests/       29 pytest tests: feature logic + full API integration
reports/     metrics_report.json/.md, PR/ROC/calibration plots, experiment_log.csv, test_run_output.txt
```

## Why this design

- **Baseline first.** `ml/baseline.py` is the "dumb" unweighted skill-overlap ranker described in the
  study guide. Every model number is reported next to it, never alone.
- **Explainability is structural, not an afterthought.** `ml/features.py` returns a feature vector
  *and* a plain-English reason list in the same call, so the API can never return a score without a "why".
- **Calibrated logistic regression**, not a deep net. ~5 engineered features, a few thousand
  rows, and the #1 product requirement is "a hiring decision you can explain" — a linear model gives
  auditable coefficients and well-behaved probabilities.
- **Real-shaped data, not a toy.** 420 students, 140 jobs, 692 weighted skill requirements, 5,880
  labelled applications with a noisy (not deterministic) ground-truth function — positive rate ~29%,
  not a clean 50/50 toy split.
- **Train/val/test, stratified, held out.** Reported numbers come exclusively from the 20% test split,
  never used for training or threshold selection.

## Quickstart

```bash
pip install -r requirements.txt

python3 data/generate_data.py     # generates the sample dataset (CSV)
python3 db/seed.py                # loads it into placemux.db
python3 ml/train.py               # trains the model, appends to reports/experiment_log.csv
python3 ml/evaluate.py            # full evaluation report -> reports/metrics_report.{json,md}

python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8011
# open http://127.0.0.1:8011  -> the frontend (student / company / model report tabs)
```

Run the tests:
```bash
pytest tests/ -v
```

## API contract

| Endpoint | Purpose |
|---|---|
| `GET /students/{id}/jobs?limit=&category=` | Deliverable 1 — ranked jobs for a student, with score, baseline score, explanation, and threshold flag. Persists every result to `match_scores`. |
| `GET /jobs/{id}/candidates?limit=` | Deliverable 2 — ranked candidates for a job, same shape. |
| `GET /students/{id}`, `GET /jobs/{id}` | Entity detail lookups. 404 on unknown id. |
| `GET /metrics` | Serves the full evaluation report (`reports/metrics_report.json`) live to the frontend's "Model report" tab. |
| `GET /health` | Liveness + whether the model/DB are loaded. |

## One real example, end to end (live demo script)

1. Open the **Student view**, pick any student, click **Rank jobs**. Each card shows the model's
   match score *and* the baseline score side by side, plus the specific skills that pushed the
   score up or down (e.g. "Meets requirement on: Python (82/100, needs 50)... Below requirement
   on: Communication (0/100, needs 50)").
2. Switch to **Company view**, pick a job, click **Rank candidates** — same model, same
   explainability, the other direction.
3. Switch to **Model report** to see precision/recall/FPR/ROC-AUC/NDCG, the baseline comparison,
   the per-category segment breakdown, and the calibration table — all computed on the held-out
   test set, refreshed live from `/metrics`.
4. Query `match_scores` in `placemux.db` to show every ranking the API has ever served is
   persisted, not just held in memory.

## Headline numbers (held-out test set, n=1,176; see `reports/metrics_report.md` for full detail)

- Precision 0.61 / Recall 0.12 / F1 0.20 / FPR 0.031 / ROC-AUC 0.68 / PR-AUC 0.48 — model beats
  baseline on F1, FPR-adjusted recall, ROC-AUC and PR-AUC at the default 0.5 threshold.
- Job ranking for students (grouped by student, 231 students evaluated): NDCG@5 0.83 vs baseline 0.82.
- Candidate ranking for companies (grouped by job, 129 jobs evaluated): NDCG@10 0.75, comparable to
  baseline — documented honestly rather than cherry-picked, see the segment table for where the
  model under/over-performs the baseline by job category.
- Calibration: predicted-probability buckets track actual outcome rates closely (e.g. the 0.6–0.8
  bucket predicts 0.67 on average and is actually right 0.70 of the time) — the score is a real
  probability, not an arbitrary ranking number.

## Known limitations (stated, not hidden)

- Sample data is synthetic; production would replace `data/generate_data.py` with the real upstream
  feed once "Search index; match vectors" (this task's stated dependency) lands.
- Ranking-quality metrics group by category and by individual student/job, but production query
  groups will be larger (a real student sees hundreds of jobs); NDCG@k should be re-validated at
  that scale per the study guide's "generalisation" warning.
- The 0.5 recommendation threshold is a simple default; a production guardrail metric (e.g. relevance
  after a future pricing or catalog change) is not yet wired up — flagged here per the study guide's
  "real metrics, not vibes" guidance rather than left silent.
