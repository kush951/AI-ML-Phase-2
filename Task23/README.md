# PlaceMux — Registry + Feature Store
### Task 23 · Hardening, Scale & MLOps · AI/ML Engineer

An end-to-end, demoable pipeline that scores student-to-job matches with a
model you can explain and defend with numbers — not "trust me, it works."

```
data/  →  feature_store/  →  models/  →  api/  →  frontend/  →  reports/
generate     build+persist     train+select   FastAPI serve    HTML UI     PDF report
raw data     features once     best of 5       /match          calls API   for founder
```

---

## 1. What's inside

| Folder | What it does |
|---|---|
| `data/` | Generates a real-shaped synthetic dataset: 600 students, 80 jobs, 6,000 labelled (student, job) pairs with realistic noise, missing values, and label noise mimicking imperfect real hiring outcomes. |
| `feature_store/` | **The core deliverable.** A `FeatureStore` class that defines features once, fits an imputer/scaler/encoder on training data, persists them, and reuses the *exact same code path* for every future training run and every live inference call — so training and serving can never drift apart. |
| `models/` | Trains **5 real model families** + 1 rule-based baseline on identical features, evaluates all of them on a held-out test set they never saw during training or tuning, logs every run, and **automatically selects the best model**. Also contains the explainability module. |
| `api/` | FastAPI service that loads the persisted feature store + best model and serves `/match` (known student/job) and `/match-custom` (any new pair) with a live explanation for every score. |
| `frontend/` | Single-file HTML console that calls the API, shows the match score, a probability bar, and the plain-English "why" behind the number. |
| `reports/` | `generate_report.py` produces `PlaceMux_Model_Report.pdf` — leaderboard, baseline comparison, and the pitfall checklist, ready to hand to the founder. |
| `experiments/` | `experiment_log.json` — an MLflow-style append-only log of every training run's metrics (reproducibility). |

---

## 2. Feature Store — how it satisfies the requirement

> "A centralized place where model features are stored and reused."

* **Registry** (`feature_store/feature_store.py::FEATURE_REGISTRY`) — one dictionary
  documenting every feature: what it means and where it comes from
  (skill scores, years of experience, job requirements, certifications,
  aptitude scores, plus derived features like `skill_fit` and `experience_ratio`).
* **Preprocess** — `build_raw_features()` turns raw student/job rows into
  model-ready columns (skill overlap, skill gap, experience ratio, verified
  flag). This function is called identically at training time and inference
  time.
* **Save** — `FeatureStore.fit()` learns a median imputer, a standard
  scaler, and a one-hot encoder on the training split only, then
  `FeatureStore.save()` persists all three (`feature_store/artifacts/feature_store.joblib`)
  with `joblib`.
* **Load during inference** — `api/main.py` calls `FeatureStore.load()` once
  at startup and reuses it for every request — no retraining, no
  re-fitting, no drift.
* **Same features, train + predict** — both `models/train.py` and
  `api/main.py` import `build_raw_features` and the `FeatureStore` class
  from the same module. There is exactly one implementation of "what a
  feature is."
* **Online pair cache** — `get_online_features()` / `materialize_pair()`
  give a lightweight materialized store: repeat lookups for the same
  (student, job) pair reuse cached engineered features; unseen pairs are
  computed fresh with the identical logic.

---

## 3. Multiple models, best-fit selection

`models/train.py` trains and compares, on identical feature-store output:

1. **Baseline** — rule-based skill-overlap threshold (no learning). Every
   other number is judged against this.
2. **Logistic Regression** (class-balanced)
3. **Random Forest**
4. **Gradient Boosting**
5. **SVM (RBF kernel)**
6. **K-Nearest Neighbours**

Each candidate is trained on the train split, scored on a **validation
split**, and the model with the **highest validation ROC-AUC** (tie-broken
by F1) is selected automatically. Its performance is then confirmed on a
completely separate **held-out test split** it never touched during
training or selection — this is what makes the reported numbers real
rather than tuned-to-the-demo.

Latest run on the generated dataset (see `reports/PlaceMux_Model_Report.pdf`
for the full leaderboard):

| | Accuracy | ROC-AUC | F1 | FPR |
|---|---|---|---|---|
| Baseline (skill overlap) | ~67.6% | — | 0.75 | ~40.8% |
| **Selected model (gradient boosting)** | **~77.5%** | **~0.82** | **~0.85** | lower than baseline |

Re-running `python3 models/train.py` will re-fit everything and may select
a different winner if the data or split changes — that's expected and is
exactly the point of automatic model selection instead of hard-coding one
algorithm.

---

## 4. Explainability

Every score returned by the API includes the **top factors** that drove it
for that specific student-job pair (e.g. "the student's average score on
the job's required skills — pushes this TOWARDS a match"), derived from the
selected model's own feature importances / coefficients applied to that
row. No prediction is returned as a bare number.

---

## 5. How to run it end to end

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Generate the dataset (students, jobs, labelled pairs)
python3 data/generate_data.py

# 2. Train, compare 5 models + baseline, auto-select the best, persist artifacts
python3 models/train.py

# 3. Serve the model behind an API
uvicorn api.main:app --reload --port 8000
# -> open http://localhost:8000/docs for the interactive API explorer

# 4. Open the frontend (in a second terminal / just open the file)
#    frontend/index.html calls http://localhost:8000 by default —
#    edit window.API_BASE at the top of the <script> tag if you serve
#    the API elsewhere.
open frontend/index.html      # or just double-click it

# 5. Regenerate the PDF report for the founder
python3 reports/generate_report.py
```

### API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + which model is currently serving |
| GET | `/model-card` | Full leaderboard, baseline metrics, selected model |
| GET | `/students?limit=` | List stored students |
| GET | `/jobs?limit=` | List stored jobs |
| POST | `/match` | `{student_id, job_id}` → probability + explanation |
| POST | `/match-custom` | Score any new student/job pair not in storage |

One real example end-to-end:

```bash
curl -X POST localhost:8000/match -H "Content-Type: application/json" \
  -d '{"student_id": "S0000", "job_id": "J000"}'
```

returns a match probability, a match/no-match verdict, the top explanatory
factors, and which model produced the score — the exact "walk me through
one real example" demo the study guide asks for.

---

## 6. Definition-of-done checklist (Task 23, Section 7)

- [x] MLOps foundation live — training pipeline, experiment log, persisted
      artifacts, served API.
- [x] "Registry + feature store" complete, persisted, and demoable
      end-to-end (data → features → model → API → frontend).
- [x] Numbers, not vibes — baseline vs. 5-model leaderboard with accuracy,
      precision, recall, F1, false-positive rate, ROC-AUC, all on a
      held-out test set.
- [x] Explainable — every prediction ships with a plain-English reason.
- [x] Demoable live — FastAPI `/docs`, curl example, and a working
      frontend console, not slides.

## 7. Known limitations / next steps

* Dataset is synthetic (real-shaped, not real) — swap `data/generate_data.py`
  for a loader against the real verified-skill-score dataset when
  available; the feature store and training pipeline don't need to change.
* No drift-detection or retraining trigger yet — logged as a "go deeper"
  item (Section 12 of the study guide); `experiments/experiment_log.json`
  is the natural place to start.
* No load testing included in this deliverable — this task focused on the
  feature-store + model-selection requirement; add a `locust`/`k6` script
  against the FastAPI service for the scale-testing half of Week 6.
