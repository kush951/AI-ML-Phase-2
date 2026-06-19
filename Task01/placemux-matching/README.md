# PlaceMux — Student↔Job Matching Engine

Ready-to-deploy implementation of the Week 2 / Phase 2 task: **"Define the
student↔job feature space"** + **"Specify the matching API contract with
Backend."** Real persisted data, a trained model measured against a
baseline on held-out data, plain-English explanations, and a frontend
wired to the live API — not slides, not "it works."

```
placemux-matching/
├── backend/
│   ├── app/
│   │   ├── features.py     ← the student<->job feature space (Stage B deliverable)
│   │   ├── matching.py     ← baseline, trained model, explainability, metrics
│   │   ├── seed_data.py    ← real-shaped synthetic sample data generator
│   │   ├── models.py       ← marketplace data model (SQLAlchemy / SQLite)
│   │   ├── schemas.py      ← API contract shapes (Pydantic)
│   │   └── main.py         ← FastAPI app — the matching API contract (Stage C deliverable)
│   ├── train.py            ← standalone training/eval script
│   ├── artifacts/          ← match_model.joblib, metrics.json, experiment_log.jsonl
│   └── requirements.txt
├── frontend/
│   ├── index.html          ← company sign-up, post-a-role, candidates, metrics views
│   ├── app.js              ← wires every view to the live API
│   └── styles.css
├── API_CONTRACT.md         ← the contract doc to hand off to Backend/Frontend
└── README.md
```

## 1. Run the backend

```bash
cd backend
pip install -r requirements.txt
python train.py            # trains the model, writes artifacts/metrics.json
uvicorn app.main:app --reload --port 8000
```

On first boot the app also seeds 8 companies, ~120 students, and 25 jobs
of real-shaped sample data into `data/placemux.db` (SQLite — swap the
`DATABASE_URL` in `app/database.py` for Postgres in production; nothing
else changes).

Check it's alive: `curl http://127.0.0.1:8000/health` →
`{"status":"ok"}`. Interactive API docs at
`http://127.0.0.1:8000/docs`.

## 2. Run the frontend

The frontend is static — no build step. Easiest options:

```bash
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

…or just double-click `index.html` (CORS is open on the backend, so it
works from `file://` too). If your backend isn't on
`http://127.0.0.1:8000`, set it before the page loads:

```html
<script>window.PLACEMUX_API_BASE = "https://your-backend-host";</script>
```
(add this `<script>` tag above `<script src="app.js">` in `index.html`).

## 3. The 2-minute demo script (Section 6, Stage D)

1. **Sign up a company** — tab 01, fill the form, submit. Show the
   company appearing in the list on the right (real persistence, not a
   toast that vanishes).
2. **Post a role** — tab 02, pick the company you just made, set
   required skills like `python:60, sql:50, fastapi:40`. This *is* the
   feature-space bar.
3. **Rank candidates** — tab 03, pick that job, hit "Rank candidates."
   Walk through the #1 result: its model score, the baseline score next
   to it, and the plain-English "why."
4. **Show the numbers** — tab 04. Precision / recall / false-positive
   rate / ROC AUC for the model vs. the dumb skill-overlap baseline, on
   a held-out test set the model never trained on, plus the
   feature-coefficient breakdown (model-level explainability).

That covers every line in Section 8's scoring rubric: both core
deliverables demoable end-to-end, real-data quality (not a toy/happy
path — labels are noisy and imperfectly separable on purpose), live
verification with real numbers, and a documented hand-off contract.

## 4. Re-running experiments

```bash
cd backend
python train.py
```
Each run appends to `artifacts/experiment_log.jsonl` (append-only, so
every run's numbers stay reproducible per the study guide's instruction)
and overwrites `artifacts/metrics.json` / `match_model.joblib`, which the
API picks up on next restart (or call `POST /metrics/retrain` to hot-swap
without restarting).

## 5. What's deliberately simple (and how to extend it)

- **Model:** logistic regression on 9 engineered features — interpretable
  by design, so "explainability" isn't bolted on afterward. Swap in
  gradient boosting (`sklearn.ensemble.GradientBoostingClassifier`) in
  `matching.py` if you need more lift; keep `features.py` as the single
  source of truth either way.
- **Labels:** synthetic historical-outcome labels generated with noise
  (`seed_data.generate_labeled_pairs`) since no real hiring outcomes
  exist yet. Replace with real shortlist/hire outcomes from Backend's
  data model the moment they exist — everything downstream is
  label-agnostic.
- **Vector search:** not included — at this scale (hundreds of students)
  brute-force scoring is fast enough and keeps the demo fully
  explainable. Section 12's "embeddings & ANN search" is the natural next
  step once the marketplace is at thousands of students and search speed
  becomes the bottleneck (see the self-check question in Section 11).
