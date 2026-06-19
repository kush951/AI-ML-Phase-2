# Matching API Contract — PlaceMux

This is the contract the matching engine exposes to **Backend** (marketplace
data model) and **Frontend** (candidate / company views). It's a live
FastAPI app, not a mock — every endpoint below is implemented in
`backend/app/main.py` and backed by persisted SQLite data
(`backend/data/placemux.db`).

Base URL (local dev): `http://127.0.0.1:8000`
Interactive docs (auto-generated): `http://127.0.0.1:8000/docs`

---

## 1. Companies

### `POST /companies/signup`
Founder-verification path: "let a company sign up."

Request:
```json
{ "name": "Orbital Stack", "email": "hr@orbitalstack.com", "location": "Pune" }
```
Response `200`:
```json
{ "id": 9, "name": "Orbital Stack", "email": "hr@orbitalstack.com", "location": "Pune" }
```
`409` if the email is already registered.

### `GET /companies`
Returns all companies on the marketplace.

---

## 2. Jobs

### `POST /companies/{company_id}/jobs`
Request:
```json
{
  "title": "Backend Engineer",
  "location": "Pune",
  "min_experience": 1,
  "required_skills": { "python": 60, "sql": 50, "fastapi": 40 },
  "remote_ok": true
}
```
`required_skills` is `skill -> minimum verified score (0–100)`. This map
*is* the job's feature-space bar — every candidate is scored against it.

Response `200`: the created `Job`. `404` if the company hasn't signed up.

### `GET /jobs`
All jobs across all companies (used by Frontend to populate the
candidates view).

### `GET /companies/{company_id}/jobs`
A single company's jobs.

---

## 3. Students

### `POST /students`
Request:
```json
{
  "name": "Asha Verma",
  "location": "Pune",
  "years_experience": 1.5,
  "education_level": 2,
  "verified_skills": { "python": 78, "sql": 65, "fastapi": 55 },
  "remote_ok": true
}
```
`verified_skills` comes from PlaceMux's upstream skill-verification
pipeline — this contract assumes scores already exist; it doesn't compute
them.

### `GET /students`
Lists students (capped at 200 for the demo).

---

## 4. Matching (the core deliverable)

### `GET /jobs/{job_id}/candidates?top_k=10`
Ranks every student against a job. This is the primary endpoint Frontend
calls for the company's "candidates" view.

Response `200`:
```json
[
  {
    "student_id": 44,
    "student_name": "Student_044",
    "job_id": 26,
    "model_score": 0.9939,
    "baseline_score": 0.6667,
    "rank": 1,
    "explanation": {
      "matched_skills": ["fastapi", "python"],
      "missing_skills": ["sql"],
      "extra_skills": ["devops", "django", "..."],
      "experience_fit": "meets the 1.0yr requirement (4.3yr on record)",
      "location_fit": "remote/location compatible",
      "plain_english": "Cleared 2/3 required skills (fastapi, python); missing or below bar on sql. Candidate meets the 1.0yr requirement (4.3yr on record)."
    }
  }
]
```
- `model_score` — calibrated match probability (0–1) from the trained
  classifier. This is what ranking is sorted by.
- `baseline_score` — the dumb skill-overlap baseline, returned alongside
  so the lift is always visible, never hidden.
- `explanation` — the plain-English "why," required for every match per
  the study guide's explainability principle.

Every call is persisted to `match_log` so scores are auditable later, not
ephemeral.

### `GET /students/{student_id}/jobs?top_k=10`
Reciprocal: ranks jobs for a given student (candidate-facing view).

### `GET /match/explain?student_id=&job_id=`
Single-pair breakdown — "walk me through this student, this job, and
why it's a match," exactly as Section 8's verification script asks for.

---

## 5. Metrics (Definition-of-Done evidence)

### `GET /metrics`
Held-out test-set precision, recall, false-positive rate, ROC AUC, and
accuracy for the trained model **and** the baseline, plus the model's
feature coefficients (explainability at the model level, not just the
per-match level).

```json
{
  "model_version": "logreg_v1",
  "trained_on_pairs": 1500,
  "test_set_size": 375,
  "baseline": { "precision": 0.31, "recall": 0.72, "false_positive_rate": 0.13, "roc_auc": 0.88, ... },
  "model":    { "precision": 0.39, "recall": 0.76, "false_positive_rate": 0.10, "roc_auc": 0.93, ... },
  "lift_over_baseline_pct": 23.17
}
```

### `POST /metrics/retrain`
Re-runs training on a fresh sample draw and appends to the append-only
experiment log (`backend/artifacts/experiment_log.jsonl`) — every run's
numbers stay reproducible, per the study guide's "keep the experiment
log" instruction.

---

## 6. Hand-off notes

- **To Backend:** swap `app/seed_data.py`'s synthetic generator for a real
  export from the marketplace data model once it's live. Nothing in
  `features.py`, `matching.py`, or `main.py` needs to change — they only
  depend on the `Student` / `Job` shapes above.
- **To Frontend:** every field in this contract is what `frontend/app.js`
  consumes today. Treat this doc as the source of truth if the two drift.
- **Upstream dependency:** verified skill scores (`verified_skills`,
  `required_skills`) are assumed to already exist on input — this task
  doesn't compute them.
