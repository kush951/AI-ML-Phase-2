# PlaceMux — Task 2: Job Posting with Skill Thresholds

A working AI/ML engineer slice of PlaceMux: a company posts a job gated by **L1–L100**
skill thresholds, and the system ranks verified students against it with a real,
explainable match score — backed by measured precision/recall/false-positive-rate,
not vibes.

## What's inside


## Run it

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install fastapi uvicorn scikit-learn pandas numpy
uvicorn app:app --reload
```

> If `pip install` fails with `externally-managed-environment`, your system Python
> is locked down (common with `uv`-managed or distro Pythons). Either use the venv
> above (recommended) or run `pip install ... --break-system-packages`.

Open **http://localhost:8000**. The frontend is served by the same FastAPI app, so
there's no separate frontend server or CORS setup needed.

## API

| Method | Path | What |
|---|---|---|
| GET | `/api/skills` | The skill taxonomy |
| GET | `/api/students` | Sample students with verified L1–L100 scores |
| GET | `/api/jobs` | All posted jobs |
| POST | `/api/jobs` | Post a job: `{title, company, thresholds: {skill: L1-100}}` — runs threshold validation, 400s with specific errors on bad input |
| GET | `/api/jobs/{id}/candidates` | Ranked candidates: baseline score, model score, met/missing thresholds, plain-English reasons per skill |
| GET | `/api/metrics` | Precision / recall / false-positive-rate, baseline vs model, on held-out data |
| GET | `/api/competency-band/{level}` | Maps a level to a competency band (Novice → Expert) |

## How this maps to the study guide

- **Feature space**: `build_match_vector()` turns a (student, job) pair into 5
  interpretable numbers — coverage ratio, average excess over threshold, total
  deficit, a difficulty-weighted score, and how many skills are gated. No black-box
  embeddings; every number can be explained in a sentence.
- **Baseline first**: `baseline_score()` is the dumb rule (overlap of required vs
  verified skills). The model's whole job is to beat it — see `/api/metrics`.
- **Explainability**: `explain_match()` produces a per-skill, plain-English reason
  for every match or gap ("Python: verified at L24, falls short of the L60
  threshold by 36 levels"). Shown directly in the UI under "Why".
- **Real metrics, not vibes**: `model.py` trains on 752 (student, job) pairs and
  evaluates on 323 **held-out** pairs the model never trained on. Typical run:
  baseline precision ~0.89 / recall ~0.09 (very conservative — it almost never
  says yes) vs model precision ~0.71 / recall ~0.87 (catches far more genuine
  matches, at the cost of a higher false-positive rate). That trade-off is the
  real story to walk through live, not a single accuracy number.
- **Threshold validation**: `validate_thresholds()` rejects unknown skills and
  out-of-range levels (must be a 1–100 integer) before a job is ever posted —
  tested live via the "invalid job" case in the demo script.
- **Generalisation**: the model is evaluated only on the held-out split; the
  3 sample jobs used in the UI are separate from the 40 synthetic jobs used to
  give the model enough training signal, so the demo isn't tuned to itself.

## Demo script (2 minutes)

1. **Post a job** — pick a title/company, check 3–5 skills, set thresholds. Submit
   and show the validation error case too (e.g. a level of 200).
2. **Candidates** — switch jobs via the chips, open "Why" on a top and a borderline
   candidate to show the plain-English reasoning.
3. **Metrics** — show precision/recall/FPR, baseline vs model, and explain the
   trade-off (model catches more real matches, baseline plays it too safe).

## Known simplifications (be upfront about these in review)

- Labels for "good match" are **simulated** from the same signals a recruiter
  would use (see the docstring in `model.py::simulate_label`) since there's no
  real hiring-outcome history yet. Swap in real outcomes once the feedback loop
  exists — nothing else in the pipeline changes.
- In-memory job store (resets on restart). Swap for a real DB before this leaves
  the demo stage.
- No auth — anyone hitting the API can post a job. Out of scope for this task's
  brief but flag it before going further than a demo.