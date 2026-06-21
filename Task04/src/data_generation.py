"""
PlaceMux — Explainability Task
Stage A.3 — Pull / generate the agreed sample data.

Generates a real-shaped (not toy) student <-> job application dataset.

Design notes (why it's not a toy example):
  * Skills are drawn from a realistic 40-skill taxonomy across 4 domains
    (Software, Data/AI, Design, Business) so overlap isn't trivially binary.
  * Verified skill scores are noisy (a student's "true" ability vs. the
    score the verification quiz returned) — mirrors real assessment noise.
  * The label (`is_good_match`) is generated from a latent quality function
    plus genuine label noise (~8%), so no model can hit 100% — accuracy
    has to be earned, and a baseline is meaningful to compare against.
  * JDs have both "required" and "nice-to-have" skills with different
    weights, and a seniority/experience gap term, to mimic a real ATS-style
    matching problem rather than pure set-overlap.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

DOMAINS = {
    "Software": ["Python", "Java", "JavaScript", "SQL", "Git", "REST APIs",
                 "Docker", "Linux", "Data Structures", "System Design"],
    "Data/AI": ["Python", "SQL", "Pandas", "Scikit-learn", "Statistics",
                "Machine Learning", "Deep Learning", "NLP", "Data Visualization", "MLOps"],
    "Design": ["Figma", "UI Design", "UX Research", "Wireframing",
               "Design Systems", "Prototyping", "Typography", "Adobe XD",
               "User Testing", "Visual Design"],
    "Business": ["Excel", "SQL", "Market Research", "Financial Modeling",
                 "Presentation", "Negotiation", "Project Management",
                 "Stakeholder Management", "Business Analytics", "CRM Tools"],
}
ALL_SKILLS = sorted({s for skills in DOMAINS.values() for s in skills})

N_STUDENTS = 600
N_JOBS = 120
N_PAIRS = 4000  # application/shortlist candidate pairs


def gen_students(n):
    rows = []
    for sid in range(1, n + 1):
        domain = RNG.choice(list(DOMAINS.keys()))
        pool = DOMAINS[domain]
        n_skills = RNG.integers(4, 9)
        skills = list(RNG.choice(pool, size=min(n_skills, len(pool)), replace=False))
        # small chance of a cross-domain skill (real students aren't siloed)
        if RNG.random() < 0.25:
            skills.append(RNG.choice(ALL_SKILLS))
        skills = sorted(set(skills))

        true_ability = RNG.normal(65, 15)  # latent, unobserved
        verified_scores = {}
        for s in skills:
            noise = RNG.normal(0, 8)
            score = np.clip(true_ability + noise + RNG.normal(0, 5), 5, 100)
            verified_scores[s] = round(float(score), 1)

        years_exp = float(np.clip(RNG.exponential(1.4), 0, 8))
        rows.append({
            "student_id": f"S{sid:04d}",
            "domain": domain,
            "skills": skills,
            "verified_scores": verified_scores,
            "years_experience": round(years_exp, 1),
            "true_ability": round(float(true_ability), 1),  # NOT a feature — kept only for audit
        })
    return rows


def gen_jobs(n):
    rows = []
    for jid in range(1, n + 1):
        domain = RNG.choice(list(DOMAINS.keys()))
        pool = DOMAINS[domain]
        n_req = RNG.integers(3, 6)
        required = list(RNG.choice(pool, size=min(n_req, len(pool)), replace=False))
        remaining = [s for s in pool if s not in required]
        n_nice = RNG.integers(1, 4)
        nice_to_have = list(RNG.choice(remaining, size=min(n_nice, len(remaining)), replace=False)) if remaining else []

        min_years = float(RNG.choice([0, 0, 1, 1, 2, 3]))
        min_score = float(RNG.choice([40, 50, 55, 60, 65]))

        rows.append({
            "job_id": f"J{jid:04d}",
            "domain": domain,
            "required_skills": sorted(required),
            "nice_to_have_skills": sorted(nice_to_have),
            "min_years_experience": min_years,
            "min_verified_score": min_score,
        })
    return rows


def build_features(students, jobs, n_pairs):
    s_by_id = {s["student_id"]: s for s in students}
    j_by_id = {j["job_id"]: j for j in jobs}
    s_ids = list(s_by_id.keys())
    j_ids = list(j_by_id.keys())

    records = []
    seen = set()
    attempts = 0
    while len(records) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        sid = RNG.choice(s_ids)
        jid = RNG.choice(j_ids)
        key = (sid, jid)
        if key in seen:
            continue
        seen.add(key)

        s = s_by_id[sid]
        j = j_by_id[jid]
        s_skills = set(s["skills"])
        req = set(j["required_skills"])
        nice = set(j["nice_to_have_skills"])

        req_overlap = s_skills & req
        nice_overlap = s_skills & nice

        req_overlap_ratio = len(req_overlap) / max(len(req), 1)
        nice_overlap_ratio = len(nice_overlap) / max(len(nice), 1)

        overlap_scores = [s["verified_scores"][sk] for sk in req_overlap]
        avg_req_score = float(np.mean(overlap_scores)) if overlap_scores else 0.0
        min_req_score = float(np.min(overlap_scores)) if overlap_scores else 0.0

        domain_match = 1.0 if s["domain"] == j["domain"] else 0.0
        years_gap = s["years_experience"] - j["min_years_experience"]
        meets_min_score = 1.0 if avg_req_score >= j["min_verified_score"] else 0.0
        verified_breadth = len(s["skills"])

        # ---- latent "true" match quality (not given to the model directly) ----
        latent = (
            3.2 * req_overlap_ratio
            + 0.9 * nice_overlap_ratio
            + 0.02 * avg_req_score
            + 0.9 * domain_match
            + 0.35 * np.tanh(years_gap)
            + 0.015 * min_req_score
            - 1.6
        )
        prob = 1 / (1 + np.exp(-latent))
        label = 1 if RNG.random() < prob else 0
        # genuine label noise — even a "correct" verification process has error
        if RNG.random() < 0.08:
            label = 1 - label

        records.append({
            "student_id": sid,
            "job_id": jid,
            "domain_match": domain_match,
            "required_overlap_count": len(req_overlap),
            "required_overlap_ratio": round(req_overlap_ratio, 4),
            "nice_overlap_count": len(nice_overlap),
            "nice_overlap_ratio": round(nice_overlap_ratio, 4),
            "avg_required_skill_score": round(avg_req_score, 2),
            "min_required_skill_score": round(min_req_score, 2),
            "meets_min_score_threshold": meets_min_score,
            "years_experience": s["years_experience"],
            "min_years_required": j["min_years_required"] if "min_years_required" in j else j["min_years_experience"],
            "years_gap": round(years_gap, 2),
            "verified_skill_breadth": verified_breadth,
            "is_good_match": label,
        })
    return pd.DataFrame.from_records(records)


def main():
    students = gen_students(N_STUDENTS)
    jobs = gen_jobs(N_JOBS)
    df = build_features(students, jobs, N_PAIRS)

    out_dir = str(DATA_DIR)
    df.to_csv(f"{out_dir}/match_pairs.csv", index=False)

    with open(f"{out_dir}/students.json", "w") as f:
        json.dump(students, f, indent=2)
    with open(f"{out_dir}/jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"students: {len(students)}, jobs: {len(jobs)}, pairs: {len(df)}")
    print(f"positive rate: {df['is_good_match'].mean():.3f}")
    print(df.head())


if __name__ == "__main__":
    main()
