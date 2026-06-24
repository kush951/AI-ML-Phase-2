"""
generate_data.py
PlaceMux · Task 9 — Synthetic real-shaped data for job matching
Produces: students, jobs, and labelled match pairs (with payment tier)
"""

import numpy as np
import pandas as pd
import random
import json
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Skill taxonomy ──────────────────────────────────────────────────────────
SKILLS = [
    "Python", "Machine Learning", "Data Analysis", "SQL", "Deep Learning",
    "NLP", "Computer Vision", "Data Engineering", "Cloud (AWS/GCP/Azure)",
    "Statistics", "Java", "JavaScript", "React", "Node.js", "FastAPI",
    "Docker", "Kubernetes", "DevOps", "System Design", "Agile/Scrum",
    "Communication", "Leadership", "Project Management", "Business Analysis",
    "Data Visualization", "Tableau", "Power BI", "Excel", "Spark", "Kafka",
]

JOB_ROLES = [
    "Data Scientist", "ML Engineer", "Data Analyst", "Backend Engineer",
    "Frontend Engineer", "Full Stack Developer", "DevOps Engineer",
    "Data Engineer", "Product Manager", "Business Analyst",
]

EDUCATION_LEVELS = ["Diploma", "Bachelor's", "Master's", "PhD"]
LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Remote"]


def generate_student(sid: int, payment_tier: str) -> dict:
    """Generate a single student profile."""
    n_skills = random.randint(3, 10)
    skill_subset = random.sample(SKILLS, n_skills)

    # Paid students get slightly higher verified scores (simulates bias to detect)
    score_boost = 0.05 if payment_tier == "paid" else 0.0
    verified_scores = {
        skill: round(min(1.0, random.gauss(0.6 + score_boost, 0.2)), 2)
        for skill in skill_subset
    }

    return {
        "student_id": f"S{sid:04d}",
        "payment_tier": payment_tier,
        "verified_skills": verified_scores,
        "years_experience": round(random.uniform(0, 8), 1),
        "education_level": random.choice(EDUCATION_LEVELS),
        "location": random.choice(LOCATIONS),
        "preferred_role": random.choice(JOB_ROLES),
        "avg_skill_score": round(np.mean(list(verified_scores.values())), 3),
        "n_verified_skills": n_skills,
    }


def generate_job(jid: int) -> dict:
    """Generate a single job listing."""
    n_required = random.randint(3, 8)
    n_nice = random.randint(1, 4)
    all_req = random.sample(SKILLS, n_required)
    nice_to_have = random.sample([s for s in SKILLS if s not in all_req], n_nice)

    return {
        "job_id": f"J{jid:04d}",
        "role": random.choice(JOB_ROLES),
        "required_skills": all_req,
        "nice_to_have": nice_to_have,
        "min_experience": random.choice([0, 1, 2, 3, 5]),
        "education_required": random.choice(EDUCATION_LEVELS),
        "location": random.choice(LOCATIONS + ["Any"]),
        "salary_lpa": round(random.uniform(4, 40), 1),
        "n_required_skills": n_required,
    }


def compute_match_features(student: dict, job: dict) -> dict:
    """Compute features for a (student, job) pair — these feed the model."""
    s_skills = set(student["verified_skills"].keys())
    j_required = set(job["required_skills"])
    j_nice = set(job["nice_to_have"])

    overlap = s_skills & j_required
    overlap_nice = s_skills & j_nice

    skill_coverage = len(overlap) / max(len(j_required), 1)
    avg_score_on_required = (
        np.mean([student["verified_skills"][sk] for sk in overlap])
        if overlap else 0.0
    )

    edu_rank = {d: i for i, d in enumerate(EDUCATION_LEVELS)}
    edu_match = int(
        edu_rank.get(student["education_level"], 0)
        >= edu_rank.get(job["education_required"], 0)
    )

    exp_match = int(student["years_experience"] >= job["min_experience"])
    loc_match = int(
        job["location"] == "Any"
        or job["location"] == student["location"]
    )
    nice_coverage = len(overlap_nice) / max(len(j_nice), 1)

    return {
        "skill_coverage": round(skill_coverage, 4),
        "avg_score_on_required": round(avg_score_on_required, 4),
        "n_skills_matched": len(overlap),
        "nice_to_have_coverage": round(nice_coverage, 4),
        "edu_match": edu_match,
        "exp_match": exp_match,
        "loc_match": loc_match,
        "student_avg_skill": student["avg_skill_score"],
        "years_experience": student["years_experience"],
        "n_student_skills": student["n_verified_skills"],
        "n_job_required": job["n_required_skills"],
        "payment_tier_paid": int(student["payment_tier"] == "paid"),
    }


def compute_ground_truth(features: dict) -> int:
    """
    Rule-based ground truth:
    A match is GOOD if skill coverage ≥ 0.5, edu & exp both match,
    and avg score on required skills ≥ 0.5.
    """
    return int(
        features["skill_coverage"] >= 0.5
        and features["edu_match"] == 1
        and features["exp_match"] == 1
        and features["avg_score_on_required"] >= 0.45
    )


def generate_dataset(n_students=400, n_jobs=80, pairs_per_student=5) -> pd.DataFrame:
    """Build labelled pairs dataset."""
    # Split students: 60% paid, 40% free
    n_paid = int(n_students * 0.6)
    students = (
        [generate_student(i, "paid") for i in range(n_paid)]
        + [generate_student(i + n_paid, "free") for i in range(n_students - n_paid)]
    )
    jobs = [generate_job(j) for j in range(n_jobs)]

    records = []
    for student in students:
        sampled_jobs = random.sample(jobs, min(pairs_per_student, len(jobs)))
        for job in sampled_jobs:
            feats = compute_match_features(student, job)
            label = compute_ground_truth(feats)
            records.append({
                "student_id": student["student_id"],
                "job_id": job["job_id"],
                "payment_tier": student["payment_tier"],
                "role": job["role"],
                "preferred_role": student["preferred_role"],
                "label": label,
                **feats,
            })

    df = pd.DataFrame(records)
    print(f"Dataset shape  : {df.shape}")
    print(f"Positive rate  : {df['label'].mean():.3f}")
    print(f"Paid students  : {(df['payment_tier']=='paid').sum()} pairs")
    print(f"Free students  : {(df['payment_tier']=='free').sum()} pairs")

    # Save
    out = Path(__file__).parent
    df.to_csv(out / "matches.csv", index=False)
    pd.DataFrame(students).to_json(out / "students.json", orient="records", indent=2)
    pd.DataFrame(jobs).to_json(out / "jobs.json", orient="records", indent=2)
    print("Saved → data/matches.csv, students.json, jobs.json")
    return df


if __name__ == "__main__":
    generate_dataset()
