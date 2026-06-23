"""
PlaceMux · AI/ML · Data Generator
Generates realistic student-job matching dataset for training and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple

np.random.seed(42)

SKILLS = [
    "Python", "JavaScript", "React", "SQL", "Machine Learning",
    "Data Analysis", "Communication", "Leadership", "Project Management",
    "Node.js", "AWS", "Docker", "Java", "C++", "Product Design",
    "Excel", "Tableau", "Statistics", "NLP", "Deep Learning",
    "Git", "Agile", "Problem Solving", "FastAPI", "MongoDB"
]

JOB_ROLES = [
    ("Data Scientist", ["Python", "Machine Learning", "Statistics", "SQL", "Deep Learning"]),
    ("Frontend Developer", ["JavaScript", "React", "Git", "Problem Solving", "Agile"]),
    ("Backend Developer", ["Python", "Node.js", "SQL", "Docker", "AWS"]),
    ("ML Engineer", ["Python", "Machine Learning", "Deep Learning", "Docker", "FastAPI"]),
    ("Data Analyst", ["SQL", "Excel", "Tableau", "Statistics", "Data Analysis"]),
    ("Product Manager", ["Communication", "Leadership", "Agile", "Project Management", "Product Design"]),
    ("DevOps Engineer", ["Docker", "AWS", "Git", "Python", "Agile"]),
    ("NLP Engineer", ["Python", "NLP", "Deep Learning", "Machine Learning", "Statistics"]),
]


def generate_student(student_id: int) -> dict:
    n_skills = np.random.randint(4, 12)
    chosen_skills = np.random.choice(SKILLS, n_skills, replace=False)
    skill_scores = {skill: round(np.random.beta(2, 1.5) * 100, 1) for skill in chosen_skills}
    return {
        "student_id": f"STU{student_id:04d}",
        "years_experience": round(np.random.exponential(1.5), 1),
        "cgpa": round(float(np.clip(np.random.normal(7.5, 1.0), 4.0, 10.0)), 2),
        **{f"skill_{s.replace(' ', '_').lower()}": skill_scores.get(s, 0.0) for s in SKILLS}
    }


def generate_job(job_id: int) -> dict:
    role_name, core_skills = JOB_ROLES[job_id % len(JOB_ROLES)]
    min_exp = round(np.random.choice([0, 0.5, 1, 2, 3]), 1)
    min_cgpa = round(np.random.choice([6.0, 6.5, 7.0, 7.5]), 1)
    required = {s: round(np.random.uniform(60, 90), 1) for s in core_skills}
    extra = np.random.choice([s for s in SKILLS if s not in core_skills],
                             np.random.randint(0, 4), replace=False)
    optional = {s: round(np.random.uniform(40, 70), 1) for s in extra}
    return {
        "job_id": f"JOB{job_id:04d}",
        "role": role_name,
        "min_experience": min_exp,
        "min_cgpa": min_cgpa,
        **{f"req_skill_{s.replace(' ', '_').lower()}": required.get(s, 0.0) for s in SKILLS},
        **{f"opt_skill_{s.replace(' ', '_').lower()}": optional.get(s, 0.0) for s in SKILLS},
    }


def compute_label(student: dict, job: dict) -> Tuple[int, float]:
    """Compute ground-truth match label and continuous match score."""
    skill_overlap = 0
    skill_deficit = 0
    total_required = 0

    for skill in SKILLS:
        col = skill.replace(' ', '_').lower()
        stu_score = student.get(f"skill_{col}", 0.0)
        req_score = job.get(f"req_skill_{col}", 0.0)
        if req_score > 0:
            total_required += 1
            if stu_score >= req_score * 0.75:
                skill_overlap += 1
            else:
                skill_deficit += req_score - stu_score

    coverage = skill_overlap / max(total_required, 1)
    exp_ok = student["years_experience"] >= job["min_experience"]
    cgpa_ok = student["cgpa"] >= job["min_cgpa"]
    base_score = coverage * 0.6 + (0.2 if exp_ok else 0) + (0.2 if cgpa_ok else 0)
    noise = np.random.normal(0, 0.05)
    match_score = float(np.clip(base_score + noise, 0, 1))
    label = 1 if match_score >= 0.55 else 0
    return label, round(match_score, 4)


def generate_dataset(n_students=500, n_jobs=100, pairs_per_student=8) -> pd.DataFrame:
    students = [generate_student(i) for i in range(n_students)]
    jobs = [generate_job(i) for i in range(n_jobs)]

    records = []
    for stu in students:
        sampled_jobs = np.random.choice(jobs, pairs_per_student, replace=False)
        for job in sampled_jobs:
            label, score = compute_label(stu, job)
            records.append({**stu, **job, "match_label": label, "match_score": score})

    df = pd.DataFrame(records)
    return df, students, jobs


if __name__ == "__main__":
    df, _, _ = generate_dataset()
    df.to_csv("/home/claude/placemux/ml/data/placemux_dataset.csv", index=False)
    print(f"Dataset: {df.shape}, Positive rate: {df['match_label'].mean():.2%}")
