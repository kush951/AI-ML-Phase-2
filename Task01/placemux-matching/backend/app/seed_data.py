"""
Generates a real-shaped sample dataset (students, companies, jobs, and
historical outcome labels) so the matching model has something to train
and evaluate against before live marketplace data exists.

This is clearly-synthetic data used for demo/training purposes — the
study guide explicitly asks for "real-shaped, even if small" sample data,
not a hand-picked toy example. Swap this generator for a real export
once Backend's marketplace data model is live; nothing downstream
(features.py, matching.py) needs to change.
"""
import random
from typing import List, Dict, Tuple

random.seed(42)

SKILL_POOL = [
    "python", "sql", "javascript", "react", "node", "java", "c++",
    "machine_learning", "data_analysis", "statistics", "excel",
    "communication", "project_management", "aws", "docker", "kubernetes",
    "figma", "ui_design", "product_sense", "git", "testing",
    "django", "fastapi", "spring_boot", "tableau", "powerbi",
    "nlp", "computer_vision", "devops", "sales", "content_writing",
]

LOCATIONS = ["Mumbai", "Pune", "Bengaluru", "Hyderabad", "Delhi NCR", "Remote"]

ROLE_TITLES = [
    "Junior Data Analyst", "Backend Engineer", "Frontend Engineer",
    "ML Engineer Intern", "Product Analyst", "Full-Stack Developer",
    "QA Engineer", "DevOps Engineer", "UI/UX Designer", "Business Analyst",
]

COMPANY_NAMES = [
    "Vertex Robotics", "NimbusCart", "Quanta Health", "BrightPay",
    "Loopline Logistics", "Aarav Fintech", "Solstice Labs", "GreenGrid Energy",
]


def _sample_skills(n_low=3, n_high=8, score_low=40, score_high=98) -> Dict[str, float]:
    n = random.randint(n_low, n_high)
    skills = random.sample(SKILL_POOL, n)
    return {s: round(random.uniform(score_low, score_high), 1) for s in skills}


def generate_companies(n=8) -> List[Dict]:
    return [
        {"name": name, "email": f"hr@{name.lower().replace(' ', '')}.com", "location": random.choice(LOCATIONS)}
        for name in COMPANY_NAMES[:n]
    ]


def generate_students(n=300) -> List[Dict]:
    students = []
    for i in range(n):
        students.append({
            "name": f"Student_{i+1:03d}",
            "location": random.choice(LOCATIONS),
            "years_experience": round(random.uniform(0, 6), 1),
            "education_level": random.choice([1, 2, 2, 2, 3, 3, 4]),
            "verified_skills": _sample_skills(3, 9, 35, 99),
            "remote_ok": random.random() < 0.7,
        })
    return students


def generate_jobs(company_ids: List[int], n=60) -> List[Dict]:
    jobs = []
    for i in range(n):
        req_n = random.randint(3, 6)
        skills = random.sample(SKILL_POOL, req_n)
        jobs.append({
            "company_id": random.choice(company_ids),
            "title": random.choice(ROLE_TITLES),
            "location": random.choice(LOCATIONS),
            "min_experience": round(random.choice([0, 0, 1, 1, 2, 3]), 1),
            "role_level": random.choice([1, 2, 2, 3]),
            "required_skills": {s: round(random.uniform(40, 80), 1) for s in skills},
            "remote_ok": random.random() < 0.6,
        })
    return jobs


def generate_labeled_pairs(students: List[Dict], jobs: List[Dict], pairs_per_job=25) -> List[Tuple[Dict, Dict, int]]:
    """
    Synthetic historical outcome labels (1 = went on to a successful
    shortlist/hire, 0 = did not), generated from an underlying rule with
    noise so the data is real-shaped: imperfectly separable, not a toy
    linear-sep example. Used purely to train/evaluate the demo model.
    """
    from .features import compute_features

    pairs = []
    for job in jobs:
        candidates = random.sample(students, min(pairs_per_job, len(students)))
        for student in candidates:
            f = compute_features(student, job)
            # underlying "true" propensity for a good outcome
            score = (
                0.45 * f["skill_coverage_weighted"]
                + 0.20 * f["skill_overlap_ratio"]
                + 0.15 * f["experience_met"]
                + 0.10 * f["location_match"]
                + 0.10 * (f["avg_skill_gap"] + 1) / 2
            )
            score += random.gauss(0, 0.12)  # real-world noise
            label = 1 if score >= 0.62 else 0
            pairs.append((student, job, label))
    return pairs
