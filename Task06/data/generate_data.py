"""
PlaceMux · Data Generator
Produces real-shaped synthetic candidate–job pair data for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DOMAINS = ["Software Engineering", "Data Science", "Product Design",
           "Finance & Analytics", "Marketing", "Operations"]

SKILL_POOLS = {
    "Software Engineering": ["Python","Java","Go","React","Node","SQL","Docker","K8s","AWS","FastAPI"],
    "Data Science":         ["Python","R","SQL","PyTorch","TensorFlow","Spark","dbt","Statistics","NLP","MLflow"],
    "Product Design":       ["Figma","UX Research","Prototyping","Design Systems","CSS","Motion","Accessibility"],
    "Finance & Analytics":  ["Excel","SQL","Python","Tableau","DCF","Risk Models","VBA","Power BI","Accounting"],
    "Marketing":            ["SEO","Google Ads","Meta Ads","Content","Analytics","CRM","Email","Brand Strategy"],
    "Operations":           ["Project Mgmt","Lean","SQL","Excel","Logistics","Supply Chain","ERP","Process Design"],
}


def generate_candidate(cid: int) -> dict:
    domain = np.random.choice(DOMAINS)
    pool = SKILL_POOLS[domain]
    n_skills = np.random.randint(3, len(pool))
    skills = list(np.random.choice(pool, n_skills, replace=False))
    yoe = np.random.choice([0,1,2,3,4,5,6,7,8], p=[0.08,0.12,0.15,0.15,0.14,0.13,0.10,0.08,0.05])
    verified_score = np.clip(np.random.normal(72, 14), 30, 100)
    salary_expectation = 400000 + yoe * 80000 + np.random.normal(0, 50000)
    return {
        "candidate_id": f"C{cid:04d}",
        "domain": domain,
        "skills": skills,
        "years_of_experience": yoe,
        "verified_avg_score": round(verified_score, 1),
        "city_tier": np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25]),
        "salary_expectation": round(salary_expectation),
        "profile_text": f"{yoe} years in {domain}. Key skills: {', '.join(skills)}.",
    }


def generate_job(jid: int) -> dict:
    domain = np.random.choice(DOMAINS)
    pool = SKILL_POOLS[domain]
    n_req = np.random.randint(2, min(5, len(pool)))
    required_skills = list(np.random.choice(pool, n_req, replace=False))
    min_yoe = np.random.choice([0, 1, 2, 3, 5], p=[0.10, 0.20, 0.30, 0.25, 0.15])
    salary_min = 350000 + min_yoe * 80000
    salary_max = salary_min + np.random.choice([150000, 200000, 300000])
    return {
        "job_id": f"J{jid:04d}",
        "domain": domain,
        "required_skills": required_skills,
        "min_years_experience": min_yoe,
        "city_tier_accepted": np.random.choice([1, 2, 3], p=[0.40, 0.40, 0.20]),
        "salary_min": round(salary_min),
        "salary_max": round(salary_max),
        "jd_text": f"Looking for {min_yoe}+ years in {domain}. Must have: {', '.join(required_skills)}.",
    }


def compute_label(cand: dict, job: dict) -> int:
    """
    Ground-truth label: 1 = good match, 0 = poor match.
    Based on rule-based heuristic simulating human recruiter decisions.
    """
    # Skill overlap
    cand_skills = set(cand["skills"])
    req_skills  = set(job["required_skills"])
    overlap = len(cand_skills & req_skills) / max(len(req_skills), 1)

    # Experience
    exp_ok = cand["years_of_experience"] >= job["min_years_experience"]

    # Domain match
    domain_ok = cand["domain"] == job["domain"]

    # Score quality
    score_ok = cand["verified_avg_score"] >= 60

    # Salary fit
    salary_ok = cand["salary_expectation"] <= job["salary_max"] * 1.10

    # Composite rule
    match_score = (
        overlap * 0.45
        + int(exp_ok) * 0.25
        + int(domain_ok) * 0.15
        + int(score_ok) * 0.10
        + int(salary_ok) * 0.05
    )
    # Add noise so it's not perfectly deterministic
    match_score += np.random.normal(0, 0.05)
    return int(match_score >= 0.55)


def build_feature_row(cand: dict, job: dict) -> dict:
    """Compute F1–F5 features for a candidate–job pair."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # F1 — Skill overlap (Jaccard)
    cand_skills = set(cand["skills"])
    req_skills  = set(job["required_skills"])
    union = cand_skills | req_skills
    skill_overlap = len(cand_skills & req_skills) / max(len(union), 1)

    # F2 — Experience match (delta clipped at ±5)
    exp_delta = np.clip(cand["years_of_experience"] - job["min_years_experience"], -5, 5)
    exp_match_norm = (exp_delta + 5) / 10  # [0,1]

    # F3 — Semantic similarity (TF-IDF cosine)
    try:
        vec = TfidfVectorizer(max_features=50)
        tfidf = vec.fit_transform([cand["profile_text"], job["jd_text"]])
        semantic_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        semantic_sim = 0.0

    # F4 — Verified score normalised [0,1]
    verified_score_norm = cand["verified_avg_score"] / 100.0

    # F5 — Location + salary match
    location_match = int(cand["city_tier"] <= job["city_tier_accepted"])
    salary_in_band = int(job["salary_min"] <= cand["salary_expectation"] <= job["salary_max"] * 1.10)
    loc_salary = (location_match + salary_in_band) / 2.0

    # Domain match (bonus binary feature)
    domain_match = int(cand["domain"] == job["domain"])

    label = compute_label(cand, job)

    return {
        "candidate_id":       cand["candidate_id"],
        "job_id":             job["job_id"],
        "skill_overlap":      round(skill_overlap, 4),
        "exp_match_norm":     round(exp_match_norm, 4),
        "semantic_similarity":round(semantic_sim, 4),
        "verified_score_norm":round(verified_score_norm, 4),
        "loc_salary_match":   round(loc_salary, 4),
        "domain_match":       domain_match,
        "label":              label,
    }


def generate_dataset(n_candidates: int = 600, n_jobs: int = 200,
                     pairs_per_candidate: int = 4) -> pd.DataFrame:
    print(f"Generating {n_candidates} candidates and {n_jobs} jobs...")
    candidates = [generate_candidate(i) for i in range(1, n_candidates + 1)]
    jobs       = [generate_job(j) for j in range(1, n_jobs + 1)]

    rows = []
    for cand in candidates:
        sampled_jobs = np.random.choice(jobs, pairs_per_candidate, replace=False)
        for job in sampled_jobs:
            rows.append(build_feature_row(cand, job))

    df = pd.DataFrame(rows)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).parent / "sample_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
