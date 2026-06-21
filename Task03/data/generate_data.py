"""
PlaceMux - Sample data generator
=================================
Generates a real-shaped (synthetic) dataset for the Search & Discovery task:
  - skills.csv          : master skill taxonomy
  - students.csv         : students with verified skill scores (0-100), experience, branch
  - companies.csv         : hiring companies
  - jobs.csv              : job postings with required skills + min experience
  - applications.csv      : student x job pairs with a ground-truth label (good_match)
                            derived from a hidden "true" affinity function + realistic noise,
                            used to TRAIN and EVALUATE the ranking models.

This is not a toy/happy-path set: scores are noisy, some students have sparse skill
profiles, some jobs have unrealistic requirements, and label noise is injected so no
model can hit 100% - exactly like production data.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
OUT_DIR = os.path.dirname(__file__)

SKILL_POOL = [
    "Python", "SQL", "Java", "JavaScript", "React", "Node.js", "Pandas", "NumPy",
    "scikit-learn", "PyTorch", "TensorFlow", "Excel", "PowerBI", "Tableau",
    "AWS", "Docker", "Kubernetes", "Git", "REST APIs", "FastAPI", "Django",
    "C++", "Go", "Spring Boot", "HTML/CSS", "Linux", "Spark", "Airflow",
    "Statistics", "A/B Testing", "Communication", "Figma", "DSA",
]

BRANCHES = ["CSE", "IT", "ECE", "Mech", "Civil", "EEE", "AIML", "Data Science"]
JOB_CATEGORIES = ["Data Analyst", "Backend Engineer", "Frontend Engineer",
                   "ML Engineer", "DevOps", "Full Stack", "QA/Test", "Product Analyst"]

N_STUDENTS = 420
N_COMPANIES = 60
N_JOBS = 140

# ---------------------------------------------------------------------------
# 1. Skills master table
# ---------------------------------------------------------------------------
skills_df = pd.DataFrame({"skill_id": range(1, len(SKILL_POOL) + 1), "skill_name": SKILL_POOL})
skills_df.to_csv(f"{OUT_DIR}/skills.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Students: each gets a sparse, noisy set of *verified* skill scores
# ---------------------------------------------------------------------------
student_rows = []
student_skill_rows = []
for sid in range(1, N_STUDENTS + 1):
    branch = RNG.choice(BRANCHES)
    experience_years = float(np.clip(RNG.exponential(1.1), 0, 4).round(1))
    n_skills = RNG.integers(3, 11)  # sparse profile, like real students
    chosen = RNG.choice(SKILL_POOL, size=n_skills, replace=False)
    for sk in chosen:
        # verified skill score: noisy, not a clean 0-100
        base = RNG.normal(65, 18)
        score = float(np.clip(base, 5, 100).round(1))
        student_skill_rows.append({"student_id": sid, "skill_name": sk, "verified_score": score})
    student_rows.append({
        "student_id": sid,
        "name": f"Student_{sid:04d}",
        "branch": branch,
        "experience_years": experience_years,
        "graduation_year": int(RNG.choice([2024, 2025, 2026, 2027])),
    })

students_df = pd.DataFrame(student_rows)
student_skills_df = pd.DataFrame(student_skill_rows)
students_df.to_csv(f"{OUT_DIR}/students.csv", index=False)
student_skills_df.to_csv(f"{OUT_DIR}/student_skills.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Companies & Jobs
# ---------------------------------------------------------------------------
companies_df = pd.DataFrame({
    "company_id": range(1, N_COMPANIES + 1),
    "company_name": [f"Company_{i:03d}" for i in range(1, N_COMPANIES + 1)],
})
companies_df.to_csv(f"{OUT_DIR}/companies.csv", index=False)

job_rows = []
job_skill_rows = []
for jid in range(1, N_JOBS + 1):
    company_id = int(RNG.integers(1, N_COMPANIES + 1))
    category = RNG.choice(JOB_CATEGORIES)
    min_experience = float(np.clip(RNG.exponential(0.8), 0, 3).round(1))
    n_req = RNG.integers(3, 8)
    required_skills = RNG.choice(SKILL_POOL, size=n_req, replace=False)
    for sk in required_skills:
        weight = float(np.clip(RNG.normal(1.0, 0.3), 0.3, 1.8).round(2))  # importance weight
        min_score = int(RNG.choice([40, 50, 60, 70]))
        job_skill_rows.append({
            "job_id": jid, "skill_name": sk, "weight": weight, "min_required_score": min_score
        })
    job_rows.append({
        "job_id": jid,
        "company_id": company_id,
        "title": f"{category} (Job {jid})",
        "category": category,
        "min_experience_years": min_experience,
        "openings": int(RNG.integers(1, 6)),
    })

jobs_df = pd.DataFrame(job_rows)
job_skills_df = pd.DataFrame(job_skill_rows)
jobs_df.to_csv(f"{OUT_DIR}/jobs.csv", index=False)
job_skills_df.to_csv(f"{OUT_DIR}/job_skills.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Applications: sample student-job pairs + a hidden ground-truth affinity
#    -> "good_match" label used to TRAIN and EVALUATE ranking quality.
#    The hidden function is intentionally NOT 100% deterministic (label noise)
#    so the task can't be solved with a trivial perfect rule.
# ---------------------------------------------------------------------------
student_skill_lookup = {}
for sid, grp in student_skills_df.groupby("student_id"):
    student_skill_lookup[sid] = dict(zip(grp.skill_name, grp.verified_score))

job_skill_lookup = {}
for jid, grp in job_skills_df.groupby("job_id"):
    job_skill_lookup[jid] = list(zip(grp.skill_name, grp.weight, grp.min_required_score))

job_meta = jobs_df.set_index("job_id").to_dict("index")
student_meta = students_df.set_index("student_id").to_dict("index")

app_rows = []
N_PAIRS_PER_STUDENT = 14  # each student is shown against ~14 jobs (mix of relevant/irrelevant)

for sid in range(1, N_STUDENTS + 1):
    candidate_jobs = RNG.choice(jobs_df.job_id.values, size=N_PAIRS_PER_STUDENT, replace=False)
    s_skills = student_skill_lookup.get(sid, {})
    s_exp = student_meta[sid]["experience_years"]
    for jid in candidate_jobs:
        reqs = job_skill_lookup.get(jid, [])
        j_min_exp = job_meta[jid]["min_experience_years"]

        if reqs:
            covered = [s_skills.get(sk, 0.0) >= min_req for sk, w, min_req in reqs]
            weighted_cov = sum(
                w * (1 if s_skills.get(sk, 0.0) >= min_req else 0) for sk, w, min_req in reqs
            ) / sum(w for _, w, _ in reqs)
            avg_overlap_score = np.mean([s_skills.get(sk, 0.0) for sk, _, _ in reqs]) / 100.0
        else:
            weighted_cov = 0.0
            avg_overlap_score = 0.0

        exp_ok = 1.0 if s_exp >= j_min_exp else max(0.0, 1 - (j_min_exp - s_exp) / 3)

        # hidden true affinity (latent, not given to the model directly)
        true_affinity = 0.55 * weighted_cov + 0.25 * avg_overlap_score + 0.20 * exp_ok
        true_affinity = np.clip(true_affinity + RNG.normal(0, 0.08), 0, 1)  # label noise

        label = int(RNG.random() < true_affinity)  # stochastic, not a hard threshold

        app_rows.append({
            "student_id": sid,
            "job_id": jid,
            "good_match": label,
        })

applications_df = pd.DataFrame(app_rows)
applications_df.to_csv(f"{OUT_DIR}/applications.csv", index=False)

print("Generated:")
print(f"  students.csv          {len(students_df)} rows")
print(f"  student_skills.csv    {len(student_skills_df)} rows")
print(f"  companies.csv         {len(companies_df)} rows")
print(f"  jobs.csv              {len(jobs_df)} rows")
print(f"  job_skills.csv        {len(job_skills_df)} rows")
print(f"  applications.csv      {len(applications_df)} rows, positive rate = "
      f"{applications_df.good_match.mean():.3f}")
