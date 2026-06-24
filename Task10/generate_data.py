"""
PlaceMux · Data Generator
Produces realistic student-job matching dataset with verified skill scores.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

SKILLS = [
    "python", "sql", "machine_learning", "data_analysis", "deep_learning",
    "javascript", "react", "node_js", "java", "aws", "docker", "kubernetes",
    "communication", "teamwork", "problem_solving", "project_management",
    "excel", "tableau", "spark", "statistics",
]

JOB_PROFILES = {
    "Data Scientist":       {"python":0.9,"sql":0.7,"machine_learning":0.9,"statistics":0.8,"deep_learning":0.6,"data_analysis":0.9,"communication":0.7},
    "ML Engineer":          {"python":0.9,"machine_learning":0.9,"deep_learning":0.8,"aws":0.7,"docker":0.7,"statistics":0.7,"spark":0.6},
    "Frontend Developer":   {"javascript":0.9,"react":0.9,"communication":0.7,"problem_solving":0.7,"node_js":0.5},
    "Backend Developer":    {"java":0.8,"python":0.7,"sql":0.8,"docker":0.7,"aws":0.6,"node_js":0.7},
    "Data Analyst":         {"sql":0.9,"excel":0.8,"tableau":0.7,"data_analysis":0.9,"statistics":0.7,"communication":0.8,"python":0.6},
    "DevOps Engineer":      {"docker":0.9,"kubernetes":0.9,"aws":0.9,"python":0.6,"problem_solving":0.8},
    "Full Stack Developer": {"javascript":0.8,"react":0.7,"node_js":0.8,"sql":0.7,"python":0.6,"docker":0.6},
    "Project Manager":      {"project_management":0.95,"communication":0.9,"teamwork":0.9,"problem_solving":0.8,"excel":0.7},
}

def _generate_student_skills(dominant_role: str) -> dict:
    profile = JOB_PROFILES[dominant_role]
    scores = {}
    for skill in SKILLS:
        base = profile.get(skill, 0.2)
        score = float(np.clip(rng.normal(base, 0.15), 0.0, 1.0))
        scores[skill] = round(score, 3)
    return scores

def _compute_true_match(student_skills: dict, job_role: str) -> float:
    profile = JOB_PROFILES[job_role]
    weighted_sum, weight_total = 0.0, 0.0
    for skill, req in profile.items():
        student_score = student_skills.get(skill, 0.0)
        gap = max(0.0, req - student_score)
        weight = req
        weighted_sum += weight * (1.0 - gap / req if req > 0 else 0)
        weight_total += weight
    return round(weighted_sum / weight_total, 4) if weight_total > 0 else 0.0

def generate_dataset(n_students=800, n_jobs=200, seed=42):
    job_roles_list = list(JOB_PROFILES.keys())

    # --- Jobs ---
    jobs = []
    for i in range(n_jobs):
        role = rng.choice(job_roles_list)
        jobs.append({
            "job_id": f"JOB_{i+1:04d}",
            "title": role,
            "company": f"Company_{rng.integers(1, 60):02d}",
            "salary_lpa": round(float(rng.uniform(5, 35)), 1),
            **{f"req_{s}": round(float(JOB_PROFILES[role].get(s, rng.uniform(0, 0.3))), 3) for s in SKILLS},
        })
    jobs_df = pd.DataFrame(jobs)

    # --- Students ---
    students = []
    dominant = rng.choice(job_roles_list, size=n_students)
    for i, dom in enumerate(dominant):
        skills = _generate_student_skills(dom)
        students.append({
            "student_id": f"STU_{i+1:04d}",
            "dominant_background": dom,
            "years_experience": round(float(rng.uniform(0, 8)), 1),
            "gpa": round(float(np.clip(rng.normal(3.2, 0.4), 2.0, 4.0)), 2),
            **{f"skill_{s}": v for s, v in skills.items()},
        })
    students_df = pd.DataFrame(students)

    # --- Pairs (matching dataset) ---
    pairs = []
    for _, student in students_df.iterrows():
        student_skills = {s: student[f"skill_{s}"] for s in SKILLS}
        # Each student gets 5 random job pairs (mixed relevant + irrelevant)
        sampled_jobs = jobs_df.sample(5, random_state=int(student["student_id"].split("_")[1]))
        for _, job in sampled_jobs.iterrows():
            job_req = {s: job[f"req_{s}"] for s in SKILLS}
            true_score = _compute_true_match(student_skills, job["title"])
            # skill_overlap: fraction of required skills where student ≥ 0.6
            overlap = sum(1 for s, req in job_req.items() if req >= 0.5 and student_skills.get(s, 0) >= 0.6)
            total_req = sum(1 for req in job_req.values() if req >= 0.5)
            skill_overlap = overlap / total_req if total_req > 0 else 0.0
            # euclidean distance between skill vectors (normalised)
            dist = float(np.sqrt(sum((student_skills[s] - job_req[s])**2 for s in SKILLS)) / len(SKILLS))
            # label: 1 if strong match
            label = 1 if true_score >= 0.70 else 0
            pairs.append({
                "student_id": student["student_id"],
                "job_id": job["job_id"],
                "job_title": job["title"],
                "true_match_score": true_score,
                "skill_overlap_ratio": round(skill_overlap, 4),
                "skill_vector_distance": round(dist, 4),
                "student_experience": student["years_experience"],
                "student_gpa": student["gpa"],
                "job_salary_lpa": job["salary_lpa"],
                **{f"gap_{s}": round(max(0.0, job[f"req_{s}"] - student[f"skill_{s}"]), 3) for s in SKILLS},
                "label": label,
            })
    pairs_df = pd.DataFrame(pairs)
    return students_df, jobs_df, pairs_df


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    students_df, jobs_df, pairs_df = generate_dataset()
    students_df.to_csv(out / "students.csv", index=False)
    jobs_df.to_csv(out / "jobs.csv", index=False)
    pairs_df.to_csv(out / "matching_pairs.csv", index=False)
    print(f"Students: {len(students_df)}")
    print(f"Jobs    : {len(jobs_df)}")
    print(f"Pairs   : {len(pairs_df)}")
    print(f"Positive labels: {pairs_df['label'].mean():.1%}")
    print("Data saved to /home/claude/placemux/data/")
