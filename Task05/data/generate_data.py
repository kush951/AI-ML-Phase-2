"""
PlaceMux · Data Generator
Generates realistic synthetic students and jobs for matching validation.
"""
import json
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ─── Skill taxonomy ────────────────────────────────────────────────────────────
SKILL_DOMAINS = {
    "software_engineering": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust",
        "React", "Node.js", "Django", "FastAPI", "Spring Boot", "Docker",
        "Kubernetes", "AWS", "GCP", "Azure", "PostgreSQL", "MongoDB", "Redis",
        "Git", "CI/CD", "REST APIs", "GraphQL", "Microservices"
    ],
    "data_science": [
        "Python", "R", "SQL", "Machine Learning", "Deep Learning", "NLP",
        "Computer Vision", "TensorFlow", "PyTorch", "scikit-learn", "Pandas",
        "NumPy", "Tableau", "Power BI", "Spark", "Hadoop", "Statistics",
        "A/B Testing", "Feature Engineering", "Model Deployment"
    ],
    "product_management": [
        "Product Strategy", "Roadmapping", "User Research", "Agile", "Scrum",
        "JIRA", "Figma", "Data Analysis", "A/B Testing", "OKRs",
        "Stakeholder Management", "Market Research", "SQL", "Wireframing",
        "Customer Development", "Go-to-Market", "Competitive Analysis"
    ],
    "design": [
        "Figma", "Adobe XD", "Sketch", "UI Design", "UX Research",
        "Prototyping", "Wireframing", "User Testing", "Design Systems",
        "Accessibility", "Motion Design", "After Effects", "Illustrator",
        "Photoshop", "CSS", "HTML", "Branding", "Typography"
    ],
    "marketing": [
        "SEO", "SEM", "Google Analytics", "Meta Ads", "Content Marketing",
        "Email Marketing", "HubSpot", "Salesforce", "Copywriting",
        "Brand Strategy", "Social Media", "Campaign Management",
        "Marketing Automation", "CRM", "Data Analysis", "SQL"
    ]
}

ALL_SKILLS = list(set(s for skills in SKILL_DOMAINS.values() for s in skills))
DOMAINS = list(SKILL_DOMAINS.keys())

EDUCATION_LEVELS = ["High School", "Associate", "Bachelor", "Master", "PhD"]
CITIES = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune", "Chennai",
          "Kolkata", "Ahmedabad", "Remote"]
SENIORITY = ["Intern", "Junior", "Mid", "Senior", "Lead"]

JOB_TITLES = {
    "software_engineering": ["Software Engineer", "Backend Developer", "Full Stack Developer",
                              "DevOps Engineer", "Cloud Engineer", "SRE"],
    "data_science": ["Data Scientist", "ML Engineer", "Data Analyst", "AI Researcher",
                     "Data Engineer", "MLOps Engineer"],
    "product_management": ["Product Manager", "Associate PM", "Senior PM", "Product Lead"],
    "design": ["UI Designer", "UX Designer", "Product Designer", "Visual Designer"],
    "marketing": ["Marketing Manager", "Growth Manager", "Content Strategist", "SEO Specialist"]
}


def pick_skills(domain: str, n_skills: int) -> list[str]:
    pool = SKILL_DOMAINS[domain] + random.sample(ALL_SKILLS, min(5, len(ALL_SKILLS)))
    return random.sample(list(set(pool)), min(n_skills, len(set(pool))))


def generate_student(student_id: int) -> dict:
    domain = random.choice(DOMAINS)
    level_idx = random.randint(0, 4)
    exp_years = max(0, level_idx * 1.5 + random.gauss(0, 0.5))
    edu_idx = max(2, min(4, level_idx + random.randint(-1, 1)))
    n_skills = random.randint(4, 14)

    # Verified skill scores (0-100, simulating a skill assessment)
    skills_raw = pick_skills(domain, n_skills)
    verified_skills = {
        skill: round(np.clip(random.gauss(65 + level_idx * 5, 15), 20, 99), 1)
        for skill in skills_raw
    }

    return {
        "id": f"STU{student_id:04d}",
        "name": f"Student_{student_id}",
        "domain": domain,
        "education_level": EDUCATION_LEVELS[edu_idx],
        "education_level_idx": edu_idx,
        "seniority": SENIORITY[level_idx],
        "seniority_idx": level_idx,
        "experience_years": round(exp_years, 1),
        "verified_skills": verified_skills,
        "skill_list": skills_raw,
        "city": random.choice(CITIES),
        "salary_expectation_lpa": round(max(3, exp_years * 2 + random.gauss(4, 2)), 1),
        "open_to_remote": random.random() > 0.3,
        "bio": f"Experienced {SENIORITY[level_idx]} professional in {domain.replace('_', ' ')} "
               f"with {round(exp_years, 1)} years of experience. "
               f"Skilled in {', '.join(skills_raw[:3])}."
    }


def generate_job(job_id: int) -> dict:
    domain = random.choice(DOMAINS)
    level_idx = random.randint(0, 4)
    n_required = random.randint(3, 8)
    n_preferred = random.randint(2, 5)

    required_skills = pick_skills(domain, n_required)
    preferred_skills = [s for s in pick_skills(domain, n_preferred + 2)
                        if s not in required_skills][:n_preferred]

    min_skill_threshold = random.randint(55, 75)  # minimum score to qualify

    return {
        "id": f"JOB{job_id:04d}",
        "title": random.choice(JOB_TITLES[domain]),
        "company": f"Company_{job_id}",
        "domain": domain,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "min_skill_threshold": min_skill_threshold,
        "min_experience_years": max(0, level_idx * 1.2 + random.gauss(0, 0.3)),
        "seniority_required": SENIORITY[level_idx],
        "seniority_required_idx": level_idx,
        "education_required": EDUCATION_LEVELS[max(2, edu_idx)]
            if (edu_idx := max(2, min(4, level_idx + random.randint(-1, 0)))) else "Bachelor",
        "education_required_idx": max(2, min(4, level_idx + random.randint(-1, 0))),
        "city": random.choice(CITIES),
        "remote_ok": random.random() > 0.4,
        "salary_budget_lpa": round((level_idx * 3 + 6) + random.gauss(0, 2), 1),
        "description": f"We are looking for a {SENIORITY[level_idx]} {domain.replace('_', ' ')} "
                       f"professional. Required: {', '.join(required_skills[:3])}. "
                       f"Nice to have: {', '.join(preferred_skills[:2]) if preferred_skills else 'N/A'}."
    }


def create_label(student: dict, job: dict) -> int:
    """Deterministic ground-truth label for training (1=good match, 0=not)."""
    score = 0.0

    # 1. Domain match (binary)
    if student["domain"] == job["domain"]:
        score += 0.30

    # 2. Skill match ratio
    req = set(job["required_skills"])
    stu = set(student["skill_list"])
    matched = req & stu

    if req:
        ratio = len(matched) / len(req)
        # Check skill thresholds
        qualified = sum(
            1 for s in matched
            if student["verified_skills"].get(s, 0) >= job["min_skill_threshold"]
        )
        score += 0.30 * (qualified / len(req))

    # 3. Experience match
    exp_gap = student["experience_years"] - job["min_experience_years"]
    if exp_gap >= 0:
        score += 0.15 * min(1.0, 1.0 - exp_gap / 10)
    else:
        score += 0.15 * max(0, 1.0 + exp_gap / 3)  # penalty for under-experience

    # 4. Education
    if student["education_level_idx"] >= job["education_required_idx"]:
        score += 0.10

    # 5. Location / remote
    if student["city"] == job["city"] or job["remote_ok"] or student["open_to_remote"]:
        score += 0.08

    # 6. Salary fit
    if student["salary_expectation_lpa"] <= job["salary_budget_lpa"] * 1.1:
        score += 0.07

    # Label: positive if composite score > 0.55 (with some noise)
    threshold = 0.55 + random.gauss(0, 0.03)
    return int(score >= threshold), round(score, 3)


def generate_dataset(n_students=300, n_jobs=100):
    students = [generate_student(i) for i in range(n_students)]
    jobs = [generate_job(j) for j in range(n_jobs)]

    # Create pairs (each student matched against a sample of jobs)
    pairs = []
    for student in students:
        sampled_jobs = random.sample(jobs, min(10, len(jobs)))
        for job in sampled_jobs:
            label, raw_score = create_label(student, job)
            pairs.append({
                "student_id": student["id"],
                "job_id": job["id"],
                "label": label,
                "raw_score": raw_score
            })

    print(f"Generated {len(students)} students, {len(jobs)} jobs, {len(pairs)} pairs")
    print(f"Positive rate: {sum(p['label'] for p in pairs) / len(pairs):.2%}")

    return {
        "students": students,
        "jobs": jobs,
        "pairs": pairs
    }


if __name__ == "__main__":
    data = generate_dataset()
    out = Path(__file__).parent
    with open(out / "dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved → data/dataset.json")
