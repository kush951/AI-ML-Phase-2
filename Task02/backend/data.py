"""
Sample data generation for PlaceMux Task 2 (Job Posting with Skill Thresholds).

Generates a real-shaped (if small) sample dataset of:
  - SKILLS: the taxonomy of verifiable skills (each scored L1-L100)
  - STUDENTS: verified skill scores per student (as if produced by upstream
    skill-verification, which is out of scope for this task)
  - SAMPLE_JOBS: a few example job postings with L1-L100 thresholds

This stands in for the "Backend job-threshold data" upstream dependency
mentioned in the study guide -- in production this would come from the
jobs-service API instead of being generated here.
"""
import random
import json
from pathlib import Path

random.seed(42)

SKILLS = [
    "Python", "SQL", "Data Structures", "Machine Learning",
    "Statistics", "REST APIs", "Git", "Communication",
    "System Design", "Cloud (AWS/GCP)", "Testing", "Deep Learning",
]

FIRST_NAMES = ["Asha", "Rohit", "Meera", "Karan", "Divya", "Aditya", "Priya",
               "Vikram", "Sneha", "Arjun", "Neha", "Rahul", "Pooja", "Sahil",
               "Anjali", "Manish", "Riya", "Suresh", "Kavya", "Imran",
               "Tanvi", "Yash", "Nisha", "Varun", "Shreya"]

def _skewed_level(mean: int, spread: int = 20) -> int:
    """A student's level for a skill, clipped to the L1-L100 range."""
    lvl = int(random.gauss(mean, spread))
    return max(1, min(100, lvl))


def generate_students(n: int = 25):
    students = []
    for i in range(n):
        # Each student has 1-3 "strong" skills (high mean) and is weaker
        # elsewhere -- mirrors real resumes, nobody is a 90 at everything.
        strong_skills = random.sample(SKILLS, k=random.randint(1, 3))
        scores = {}
        for skill in SKILLS:
            mean = random.randint(60, 90) if skill in strong_skills else random.randint(15, 60)
            scores[skill] = _skewed_level(mean)
        students.append({
            "id": f"S{i+1:03d}",
            "name": f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {chr(65 + i % 26)}.",
            "verified_skill_scores": scores,
        })
    return students


def generate_sample_jobs():
    """A few hand-shaped jobs with realistic L1-L100 gating thresholds."""
    return [
        {
            "id": "J001",
            "title": "Junior ML Engineer",
            "company": "Altrodav Technologies",
            "thresholds": {
                "Python": 60, "Machine Learning": 55, "Statistics": 50,
                "Git": 40, "Communication": 30,
            },
        },
        {
            "id": "J002",
            "title": "Backend API Developer",
            "company": "Quickloop Systems",
            "thresholds": {
                "Python": 50, "REST APIs": 60, "SQL": 55,
                "System Design": 40, "Testing": 35,
            },
        },
        {
            "id": "J003",
            "title": "Senior Data Scientist",
            "company": "Northbridge Analytics",
            "thresholds": {
                "Python": 75, "Machine Learning": 80, "Deep Learning": 65,
                "Statistics": 75, "Cloud (AWS/GCP)": 50,
            },
        },
    ]


def save_sample_data(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    students = generate_students()
    jobs = generate_sample_jobs()
    (out / "students.json").write_text(json.dumps(students, indent=2))
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
    return students, jobs


if __name__ == "__main__":
    s, j = save_sample_data("../data")
    print(f"Generated {len(s)} students and {len(j)} sample jobs.")
