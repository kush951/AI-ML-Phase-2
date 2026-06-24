import joblib
import numpy as np
from pathlib import Path

SKILLS = [
    "python", "sql", "machine_learning", "data_analysis", "deep_learning",
    "javascript", "react", "node_js", "java", "aws", "docker", "kubernetes",
    "communication", "teamwork", "problem_solving", "project_management",
    "excel", "tableau", "spark", "statistics",
]

MODEL_DIR = Path("")

class PlaceMuxMatcher:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "best_model.pkl")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

    def predict(self, student, job):

        score = (
            student.get("skill_python", 0) * 0.3 +
            student.get("skill_machine_learning", 0) * 0.4 +
            student.get("skill_sql", 0) * 0.3
        )

        probability = round(score * 100, 2)

        return {
            "match_probability": probability,
            "match": probability >= 60,
            "explanation": {
                "strengths": ["Python", "Machine Learning"],
                "improvement_areas": ["Docker", "AWS"]
            }
        }

    def rank_jobs_for_student(self, student, jobs, top_k=5):

        ranked = []

        for job in jobs:
            result = self.predict(student, job)

            ranked.append({
                "job_id": job["job_id"],
                "title": job["title"],
                "salary_lpa": job["salary_lpa"],
                "match_probability": result["match_probability"]
            })

        ranked.sort(
            key=lambda x: x["match_probability"],
            reverse=True
        )

        return ranked[:top_k]