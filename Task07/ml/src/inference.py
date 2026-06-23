"""
PlaceMux · AI/ML · Inference & Explainability Engine
Provides match scoring, ranking, and plain-English explanations.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import build_features

MODELS_DIR = Path(__file__).parent.parent / "models"

SKILLS = [
    "Python", "JavaScript", "React", "SQL", "Machine Learning",
    "Data Analysis", "Communication", "Leadership", "Project Management",
    "Node.js", "AWS", "Docker", "Java", "C++", "Product Design",
    "Excel", "Tableau", "Statistics", "NLP", "Deep Learning",
    "Git", "Agile", "Problem Solving", "FastAPI", "MongoDB"
]


class PlaceMuxMatcher:
    """Rank jobs for a student and explain each match in plain English."""

    def __init__(self, model_path=None):
        path = model_path or MODELS_DIR / "best_model.pkl"
        self.model = joblib.load(path)

    def score_pair(self, student: Dict, job: Dict) -> Dict[str, Any]:
        """Score a single student-job pair and return explanation."""
        row = {**student, **job}
        df_row = pd.DataFrame([row])
        features = build_features(df_row)
        prob = self.model.predict_proba(features)[0][1]
        pred = int(prob >= 0.5)

        explanation = self._explain(student, job, features.iloc[0])
        return {
            "match_probability": round(float(prob), 4),
            "match_label": pred,
            "match_grade": _grade(prob),
            "explanation": explanation,
            "feature_snapshot": features.iloc[0].to_dict(),
        }

    def rank_jobs(self, student: Dict, jobs: List[Dict]) -> List[Dict]:
        """Rank all jobs for a student by match probability."""
        scored = []
        for job in jobs:
            result = self.score_pair(student, job)
            scored.append({
                "job_id": job.get("job_id", "?"),
                "role": job.get("role", "Unknown"),
                **result
            })
        return sorted(scored, key=lambda x: x["match_probability"], reverse=True)

    def _explain(self, student: Dict, job: Dict, feats: pd.Series) -> Dict:
        """Generate structured, plain-English explanation for a match."""
        reasons_for = []
        reasons_against = []
        skill_details = []

        for skill in SKILLS:
            col = skill.replace(' ', '_').lower()
            stu_score = student.get(f"skill_{col}", 0.0)
            req_score = job.get(f"req_skill_{col}", 0.0)
            if req_score > 0:
                ratio = stu_score / (req_score + 1e-9)
                skill_details.append({
                    "skill": skill,
                    "student_score": round(stu_score, 1),
                    "required_score": round(req_score, 1),
                    "met": stu_score >= req_score * 0.75
                })
                if stu_score >= req_score:
                    reasons_for.append(f"Strong {skill} ({stu_score:.0f} ≥ {req_score:.0f} required)")
                elif stu_score >= req_score * 0.75:
                    reasons_for.append(f"Adequate {skill} ({stu_score:.0f}, ≥75% of {req_score:.0f} needed)")
                else:
                    reasons_against.append(f"Skill gap in {skill} ({stu_score:.0f} vs {req_score:.0f} required)")

        # Experience
        exp_gap = student.get("years_experience", 0) - job.get("min_experience", 0)
        if exp_gap >= 0:
            reasons_for.append(f"Experience OK ({student['years_experience']:.1f} yrs ≥ {job['min_experience']:.1f} required)")
        else:
            reasons_against.append(f"Under-experienced by {abs(exp_gap):.1f} years")

        # CGPA
        cgpa_gap = student.get("cgpa", 0) - job.get("min_cgpa", 0)
        if cgpa_gap >= 0:
            reasons_for.append(f"CGPA meets threshold ({student['cgpa']:.2f} ≥ {job['min_cgpa']:.1f})")
        else:
            reasons_against.append(f"CGPA {student['cgpa']:.2f} below minimum {job['min_cgpa']:.1f}")

        coverage = feats.get("skill_coverage_hard", 0)
        summary = (
            f"This student covers {coverage:.0%} of the required skills for the {job.get('role', 'role')}. "
            f"{'Overall a strong candidate.' if coverage >= 0.7 else 'Some skill gaps to address.'}"
        )

        return {
            "summary": summary,
            "skill_coverage": f"{coverage:.1%}",
            "skill_details": skill_details,
            "strengths": reasons_for[:5],
            "gaps": reasons_against[:5],
        }


def _grade(prob: float) -> str:
    if prob >= 0.85: return "Excellent"
    if prob >= 0.70: return "Strong"
    if prob >= 0.55: return "Moderate"
    if prob >= 0.40: return "Weak"
    return "Poor"


# ── Demo walkthrough ──────────────────────────────────────────────────────────
def demo_walkthrough():
    """One-example demo: this student, this job, and why it's a match."""
    matcher = PlaceMuxMatcher()

    student = {
        "student_id": "STU0001",
        "years_experience": 2.0,
        "cgpa": 8.1,
        "skill_python": 85.0, "skill_machine_learning": 78.0,
        "skill_statistics": 72.0, "skill_sql": 65.0,
        "skill_deep_learning": 60.0, "skill_data_analysis": 70.0,
        **{f"skill_{s.replace(' ', '_').lower()}": 0.0
           for s in ["JavaScript", "React", "Node.js", "AWS", "Docker",
                     "Java", "C++", "Product Design", "Excel", "Tableau",
                     "NLP", "Git", "Agile", "Problem Solving", "FastAPI",
                     "MongoDB", "Communication", "Leadership",
                     "Project Management"]}
    }

    job = {
        "job_id": "JOB0001",
        "role": "Data Scientist",
        "min_experience": 1.0,
        "min_cgpa": 7.0,
        "req_skill_python": 80.0, "req_skill_machine_learning": 75.0,
        "req_skill_statistics": 70.0, "req_skill_sql": 60.0,
        "req_skill_deep_learning": 65.0,
        **{f"req_skill_{s.replace(' ', '_').lower()}": 0.0
           for s in ["JavaScript", "React", "Data Analysis", "Communication",
                     "Leadership", "Project Management", "Node.js", "AWS",
                     "Docker", "Java", "C++", "Product Design", "Excel",
                     "Tableau", "NLP", "Git", "Agile", "Problem Solving",
                     "FastAPI", "MongoDB"]},
        **{f"opt_skill_{s.replace(' ', '_').lower()}": 0.0 for s in SKILLS}
    }

    result = matcher.score_pair(student, job)
    print("=" * 60)
    print("DEMO WALKTHROUGH")
    print("=" * 60)
    print(f"Student: {student['student_id']}  |  Job: {job['role']} ({job['job_id']})")
    print(f"Match Probability : {result['match_probability']:.1%}")
    print(f"Match Grade       : {result['match_grade']}")
    print(f"\nSummary: {result['explanation']['summary']}")
    print(f"\nStrengths:")
    for s in result["explanation"]["strengths"]:
        print(f"  ✓ {s}")
    print(f"\nGaps:")
    for g in result["explanation"]["gaps"]:
        print(f"  ✗ {g}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    demo_walkthrough()
