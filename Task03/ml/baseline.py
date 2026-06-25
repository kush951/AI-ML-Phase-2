"""
PlaceMux - Baseline ranker
============================
The "dumb" baseline described in the study guide: rank purely by raw overlap
of required vs verified skills (unweighted, no experience, no learned weights).
Every later models's numbers are only meaningful relative to this.
"""
import numpy as np


def baseline_score(student_skills: dict, job_requirements: list) -> float:
    """Unweighted fraction of required skills the student meets, 0-1."""
    if not job_requirements:
        return 0.0
    met = sum(1 for skill, _, min_req in job_requirements if student_skills.get(skill, 0.0) >= min_req)
    return met / len(job_requirements)
