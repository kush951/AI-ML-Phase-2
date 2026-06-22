"""
PlaceMux · Explainability
Generates plain-English match explanations for student–job pairs.
"""
from typing import Optional
import numpy as np


def explain_match(
    student: dict,
    job: dict,
    match_score: float,
    feature_breakdown: dict,
    feature_importances: Optional[dict] = None,
) -> dict:
    """
    Produce a structured, plain-English explanation for a match decision.

    Returns:
        {
            "headline": str,
            "verdict": "Strong Match" | "Moderate Match" | "Weak Match" | "No Match",
            "score_pct": int,
            "reasons_for": list[str],
            "reasons_against": list[str],
            "skill_detail": dict,
            "recommendation": str,
        }
    """
    reasons_for = []
    reasons_against = []
    skill_detail = {}

    # ─── Skill analysis ────────────────────────────────────────────────────
    req_skills = set(job["required_skills"])
    stu_skills = set(student["skill_list"])
    matched = req_skills & stu_skills
    missing = req_skills - stu_skills
    threshold = job["min_skill_threshold"]

    # Which matched skills are above / below threshold?
    above_threshold = [
        s for s in matched
        if student["verified_skills"].get(s, 0) >= threshold
    ]
    below_threshold = [
        s for s in matched
        if student["verified_skills"].get(s, 0) < threshold
    ]

    skill_detail["required"] = list(req_skills)
    skill_detail["matched"] = list(matched)
    skill_detail["missing"] = list(missing)
    skill_detail["qualified"] = above_threshold
    skill_detail["below_threshold"] = below_threshold
    skill_detail["threshold"] = threshold
    skill_detail["scores"] = {
        s: student["verified_skills"].get(s, 0) for s in matched
    }

    if above_threshold:
        pct = round(len(above_threshold) / len(req_skills) * 100)
        reasons_for.append(
            f"Meets {pct}% of required skills above threshold "
            f"({', '.join(above_threshold[:3])}{'...' if len(above_threshold) > 3 else ''})"
        )
    if below_threshold:
        reasons_against.append(
            f"Verified score below {threshold} for: {', '.join(below_threshold[:3])}"
        )
    if missing:
        reasons_against.append(
            f"Missing required skills: {', '.join(list(missing)[:3])}{'...' if len(missing) > 3 else ''}"
        )

    # ─── Preferred skills ──────────────────────────────────────────────────
    pref = set(job.get("preferred_skills", []))
    pref_matched = pref & stu_skills
    if pref_matched:
        reasons_for.append(
            f"Has {len(pref_matched)} preferred skill(s): "
            f"{', '.join(list(pref_matched)[:2])}"
        )

    # ─── Experience ────────────────────────────────────────────────────────
    exp_gap = student["experience_years"] - job["min_experience_years"]
    if exp_gap >= 0:
        if exp_gap < 3:
            reasons_for.append(
                f"Experience closely matches: "
                f"{student['experience_years']}y vs {job['min_experience_years']}y required"
            )
        else:
            reasons_for.append(
                f"Exceeds experience requirement "
                f"({student['experience_years']}y vs {job['min_experience_years']}y required)"
            )
    else:
        reasons_against.append(
            f"Under-experienced by {abs(round(exp_gap, 1))} years "
            f"({student['experience_years']}y vs {job['min_experience_years']}y required)"
        )

    # ─── Seniority ─────────────────────────────────────────────────────────
    if student["seniority"] == job["seniority_required"]:
        reasons_for.append(
            f"Seniority level matches: {student['seniority']}"
        )
    elif student["seniority_idx"] > job["seniority_required_idx"]:
        reasons_for.append(
            f"Senior than required ({student['seniority']} vs {job['seniority_required']})"
        )
    else:
        reasons_against.append(
            f"Seniority gap: student is {student['seniority']}, "
            f"role needs {job['seniority_required']}"
        )

    # ─── Education ─────────────────────────────────────────────────────────
    if student["education_level_idx"] >= job["education_required_idx"]:
        reasons_for.append(
            f"Education qualifies: {student['education_level']} "
            f"≥ {job['education_required']} required"
        )
    else:
        reasons_against.append(
            f"Education gap: {student['education_level']} vs "
            f"{job['education_required']} required"
        )

    # ─── Location / remote ─────────────────────────────────────────────────
    if student["city"] == job["city"]:
        reasons_for.append(f"Same city: {student['city']}")
    elif job["remote_ok"] or student["open_to_remote"]:
        reasons_for.append("Remote-friendly arrangement possible")
    else:
        reasons_against.append(
            f"Location mismatch: {student['city']} vs {job['city']} "
            f"(no remote option)"
        )

    # ─── Salary ────────────────────────────────────────────────────────────
    if student["salary_expectation_lpa"] <= job["salary_budget_lpa"]:
        reasons_for.append(
            f"Salary expectation fits: ₹{student['salary_expectation_lpa']}L "
            f"within ₹{job['salary_budget_lpa']}L budget"
        )
    else:
        over = round(student["salary_expectation_lpa"] - job["salary_budget_lpa"], 1)
        reasons_against.append(
            f"Salary expectation (₹{student['salary_expectation_lpa']}L) "
            f"exceeds budget by ₹{over}L"
        )

    # ─── Domain ────────────────────────────────────────────────────────────
    if student["domain"] == job["domain"]:
        reasons_for.append(
            f"Domain match: both in {student['domain'].replace('_', ' ')}"
        )
    else:
        reasons_against.append(
            f"Domain mismatch: student is in {student['domain'].replace('_', ' ')}, "
            f"role needs {job['domain'].replace('_', ' ')}"
        )

    # ─── Verdict ───────────────────────────────────────────────────────────
    score_pct = round(match_score * 100)
    if score_pct >= 75:
        verdict = "Strong Match"
        emoji = "🟢"
    elif score_pct >= 55:
        verdict = "Moderate Match"
        emoji = "🟡"
    elif score_pct >= 35:
        verdict = "Weak Match"
        emoji = "🟠"
    else:
        verdict = "No Match"
        emoji = "🔴"

    headline = (
        f"{emoji} {verdict} — {student['name']} × {job['title']} at {job['company']} "
        f"(Score: {score_pct}%)"
    )

    # ─── Recommendation ────────────────────────────────────────────────────
    if verdict == "Strong Match":
        rec = (
            f"Strongly recommend shortlisting. "
            f"Student qualifies on {len(above_threshold)}/{len(req_skills)} required skills "
            f"above the {threshold}-point threshold."
        )
    elif verdict == "Moderate Match":
        rec = (
            f"Consider for interview with awareness of gaps in: "
            f"{', '.join(list(missing)[:2]) if missing else 'skill scores'}. "
            f"Candidate shows potential."
        )
    elif verdict == "Weak Match":
        rec = (
            f"Not ideal — {len(missing)} required skills missing. "
            f"Only shortlist if pipeline is thin."
        )
    else:
        rec = (
            f"Do not shortlist. "
            f"Critical gaps: {', '.join(list(missing)[:3]) if missing else 'experience or domain'}."
        )

    return {
        "headline": headline,
        "verdict": verdict,
        "score_pct": score_pct,
        "match_score": round(match_score, 4),
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "skill_detail": skill_detail,
        "recommendation": rec,
        "feature_breakdown": feature_breakdown,
        "top_features_driving_match": (
            dict(list(feature_importances.items())[:5])
            if feature_importances else {}
        ),
    }


def plain_english_summary(explanation: dict) -> str:
    """One-paragraph summary suitable for a non-technical audience."""
    v = explanation["verdict"]
    p = explanation["score_pct"]
    pros = explanation["reasons_for"][:3]
    cons = explanation["reasons_against"][:2]

    pros_str = "; ".join(pros) if pros else "no significant strengths found"
    cons_str = "; ".join(cons) if cons else "no significant concerns"

    return (
        f"This is a {v} with an overall match score of {p}%. "
        f"Key strengths: {pros_str}. "
        f"Concerns: {cons_str}. "
        f"{explanation['recommendation']}"
    )
