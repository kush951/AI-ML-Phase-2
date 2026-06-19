"""
Pydantic schemas = the wire format of the matching API contract that
Backend and Frontend integrate against. Keep these in sync with
API_CONTRACT.md.
"""
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Dict, Optional, List


class CompanySignup(BaseModel):
    name: str
    email: EmailStr
    location: str = "Remote"


class CompanyOut(BaseModel):
    id: int
    name: str
    email: str
    location: str

    class Config:
        from_attributes = True


class JobCreate(BaseModel):
    title: str
    location: str = "Remote"
    min_experience: float = 0.0
    role_level: int = 2
    required_skills: Dict[str, float]  # skill -> minimum verified score (0-100)
    remote_ok: bool = True


class JobOut(BaseModel):
    id: int
    company_id: int
    title: str
    location: str
    min_experience: float
    required_skills: Dict[str, float]

    class Config:
        from_attributes = True


class StudentCreate(BaseModel):
    name: str
    location: str = "Remote"
    years_experience: float = 0.0
    education_level: int = 2
    verified_skills: Dict[str, float]
    remote_ok: bool = True


class StudentOut(BaseModel):
    id: int
    name: str
    location: str
    years_experience: float
    verified_skills: Dict[str, float]

    class Config:
        from_attributes = True


class MatchExplanation(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    experience_fit: str
    location_fit: str
    plain_english: str


class MatchResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    student_id: int
    student_name: str
    job_id: int
    model_score: float          # calibrated probability of a good match, 0-1
    baseline_score: float       # naive skill-overlap score, 0-1, for comparison
    rank: int
    explanation: MatchExplanation


class MetricsOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    trained_on_pairs: int
    test_set_size: int
    baseline: Dict[str, float]
    model: Dict[str, float]
    lift_over_baseline_pct: float
