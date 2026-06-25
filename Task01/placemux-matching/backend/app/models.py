"""
Marketplace data models — the persisted entities behind the student<->job
feature space. This is the schema Backend and the matching engine both
agree on (see API_CONTRACT.md for the wire contract built on top of it).
"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from .database import Base


class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    location = Column(String, default="Remote")

    jobs = relationship("Job", back_populates="company")


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, default="Remote")
    years_experience = Column(Float, default=0.0)
    education_level = Column(Integer, default=2)  # 1=highschool 2=bachelor 3=master 4=phd
    # verified_skills: {"python": 82, "sql": 70, ...}  scores are 0-100 and
    # come from PlaceMux's verified skill-assessment pipeline (upstream of this task)
    verified_skills = Column(JSON, default=dict)
    remote_ok = Column(Boolean, default=True)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    title = Column(String, nullable=False)
    location = Column(String, default="Remote")
    min_experience = Column(Float, default=0.0)
    role_level = Column(Integer, default=2)
    # required_skills: {"python": 60, "sql": 50, ...} -> skill: minimum verified score
    required_skills = Column(JSON, default=dict)
    remote_ok = Column(Boolean, default=True)

    company = relationship("Company", back_populates="jobs")


class MatchLog(Base):
    """Every scored match is persisted so scores are auditable, not ephemeral."""
    __tablename__ = "match_log"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    job_id = Column(Integer, ForeignKey("jobs.id"))
    model_score = Column(Float)
    baseline_score = Column(Float)
    explanation = Column(JSON)
