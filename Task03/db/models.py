"""
PlaceMux - Database layer (SQLAlchemy + SQLite)
==================================================
Addresses the review feedback "address database integration for a more
robust application". All entities are persisted to a real relational
database (placemux.db) instead of living only as in-memory CSVs:

  Student, Skill, StudentSkill, Company, Job, JobSkillRequirement,
  Application (ground-truth label used for training/eval),
  MatchScore (every ranking the API computes is persisted here, so a
              company or student's results are durable across restarts
              and queryable for audit / "why did we show this" later).

SQLite is used for portability in this demo; the schema is plain SQL via
SQLAlchemy's ORM so swapping in Postgres in production is a one-line
connection-string change.
"""
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "placemux.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)
Base = declarative_base()


class Student(Base):
    __tablename__ = "students"
    student_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    branch = Column(String)
    experience_years = Column(Float, default=0.0)
    graduation_year = Column(Integer)

    skills = relationship("StudentSkill", back_populates="student", cascade="all, delete-orphan")


class StudentSkill(Base):
    __tablename__ = "student_skills"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    skill_name = Column(String, nullable=False)
    verified_score = Column(Float, nullable=False)

    student = relationship("Student", back_populates="skills")
    __table_args__ = (UniqueConstraint("student_id", "skill_name", name="uq_student_skill"),)


class Company(Base):
    __tablename__ = "companies"
    company_id = Column(Integer, primary_key=True)
    company_name = Column(String, nullable=False)

    jobs = relationship("Job", back_populates="company")


class Job(Base):
    __tablename__ = "jobs"
    job_id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.company_id"), nullable=False)
    title = Column(String, nullable=False)
    category = Column(String)
    min_experience_years = Column(Float, default=0.0)
    openings = Column(Integer, default=1)

    company = relationship("Company", back_populates="jobs")
    requirements = relationship("JobSkillRequirement", back_populates="job", cascade="all, delete-orphan")


class JobSkillRequirement(Base):
    __tablename__ = "job_skill_requirements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False)
    skill_name = Column(String, nullable=False)
    weight = Column(Float, default=1.0)
    min_required_score = Column(Float, default=50.0)

    job = relationship("Job", back_populates="requirements")
    __table_args__ = (UniqueConstraint("job_id", "skill_name", name="uq_job_skill"),)


class Application(Base):
    """Ground-truth labelled (student, job) pairs used to train/evaluate the models."""
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False)
    good_match = Column(Integer, nullable=False)  # 0/1 ground truth label


class MatchScore(Base):
    """Every ranking score the API computes is persisted here -> durable, auditable,
    queryable history of 'what did we show this student/company and why'."""
    __tablename__ = "match_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False)
    score = Column(Float, nullable=False)
    baseline_score = Column(Float, nullable=False)
    explanation = Column(String)
    computed_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    query_type = Column(String)  # "jobs_for_student" or "candidates_for_job"


def init_db():
    Base.metadata.create_all(ENGINE)


def get_session():
    return SessionLocal()
