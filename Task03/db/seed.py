"""
PlaceMux - Seed the database from the generated CSVs
========================================================
Loads students.csv, student_skills.csv, companies.csv, jobs.csv,
job_skills.csv, and applications.csv into placemux.db. Idempotent: running
twice wipes and reloads cleanly rather than producing duplicates.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.models import (
    init_db, get_session, Student, StudentSkill, Company, Job,
    JobSkillRequirement, Application, MatchScore, Base, ENGINE
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def seed():
    # fresh schema every seed run (demo-friendly, idempotent)
    Base.metadata.drop_all(ENGINE)
    init_db()
    session = get_session()

    students = pd.read_csv(DATA / "students.csv")
    student_skills = pd.read_csv(DATA / "student_skills.csv")
    companies = pd.read_csv(DATA / "companies.csv")
    jobs = pd.read_csv(DATA / "jobs.csv")
    job_skills = pd.read_csv(DATA / "job_skills.csv")
    applications = pd.read_csv(DATA / "applications.csv")

    session.bulk_insert_mappings(Student, students.to_dict("records"))
    session.bulk_insert_mappings(StudentSkill, student_skills.to_dict("records"))
    session.bulk_insert_mappings(Company, companies.to_dict("records"))
    session.bulk_insert_mappings(Job, jobs.to_dict("records"))
    session.bulk_insert_mappings(JobSkillRequirement, job_skills.to_dict("records"))
    session.bulk_insert_mappings(Application, applications.to_dict("records"))
    session.commit()

    print(f"Seeded DB at {ENGINE.url}")
    print(f"  students:        {session.query(Student).count()}")
    print(f"  student_skills:  {session.query(StudentSkill).count()}")
    print(f"  companies:       {session.query(Company).count()}")
    print(f"  jobs:            {session.query(Job).count()}")
    print(f"  job_skills:      {session.query(JobSkillRequirement).count()}")
    print(f"  applications:    {session.query(Application).count()}")
    session.close()


if __name__ == "__main__":
    seed()
