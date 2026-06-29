"""
PlaceMux Data Generator
Generates realistic synthetic student-job matching data with verified skills
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json


class PlaceMuxDataGenerator:
    """Generate realistic student-job matching dataset"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

        # Skill categories
        self.technical_skills = [
            'Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'Docker', 'React',
            'Node.js', 'MongoDB', 'Machine Learning', 'Data Analysis', 'Git',
            'Linux', 'C++', 'Spring Boot', 'Kubernetes', 'Microservices'
        ]

        self.soft_skills = [
            'Communication', 'Leadership', 'Problem Solving', 'Teamwork',
            'Time Management', 'Adaptability', 'Critical Thinking'
        ]

        self.roles = [
            'Software Engineer', 'Data Scientist', 'Full Stack Developer',
            'Backend Developer', 'Frontend Developer', 'DevOps Engineer',
            'ML Engineer', 'QA Engineer', 'Solutions Architect'
        ]

    def generate_students(self, n_students=200):
        """Generate student profiles with verified skills"""
        students = []

        for i in range(n_students):
            # Randomly select skills
            n_tech_skills = np.random.randint(3, 10)
            n_soft_skills = np.random.randint(2, 5)

            tech_skills = random.sample(self.technical_skills, n_tech_skills)
            soft_skills = random.sample(self.soft_skills, n_soft_skills)

            # Verified skill scores (0-100)
            tech_scores = {skill: np.random.normal(70, 15) for skill in tech_skills}
            soft_scores = {skill: np.random.normal(75, 12) for skill in soft_skills}

            # Clamp scores to 0-100
            tech_scores = {k: max(0, min(100, v)) for k, v in tech_scores.items()}
            soft_scores = {k: max(0, min(100, v)) for k, v in soft_scores.items()}

            student = {
                'student_id': f'STU_{i + 1:05d}',
                'college': f'College_{random.randint(1, 10)}',
                'gpa': np.random.normal(7.5, 1.2),
                'years_of_experience': np.random.randint(0, 4),
                'technical_skills': tech_skills,
                'technical_scores': tech_scores,
                'soft_skills': soft_skills,
                'soft_scores': soft_scores,
                'avg_technical_score': np.mean(list(tech_scores.values())),
                'avg_soft_score': np.mean(list(soft_scores.values())),
                'registration_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
            }

            student['gpa'] = max(0, min(10, student['gpa']))
            students.append(student)

        return students

    def generate_jobs(self, n_jobs=150):
        """Generate job postings with skill requirements"""
        jobs = []

        for i in range(n_jobs):
            n_required_tech = np.random.randint(3, 8)
            n_required_soft = np.random.randint(2, 4)

            required_tech = random.sample(self.technical_skills, n_required_tech)
            required_soft = random.sample(self.soft_skills, n_required_soft)

            job = {
                'job_id': f'JOB_{i + 1:05d}',
                'company': f'Company_{random.randint(1, 30)}',
                'role': random.choice(self.roles),
                'required_technical_skills': required_tech,
                'required_soft_skills': required_soft,
                'min_gpa': np.random.normal(6.5, 1.0),
                'experience_required': np.random.randint(0, 5),
                'salary_range_min': random.choice([400000, 500000, 600000, 700000]),
                'salary_range_max': random.choice([800000, 900000, 1000000, 1200000]),
                'posted_date': (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
                'applicant_count': np.random.randint(10, 500)
            }
            job['min_gpa'] = max(0, min(10, job['min_gpa']))
            jobs.append(job)

        return jobs

    def generate_matches(self, students, jobs, n_samples=500):
        """Generate student-job matches with ground truth labels"""
        matches = []

        for _ in range(n_samples):
            student = random.choice(students)
            job = random.choice(jobs)

            # Calculate matching features
            student_tech = set(student['technical_skills'])
            job_tech = set(job['required_technical_skills'])

            student_soft = set(student['soft_skills'])
            job_soft = set(job['required_soft_skills'])

            # Feature engineering
            tech_overlap = len(student_tech & job_tech) / max(len(job_tech), 1)
            soft_overlap = len(student_soft & job_soft) / max(len(job_soft), 1)
            gpa_fit = 1 - abs(student['gpa'] - job['min_gpa']) / 10
            exp_fit = 1 - abs(student['years_of_experience'] - job['experience_required']) / 5

            tech_score_avg = student['avg_technical_score']
            soft_score_avg = student['avg_soft_score']

            # Ground truth: High overlap + good scores = good match
            match_score = (
                    0.35 * tech_overlap +
                    0.20 * soft_overlap +
                    0.20 * (tech_score_avg / 100) +
                    0.15 * (soft_score_avg / 100) +
                    0.05 * max(0, gpa_fit) +
                    0.05 * max(0, exp_fit)
            )

            # Add some noise
            match_score += np.random.normal(0, 0.05)
            match_score = max(0, min(1, match_score))

            # Label: match if score > 0.6
            is_match = 1 if match_score > 0.6 else 0

            match = {
                'student_id': student['student_id'],
                'job_id': job['job_id'],
                'college': student['college'],
                'company': job['company'],
                'tech_overlap': tech_overlap,
                'soft_overlap': soft_overlap,
                'student_avg_tech_score': tech_score_avg,
                'student_avg_soft_score': soft_score_avg,
                'student_gpa': student['gpa'],
                'student_years_exp': student['years_of_experience'],
                'job_min_gpa': job['min_gpa'],
                'job_exp_required': job['experience_required'],
                'gpa_fit': max(0, gpa_fit),
                'exp_fit': max(0, exp_fit),
                'job_applicant_count': job['applicant_count'],
                'is_good_match': is_match,
                'match_score': match_score
            }

            matches.append(match)

        return matches

    def save_data(self, students, jobs, matches, output_dir='data'):
        """Save generated data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save students
        students_df = pd.DataFrame([
            {
                'student_id': s['student_id'],
                'college': s['college'],
                'gpa': s['gpa'],
                'years_of_experience': s['years_of_experience'],
                'avg_technical_score': s['avg_technical_score'],
                'avg_soft_score': s['avg_soft_score'],
                'technical_skills_count': len(s['technical_skills']),
                'soft_skills_count': len(s['soft_skills']),
            }
            for s in students
        ])
        students_df.to_csv(f'{output_dir}/students.csv', index=False)

        # Save jobs
        jobs_df = pd.DataFrame([
            {
                'job_id': j['job_id'],
                'company': j['company'],
                'role': j['role'],
                'min_gpa': j['min_gpa'],
                'experience_required': j['experience_required'],
                'salary_min': j['salary_range_min'],
                'salary_max': j['salary_range_max'],
                'required_tech_count': len(j['required_technical_skills']),
                'required_soft_count': len(j['required_soft_skills']),
                'applicant_count': j['applicant_count'],
            }
            for j in jobs
        ])
        jobs_df.to_csv(f'{output_dir}/jobs.csv', index=False)

        # Save matches
        matches_df = pd.DataFrame(matches)
        matches_df.to_csv(f'{output_dir}/matches.csv', index=False)

        # Save raw JSON for reference
        with open(f'{output_dir}/students.json', 'w') as f:
            json.dump(students, f, indent=2)
        with open(f'{output_dir}/jobs.json', 'w') as f:
            json.dump(jobs, f, indent=2)

        return students_df, jobs_df, matches_df


if __name__ == '__main__':
    print("Generating PlaceMux dataset...")
    generator = PlaceMuxDataGenerator(seed=42)

    students = generator.generate_students(n_students=200)
    jobs = generator.generate_jobs(n_jobs=150)
    matches = generator.generate_matches(students, jobs, n_samples=500)

    students_df, jobs_df, matches_df = generator.save_data(students, jobs, matches)

    print(f"✓ Generated {len(students)} students")
    print(f"✓ Generated {len(jobs)} jobs")
    print(f"✓ Generated {len(matches)} matches")
    print(f"✓ Class distribution: {matches_df['is_good_match'].value_counts().to_dict()}")
    print(f"✓ Data saved to 'data/' directory")