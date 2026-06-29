"""
PlaceMux Data Generator
Generates realistic sample data for student-job matching system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Tuple, List, Dict

class DataGenerator:
    """Generate realistic student and job data for matching"""
    
    SKILLS = [
        'Python', 'Java', 'JavaScript', 'React', 'SQL', 'MongoDB',
        'AWS', 'Docker', 'Machine Learning', 'Data Analysis', 'API Design',
        'REST', 'GraphQL', 'Node.js', 'Django', 'Flask', 'Spring Boot',
        'Vue.js', 'Angular', 'Git', 'CI/CD', 'Kubernetes', 'Microservices',
        'TensorFlow', 'PyTorch', 'Pandas', 'Numpy', 'Spark', 'Hadoop',
        'Communication', 'Problem Solving', 'Leadership', 'Agile'
    ]
    
    LOCATIONS = [
        'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune', 
        'Chennai', 'Kolkata', 'Remote'
    ]
    
    COMPANIES = [
        'TechCorp', 'DataWorks', 'CloudSys', 'AI Innovations', 'WebScale',
        'FinTech Solutions', 'MobileFirst', 'SecureNet', 'Analytics Pro'
    ]
    
    JOB_TITLES = [
        'Software Engineer', 'Data Scientist', 'ML Engineer', 
        'Full Stack Developer', 'Backend Engineer', 'Frontend Engineer',
        'DevOps Engineer', 'Data Engineer', 'Senior Developer'
    ]
    
    COLLEGES = [
        'IIT Bombay', 'IIT Delhi', 'IIT Kanpur', 'BITS Pilani',
        'NIT Trichy', 'VIT Vellore', 'IIIT Hyderabad', 'Delhi University'
    ]
    
    @staticmethod
    def generate_students(n_students: int = 100) -> pd.DataFrame:
        """Generate synthetic student data"""
        students = []
        
        for i in range(n_students):
            # Generate skills with confidence scores
            n_skills = np.random.randint(3, 8)
            student_skills = random.sample(DataGenerator.SKILLS, n_skills)
            skill_scores = {skill: round(np.random.uniform(0.5, 1.0), 2) 
                          for skill in student_skills}
            
            # Generate years of experience
            years_exp = np.random.randint(0, 5)
            
            # Generate GPA
            gpa = round(np.random.uniform(2.5, 4.0), 2)
            
            # Verify quality of skills (more skilled students have higher GPA)
            avg_skill_score = np.mean(list(skill_scores.values()))
            gpa = max(2.5, min(4.0, gpa + (avg_skill_score - 0.75) * 0.5))
            gpa = round(gpa, 2)
            
            student = {
                'student_id': f'STU_{i:04d}',
                'name': f'Student_{i}',
                'college': random.choice(DataGenerator.COLLEGES),
                'graduation_year': 2024 + np.random.randint(0, 2),
                'gpa': gpa,
                'years_of_experience': years_exp,
                'location': random.choice(DataGenerator.LOCATIONS),
                'skills': skill_scores,
                'projects_count': np.random.randint(0, 5),
                'internship_experience': np.random.randint(0, 3),
                'onboarding_date': (datetime.now() - timedelta(days=np.random.randint(0, 60))).date(),
                'verified': np.random.random() > 0.1  # 90% verified
            }
            students.append(student)
        
        return pd.DataFrame(students)
    
    @staticmethod
    def generate_jobs(n_jobs: int = 50) -> pd.DataFrame:
        """Generate synthetic job data"""
        jobs = []
        
        for i in range(n_jobs):
            # Generate required skills
            n_required_skills = np.random.randint(3, 8)
            required_skills = random.sample(DataGenerator.SKILLS, n_required_skills)
            
            # Create job
            job = {
                'job_id': f'JOB_{i:04d}',
                'company': random.choice(DataGenerator.COMPANIES),
                'title': random.choice(DataGenerator.JOB_TITLES),
                'location': random.choice(DataGenerator.LOCATIONS),
                'required_skills': required_skills,
                'required_years_experience': np.random.randint(0, 4),
                'minimum_gpa': round(np.random.uniform(2.5, 3.5), 2),
                'salary_min': 400000 + np.random.randint(0, 200000),
                'salary_max': 800000 + np.random.randint(0, 400000),
                'posted_date': (datetime.now() - timedelta(days=np.random.randint(0, 45))).date(),
                'is_active': np.random.random() > 0.2,  # 80% active
                'applications_count': np.random.randint(5, 100),
                'quality_score': round(np.random.uniform(0.6, 1.0), 2)
            }
            jobs.append(job)
        
        return pd.DataFrame(jobs)
    
    @staticmethod
    def generate_ground_truth(students: pd.DataFrame, jobs: pd.DataFrame, 
                             n_matches: int = 200) -> pd.DataFrame:
        """Generate ground truth matches based on realistic matching logic"""
        matches = []
        
        for _ in range(n_matches):
            student = students.sample(1).iloc[0]
            job = jobs.sample(1).iloc[0]
            
            # Calculate match score based on multiple factors
            
            # Skill match
            student_skill_names = set(student['skills'].keys())
            required_skill_names = set(job['required_skills'])
            skill_overlap = len(student_skill_names & required_skill_names)
            skill_match = skill_overlap / len(required_skill_names) if required_skill_names else 0
            
            # Experience match
            exp_match = 1.0 if student['years_of_experience'] >= job['required_years_experience'] else 0.7
            
            # GPA match
            gpa_match = 1.0 if student['gpa'] >= job['minimum_gpa'] else 0.7
            
            # Location match
            location_match = 1.0 if student['location'] == job['location'] or student['location'] == 'Remote' or job['location'] == 'Remote' else 0.8
            
            # Overall quality
            overall_quality = (skill_match * 0.4 + exp_match * 0.25 + gpa_match * 0.2 + location_match * 0.15)
            
            # Generate label based on quality (with some noise)
            is_good_match = 1 if (overall_quality > 0.65 and np.random.random() > 0.1) else 0
            
            match = {
                'student_id': student['student_id'],
                'job_id': job['job_id'],
                'skill_match_ratio': round(skill_match, 3),
                'experience_match': int(exp_match == 1.0),
                'gpa_match': int(gpa_match == 1.0),
                'location_match': int(location_match == 1.0),
                'quality_score': round(overall_quality, 3),
                'is_good_match': is_good_match
            }
            matches.append(match)
        
        return pd.DataFrame(matches)
    
    @staticmethod
    def generate_full_dataset(n_students: int = 100, 
                             n_jobs: int = 50,
                             n_matches: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset"""
        students = DataGenerator.generate_students(n_students)
        jobs = DataGenerator.generate_jobs(n_jobs)
        matches = DataGenerator.generate_ground_truth(students, jobs, n_matches)
        
        return students, jobs, matches
