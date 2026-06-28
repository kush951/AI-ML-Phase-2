"""
Sample Data Generator - Create realistic training data for job matching
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Generate realistic sample data for training"""
    
    SKILLS = [
        'Python', 'Java', 'JavaScript', 'SQL', 'AWS',
        'Docker', 'Kubernetes', 'React', 'Node.js', 'Machine Learning',
        'Data Analysis', 'Communication', 'Project Management', 'Leadership',
        'Problem Solving', 'System Design', 'DevOps', 'API Design'
    ]
    
    COLLEGES = ['College_A', 'College_B', 'College_C', 'College_D', 'College_E']
    
    @staticmethod
    def generate_student_profile(student_id, college_id, match_likelihood=0.5):
        """Generate a student profile"""
        n_skills = np.random.randint(4, 10)
        skills = list(np.random.choice(SampleDataGenerator.SKILLS, n_skills, replace=False))
        skill_scores = {skill: np.random.uniform(0.6, 1.0) for skill in skills}
        
        return {
            'student_id': student_id,
            'college_id': college_id,
            'verified_skills': skills,
            'skill_scores': skill_scores,
            'gpa': np.random.uniform(2.5, 4.0),
            'experience_years': np.random.uniform(0, 5),
        }
    
    @staticmethod
    def generate_job_profile(job_id, college_id, match_likelihood=0.5):
        """Generate a job profile"""
        required_n_skills = np.random.randint(3, 7)
        required_skills = list(np.random.choice(
            SampleDataGenerator.SKILLS,
            required_n_skills,
            replace=False
        ))
        
        return {
            'job_id': job_id,
            'college_id': college_id,
            'title': f'Role_{job_id}',
            'company': f'Company_{np.random.randint(1, 20)}',
            'required_skills': required_skills,
            'required_exp_years': np.random.uniform(0, 8),
            'salary_range': f'{100000 + np.random.randint(0, 200000)}-{300000 + np.random.randint(0, 200000)}',
        }

def load_sample_data(n_samples=500, test_size=0.2):
    """
    Load or generate sample data for training
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    logger.info(f"🔄 Generating {n_samples} sample data points...")
    
    X_data = []
    y_data = []
    
    feature_names = [
        'skill_overlap_ratio',
        'required_skills_covered',
        'skill_gap',
        'avg_matching_skill_score',
        'avg_all_skill_scores',
        'experience_gap',
        'excess_experience',
        'experience_match_ratio',
        'gpa_normalized',
        'total_verified_skills',
        'required_skills_count',
        'college_match'
    ]
    
    # Generate balanced dataset
    for i in range(n_samples):
        college_id = np.random.choice(SampleDataGenerator.COLLEGES)
        
        # Decide if this should be a match or not (50-50)
        is_match = np.random.rand() > 0.5
        
        if is_match:
            # Generate matching pair
            student = SampleDataGenerator.generate_student_profile(
                f'ST_{i}', college_id, match_likelihood=0.8
            )
            job = SampleDataGenerator.generate_job_profile(
                f'JB_{i}', college_id, match_likelihood=0.8
            )
            
            # Make job requirements more aligned with student skills
            n_to_align = np.random.randint(1, len(job['required_skills']))
            aligned_skills = np.random.choice(
                student['verified_skills'],
                min(n_to_align, len(student['verified_skills'])),
                replace=False
            )
            job['required_skills'] = list(aligned_skills) + [
                s for s in job['required_skills']
                if s not in aligned_skills
            ]
            
            # Adjust experience
            if np.random.rand() > 0.5:
                student['experience_years'] = np.random.uniform(
                    job['required_exp_years'] - 1,
                    job['required_exp_years'] + 3
                )
            
            label = 1
        else:
            # Generate non-matching pair
            college_1 = np.random.choice(SampleDataGenerator.COLLEGES)
            college_2 = np.random.choice(SampleDataGenerator.COLLEGES)
            
            student = SampleDataGenerator.generate_student_profile(
                f'ST_{i}', college_1, match_likelihood=0.2
            )
            job = SampleDataGenerator.generate_job_profile(
                f'JB_{i}', college_2, match_likelihood=0.2
            )
            
            # Make requirements less aligned with student skills
            student['verified_skills'] = list(np.random.choice(
                SampleDataGenerator.SKILLS,
                np.random.randint(1, 4),
                replace=False
            ))
            job['required_skills'] = list(np.random.choice(
                SampleDataGenerator.SKILLS,
                np.random.randint(5, 10),
                replace=False
            ))
            
            # Add experience gap
            student['experience_years'] = np.random.uniform(0, 1)
            job['required_exp_years'] = np.random.uniform(5, 8)
            
            label = 0
        
        # Extract features
        features = _extract_features(student, job)
        X_data.append(features)
        y_data.append(label)
    
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    
    logger.info(f"✓ Generated data shape: {X.shape}")
    logger.info(f"✓ Label distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"✓ Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_names

def _extract_features(student, job):
    """
    Extract features from student and job profiles
    
    Must match the feature extraction in ExplainabilityEngine
    """
    features = []
    
    # Skill features
    student_skills = set(student['verified_skills'])
    required_skills = set(job['required_skills'])
    
    # Feature 1: Skill overlap ratio
    if len(required_skills) > 0:
        skill_overlap = len(student_skills.intersection(required_skills)) / len(required_skills)
    else:
        skill_overlap = 0.0
    features.append(skill_overlap)
    
    # Feature 2: Skill coverage
    required_skills_covered = len(student_skills.intersection(required_skills))
    features.append(required_skills_covered)
    
    # Feature 3: Skill gap
    skill_gap = len(required_skills) - required_skills_covered
    features.append(skill_gap)
    
    # Feature 4: Average score of matching skills
    matching_skills = student_skills.intersection(required_skills)
    if matching_skills:
        avg_matching_score = np.mean([
            student['skill_scores'].get(skill, 0)
            for skill in matching_skills
        ])
    else:
        avg_matching_score = 0.0
    features.append(avg_matching_score)
    
    # Feature 5: Average score of all student skills
    avg_all_skills = np.mean([
        score for score in student['skill_scores'].values()
    ]) if student['skill_scores'] else 0.0
    features.append(avg_all_skills)
    
    # Experience features
    exp_gap = max(0, job['required_exp_years'] - student['experience_years'])
    features.append(exp_gap)
    
    excess_exp = max(0, student['experience_years'] - job['required_exp_years'])
    features.append(excess_exp)
    
    if job['required_exp_years'] > 0:
        exp_match = min(1.0, student['experience_years'] / job['required_exp_years'])
    else:
        exp_match = 1.0
    features.append(exp_match)
    
    # Performance features
    gpa_normalized = min(1.0, student['gpa'] / 4.0)
    features.append(gpa_normalized)
    
    # Skill counts
    features.append(len(student_skills))
    features.append(len(required_skills))
    
    # College match
    college_match = 1.0 if student['college_id'] == job['college_id'] else 0.0
    features.append(college_match)
    
    return np.array(features, dtype=np.float32)
