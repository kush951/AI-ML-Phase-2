"""
Data Generation & Preparation Module
Generates realistic student and job data for recommendation system
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, MODEL_CONFIG

class DataGenerator:
    def __init__(self, n_students=300, n_jobs=200, random_state=42):
        self.n_students = n_students
        self.n_jobs = n_jobs
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_students(self):
        """Generate student data with skills, experience, and education"""
        students = []
        
        skill_domains = ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'ML', 
                        'Data Analysis', 'React', 'Django', 'API Design']
        degrees = ['BTech', 'MTech', 'BSc', 'MSc', 'MBA']
        specializations = ['CSE', 'IT', 'DS', 'AI', 'Finance', 'Business']
        
        for i in range(self.n_students):
            student = {
                'student_id': f'STU_{i+1:04d}',
                'college_id': f'COL_{np.random.randint(1, 21):02d}',
                'name': f'Student_{i+1}',
                'degree': np.random.choice(degrees),
                'specialization': np.random.choice(specializations),
                'cgpa': np.round(np.random.uniform(6.0, 9.5), 2),
                'years_of_experience': np.random.randint(0, 5),
                'verified_skills': ','.join(np.random.choice(skill_domains, 
                                                            size=np.random.randint(2, 6), 
                                                            replace=False)),
                'skill_level': np.random.choice(['Basic', 'Intermediate', 'Advanced'], 
                                               size=1, 
                                               p=[0.3, 0.5, 0.2])[0],
                'communication_score': np.round(np.random.uniform(0, 1), 2),
                'teamwork_score': np.round(np.random.uniform(0, 1), 2),
                'leadership_score': np.round(np.random.uniform(0, 1), 2),
                'internship_count': np.random.randint(0, 4),
                'project_count': np.random.randint(1, 8),
            }
            students.append(student)
        
        return pd.DataFrame(students)
    
    def generate_jobs(self):
        """Generate job data with requirements"""
        jobs = []
        
        job_titles = ['Software Engineer', 'Data Scientist', 'ML Engineer', 
                     'Full Stack Developer', 'DevOps Engineer', 'Cloud Architect',
                     'Backend Engineer', 'Frontend Developer', 'AI/ML Engineer',
                     'Data Engineer']
        
        companies = ['TechCorp', 'DataDynamics', 'CloudSys', 'AIInnovate', 
                    'DevFactory', 'WebSolutions']
        
        required_skills = ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'ML', 
                          'Data Analysis', 'React', 'Django', 'API Design']
        
        experience_levels = ['Fresher', 'Junior (0-2 yrs)', 'Mid (2-5 yrs)', 'Senior (5+ yrs)']
        
        for i in range(self.n_jobs):
            job = {
                'job_id': f'JOB_{i+1:04d}',
                'job_title': np.random.choice(job_titles),
                'company': np.random.choice(companies),
                'experience_level': np.random.choice(experience_levels),
                'min_experience': np.random.randint(0, 6),
                'required_skills': ','.join(np.random.choice(required_skills, 
                                                            size=np.random.randint(3, 7), 
                                                            replace=False)),
                'min_cgpa': np.round(np.random.uniform(6.0, 8.0), 2),
                'salary_range': f"{np.random.randint(3, 25)}L - {np.random.randint(25, 50)}L",
                'location': np.random.choice(['Bangalore', 'Hyderabad', 'Pune', 'Mumbai', 'Delhi']),
                'job_type': np.random.choice(['Full-time', 'Contract']),
                'posted_date': pd.Timestamp.now(),
            }
            jobs.append(job)
        
        return pd.DataFrame(jobs)
    
    def generate_matches(self, students_df, jobs_df):
        """Generate historical match data for training"""
        matches = []
        
        for _ in range(int(len(students_df) * len(jobs_df) * 0.1)):  # 10% of possible pairs
            student = students_df.sample(1).iloc[0]
            job = jobs_df.sample(1).iloc[0]
            
            # Calculate true match based on features
            skill_match = len(set(student['verified_skills'].split(',')) & 
                            set(job['required_skills'].split(','))) / \
                         len(set(job['required_skills'].split(',')))
            
            exp_match = 1.0 if student['years_of_experience'] >= job['min_experience'] else \
                       student['years_of_experience'] / (job['min_experience'] + 1)
            
            cgpa_match = 1.0 if student['cgpa'] >= job['min_cgpa'] else \
                        student['cgpa'] / job['min_cgpa']
            
            overall_match = (skill_match * 0.5 + exp_match * 0.3 + cgpa_match * 0.2)
            
            # Determine if it's a good match (threshold-based)
            is_match = 1 if overall_match > 0.6 else 0
            
            match = {
                'student_id': student['student_id'],
                'job_id': job['job_id'],
                'skill_match_score': np.round(skill_match, 2),
                'experience_match_score': np.round(exp_match, 2),
                'cgpa_match_score': np.round(cgpa_match, 2),
                'overall_match_score': np.round(overall_match, 2),
                'is_match': is_match,
                'student_applied': np.random.choice([0, 1], p=[0.7, 0.3]),
                'student_selected': is_match * np.random.choice([0, 1], p=[0.4, 0.6]),
            }
            matches.append(match)
        
        return pd.DataFrame(matches)
    
    def save_data(self, students_df, jobs_df, matches_df):
        """Save generated data to CSV files"""
        students_df.to_csv(DATA_DIR / 'students.csv', index=False)
        jobs_df.to_csv(DATA_DIR / 'jobs.csv', index=False)
        matches_df.to_csv(DATA_DIR / 'matches.csv', index=False)
        
        print(f"✓ Generated {len(students_df)} students")
        print(f"✓ Generated {len(jobs_df)} jobs")
        print(f"✓ Generated {len(matches_df)} historical matches")
        
    def generate_all(self):
        """Generate all data"""
        students = self.generate_students()
        jobs = self.generate_jobs()
        matches = self.generate_matches(students, jobs)
        self.save_data(students, jobs, matches)
        return students, jobs, matches


class DataLoader:
    """Load and prepare data for model training"""
    
    @staticmethod
    def load_data():
        """Load all data from CSV files"""
        students = pd.read_csv(DATA_DIR / 'students.csv')
        jobs = pd.read_csv(DATA_DIR / 'jobs.csv')
        matches = pd.read_csv(DATA_DIR / 'matches.csv')
        return students, jobs, matches
    
    @staticmethod
    def prepare_features(students_df, jobs_df, matches_df):
        """Prepare features for model training"""
        # Merge data
        data = matches_df.copy()
        data = data.merge(students_df[['student_id', 'cgpa', 'years_of_experience', 
                                        'communication_score', 'teamwork_score', 
                                        'leadership_score', 'skill_level']], 
                         on='student_id')
        data = data.merge(jobs_df[['job_id', 'min_cgpa', 'min_experience']], 
                         on='job_id')
        
        # Additional features
        data['soft_skill_avg'] = (data['communication_score'] + 
                                 data['teamwork_score'] + 
                                 data['leadership_score']) / 3
        
        data['exp_surplus'] = data['years_of_experience'] - data['min_experience']
        data['cgpa_surplus'] = data['cgpa'] - data['min_cgpa']
        
        return data
    
    @staticmethod
    def get_train_test_split(data, target_col='is_match'):
        """Split data into train, validation, and test sets"""
        from sklearn.model_selection import train_test_split
        
        X = data.drop(columns=['is_match', 'student_id', 'job_id'])
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=MODEL_CONFIG['validation_split'],
            random_state=MODEL_CONFIG['random_state']
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Generate sample data
    generator = DataGenerator(n_students=300, n_jobs=200)
    students, jobs, matches = generator.generate_all()
    
    print("\nData generation complete!")
    print(f"Students shape: {students.shape}")
    print(f"Jobs shape: {jobs.shape}")
    print(f"Matches shape: {matches.shape}")
