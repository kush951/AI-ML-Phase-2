"""
PlaceMux - Data Generator
Generates synthetic but realistic student-job matching data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self, n_students=200, n_jobs=100, random_state=42):
        np.random.seed(random_state)
        self.n_students = n_students
        self.n_jobs = n_jobs
        self.students_df = None
        self.jobs_df = None
        self.matches_df = None
        
    def generate_students(self):
        """Generate synthetic student data"""
        skills = ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'Docker', 
                 'React', 'Node.js', 'Machine Learning', 'Data Analysis']
        
        students = []
        for i in range(self.n_students):
            # Random skills (0-100 score)
            student_skills = {skill: np.random.randint(30, 100) for skill in skills}
            
            students.append({
                'student_id': f'STU_{i:04d}',
                'years_experience': np.random.randint(0, 10),
                'gpa': np.random.uniform(2.5, 4.0),
                'certifications': np.random.randint(0, 5),
                'projects_completed': np.random.randint(0, 20),
                'aptitude_score': np.random.randint(50, 100),
                **{f'skill_{skill.lower().replace(" ", "_")}': score 
                   for skill, score in student_skills.items()}
            })
        
        self.students_df = pd.DataFrame(students)
        return self.students_df
    
    def generate_jobs(self):
        """Generate synthetic job data"""
        skills = ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'Docker', 
                 'React', 'Node.js', 'Machine Learning', 'Data Analysis']
        job_titles = ['Junior Developer', 'Senior Developer', 'Data Scientist', 
                     'Full Stack Engineer', 'DevOps Engineer', 'ML Engineer']
        
        jobs = []
        for i in range(self.n_jobs):
            # Required skills (0-100 importance)
            required_skills = {skill: np.random.randint(40, 100) for skill in np.random.choice(skills, 3, replace=False)}
            
            jobs.append({
                'job_id': f'JOB_{i:04d}',
                'title': np.random.choice(job_titles),
                'min_experience': np.random.randint(0, 8),
                'min_gpa': np.random.uniform(2.5, 3.8),
                'required_certifications': np.random.randint(0, 3),
                'salary_range_min': np.random.randint(40000, 80000),
                'salary_range_max': np.random.randint(80000, 180000),
                'company_rating': np.random.uniform(3.0, 5.0),
                **{f'skill_{skill.lower().replace(" ", "_")}_required': score 
                   for skill, score in required_skills.items()}
            })
        
        self.jobs_df = pd.DataFrame(jobs)
        return self.jobs_df
    
    def generate_matches(self):
        """Generate match labels (0 = no match, 1 = good match)"""
        matches = []
        
        for _, student in self.students_df.iterrows():
            for _, job in self.jobs_df.iterrows():
                # Calculate match score based on features
                skill_match = 0
                skill_count = 0
                
                for col in student.index:
                    if col.startswith('skill_'):
                        skill_name = col.replace('skill_', '').replace('_', ' ')
                        required_col = f'skill_{skill_name}_required'
                        
                        if required_col in job.index:
                            student_score = student[col]
                            required_score = job[required_col]
                            # Skill match: how well student meets requirement
                            if required_score > 0:
                                skill_match += (student_score / required_score) * required_score
                                skill_count += required_score
                
                skill_match = (skill_match / skill_count * 100) if skill_count > 0 else 0
                
                # Experience match
                exp_match = 100 - abs(student['years_experience'] - job['min_experience']) * 5
                exp_match = max(0, min(100, exp_match))
                
                # GPA match
                gpa_match = 100 if student['gpa'] >= job['min_gpa'] else (student['gpa'] / job['min_gpa']) * 100
                
                # Calculate final match (0-100)
                final_match = (skill_match * 0.5 + exp_match * 0.25 + gpa_match * 0.25)
                
                # Add some noise
                final_match += np.random.normal(0, 5)
                final_match = max(0, min(100, final_match))
                
                # Label: good match if score > 70
                label = 1 if final_match > 65 else 0
                
                matches.append({
                    'student_id': student['student_id'],
                    'job_id': job['job_id'],
                    'match_score': final_match,
                    'label': label
                })
        
        self.matches_df = pd.DataFrame(matches)
        return self.matches_df
    
    def create_training_data(self):
        """Merge all data for training"""
        # Merge students and jobs with matches
        training_data = self.matches_df.merge(self.students_df, on='student_id')
        training_data = training_data.merge(self.jobs_df, on='job_id')
        
        return training_data
    
    def save_data(self, path='data/'):
        """Save generated data to CSV files"""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.students_df.to_csv(f'{path}students.csv', index=False)
        self.jobs_df.to_csv(f'{path}jobs.csv', index=False)
        self.matches_df.to_csv(f'{path}matches.csv', index=False)
        
        training_data = self.create_training_data()
        training_data.to_csv(f'{path}training_data.csv', index=False)
        
        print(f"✓ Data saved to {path}")
        print(f"  - {len(self.students_df)} students")
        print(f"  - {len(self.jobs_df)} jobs")
        print(f"  - {len(self.matches_df)} student-job pairs")


if __name__ == '__main__':
    generator = DataGenerator(n_students=200, n_jobs=100)
    generator.generate_students()
    generator.generate_jobs()
    generator.generate_matches()
    generator.save_data('data/')
    
    print("\n✓ Synthetic data generated successfully!")
