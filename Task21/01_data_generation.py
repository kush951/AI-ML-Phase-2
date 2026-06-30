"""
PlaceMux Data Generation & Preprocessing
Generates synthetic student-job matching dataset with realistic features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json

np.random.seed(42)

class DataGenerator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.skills = ['Python', 'JavaScript', 'Java', 'SQL', 'AWS', 'Docker', 
                      'React', 'Node.js', 'ML', 'Statistics', 'Git', 'Data Analysis']
        self.backgrounds = ['CSE', 'IT', 'ECE', 'Mechanical', 'Civil', 'Other']
        self.experience_levels = ['Fresher', '1-2 years', '2-5 years', '5+ years']
        
    def generate_students(self):
        """Generate synthetic student data"""
        data = {
            'student_id': range(1, self.n_samples + 1),
            'background': np.random.choice(self.backgrounds, self.n_samples),
            'years_exp': np.random.choice([0, 1, 2, 3, 5, 7], self.n_samples),
            'gpa': np.random.uniform(6.0, 9.5, self.n_samples),
            'num_projects': np.random.randint(0, 15, self.n_samples),
            'internships': np.random.randint(0, 4, self.n_samples),
            'certifications': np.random.randint(0, 3, self.n_samples),
        }
        
        # Add verified skill scores (0-100)
        for skill in self.skills[:6]:  # Primary skills
            data[f'verified_{skill.lower()}'] = np.random.choice(
                [0, 0, 0, 25, 50, 75, 85, 90, 95, 100], self.n_samples
            )
        
        # Add demographic info (for fairness audit)
        data['gender'] = np.random.choice(['M', 'F'], self.n_samples, p=[0.6, 0.4])
        data['region'] = np.random.choice(['Metro', 'Tier1', 'Tier2', 'Tier3'], 
                                         self.n_samples, p=[0.3, 0.3, 0.25, 0.15])
        
        return pd.DataFrame(data)
    
    def generate_jobs(self):
        """Generate synthetic job postings"""
        n_jobs = int(self.n_samples * 0.1)  # 10% of students
        data = {
            'job_id': range(1, n_jobs + 1),
            'company_size': np.random.choice(['Startup', 'SME', 'Large'], n_jobs, 
                                            p=[0.3, 0.4, 0.3]),
            'salary_min': np.random.randint(200000, 600000, n_jobs),
            'salary_max': np.random.randint(600000, 1500000, n_jobs),
            'required_exp_min': np.random.randint(0, 5, n_jobs),
            'required_exp_max': np.random.randint(5, 15, n_jobs),
            'required_gpa_min': np.random.uniform(5.5, 7.5, n_jobs),
            'urgency_level': np.random.choice(['Low', 'Medium', 'High'], n_jobs),
        }
        
        # Add required skill scores
        for skill in self.skills[:6]:
            data[f'req_{skill.lower()}'] = np.random.choice(
                [0, 0, 0, 0, 25, 50, 75, 85], n_jobs
            )
        
        return pd.DataFrame(data)
    
    def compute_match_score(self, student, job):
        """Compute actual match score (ground truth)"""
        score = 0
        weights = {}
        
        # Verified skills matching (highest weight)
        skills_weight = 0
        skills_count = 0
        for skill in self.skills[:6]:
            student_score = student.get(f'verified_{skill.lower()}', 0)
            job_req = job.get(f'req_{skill.lower()}', 0)
            if job_req > 0:
                match = min(student_score, job_req) / max(student_score, job_req) if max(student_score, job_req) > 0 else 0
                skills_weight += match
                skills_count += 1
        skills_weight = (skills_weight / skills_count * 100) if skills_count > 0 else 0
        
        # Experience matching
        exp_diff = abs(student['years_exp'] - job['required_exp_min'])
        exp_score = max(0, 100 - (exp_diff * 15))
        
        # GPA matching
        gpa_match = 100 if student['gpa'] >= job['required_gpa_min'] else max(0, 100 - (job['required_gpa_min'] - student['gpa']) * 20)
        
        # Projects and internships
        activity_score = min(100, (student['num_projects'] + student['internships'] * 2) * 8)
        
        # Final score (weighted)
        final_score = (skills_weight * 0.4 + exp_score * 0.25 + gpa_match * 0.2 + activity_score * 0.15)
        
        return min(100, max(0, final_score))
    
    def generate_matches(self, students, jobs):
        """Generate student-job matches"""
        matches = []
        
        for idx, (_, student) in enumerate(students.iterrows()):
            # Each student gets compared to random jobs
            sample_jobs = jobs.sample(min(20, len(jobs)))
            
            for _, job in sample_jobs.iterrows():
                match_score = self.compute_match_score(student.to_dict(), job.to_dict())
                
                # Convert to binary: match if score > 70
                is_match = 1 if match_score >= 70 else 0
                
                # Add some noise (real-world scenarios don't have perfect predictions)
                if np.random.random() < 0.05:  # 5% noise
                    is_match = 1 - is_match
                
                matches.append({
                    'student_id': student['student_id'],
                    'job_id': job['job_id'],
                    'match_score': match_score,
                    'is_match': is_match,
                    'gender': student['gender'],
                    'region': student['region'],
                    'background': student['background'],
                    'company_size': job['company_size'],
                })
        
        return pd.DataFrame(matches)
    
    def create_feature_matrix(self, students, jobs, matches):
        """Create feature matrix for modeling"""
        data = []
        
        for _, match in matches.iterrows():
            student = students[students['student_id'] == match['student_id']].iloc[0]
            job = jobs[jobs['job_id'] == match['job_id']].iloc[0]
            
            row = {
                'student_id': match['student_id'],
                'job_id': match['job_id'],
                'is_match': match['is_match'],
                'gender': match['gender'],
                'region': match['region'],
                'background': match['background'],
                'company_size': match['company_size'],
                
                # Student features
                'years_exp': student['years_exp'],
                'gpa': student['gpa'],
                'num_projects': student['num_projects'],
                'internships': student['internships'],
                'certifications': student['certifications'],
                
                # Job features
                'salary_min': job['salary_min'],
                'salary_max': job['salary_max'],
                'required_exp_min': job['required_exp_min'],
                'required_exp_max': job['required_exp_max'],
                'required_gpa_min': job['required_gpa_min'],
                'urgency_level': job['urgency_level'],
                
                # Skill matching features
                'python_match': min(student['verified_python'], job['req_python']) / 100 if max(student['verified_python'], job['req_python']) > 0 else 0,
                'javascript_match': min(student['verified_javascript'], job['req_javascript']) / 100 if max(student['verified_javascript'], job['req_javascript']) > 0 else 0,
                'sql_match': min(student['verified_sql'], job['req_sql']) / 100 if max(student['verified_sql'], job['req_sql']) > 0 else 0,
                'aws_match': min(student['verified_aws'], job['req_aws']) / 100 if max(student['verified_aws'], job['req_aws']) > 0 else 0,
                
                # Computed features
                'total_score': match['match_score'],
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate(self):
        """Generate complete dataset"""
        print("Generating synthetic data...")
        students = self.generate_students()
        jobs = self.generate_jobs()
        matches = self.generate_matches(students, jobs)
        features = self.create_feature_matrix(students, jobs, matches)
        
        print(f"✓ Generated {len(students)} students")
        print(f"✓ Generated {len(jobs)} jobs")
        print(f"✓ Generated {len(features)} student-job comparisons")
        print(f"✓ Match rate: {features['is_match'].sum() / len(features) * 100:.1f}%")
        
        return students, jobs, features, matches
    
    def save_data(self, students, jobs, features, matches, output_dir='data'):
        """Save generated data to CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        students.to_csv(f'{output_dir}/students.csv', index=False)
        jobs.to_csv(f'{output_dir}/jobs.csv', index=False)
        features.to_csv(f'{output_dir}/features.csv', index=False)
        matches.to_csv(f'{output_dir}/matches.csv', index=False)
        
        print(f"\n✓ Data saved to {output_dir}/")


if __name__ == "__main__":
    generator = DataGenerator(n_samples=1000)
    students, jobs, features, matches = generator.generate()
    generator.save_data(students, jobs, features, matches)
    
    # Print data samples
    print("\n--- Sample Student Data ---")
    print(students.head(3))
    print("\n--- Sample Feature Matrix (first 3 rows) ---")
    print(features.head(3))
