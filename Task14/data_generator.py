
import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data():
    """Generate realistic sample data for skill matching"""

    # ------------------------------------------------------------
    # RESUMES
    # ------------------------------------------------------------

    resumes = pd.DataFrame({
        'candidate_id': [
            'C001', 'C002', 'C003', 'C004',
            'C005', 'C006', 'C007', 'C008'
        ],

        'resume_text': [
            'Python Django REST APIs PostgreSQL backend developer 5 years',
            'Machine Learning TensorFlow NLP PyTorch Data Analysis 3 years',
            'React JavaScript CSS HTML Frontend Vue.js developer 4 years',
            'Java Spring Boot Microservices AWS Docker 6 years',
            'Python SQL Statistics Tableau Data Science 5 years',
            'Docker Kubernetes Jenkins AWS DevOps CI/CD 4 years',
            'React Node.js MongoDB Express Full Stack 3 years',
            'AWS Azure GCP Cloud Architecture Infrastructure 7 years'
        ],

        'verified_skills': [
            'Python,Django,REST APIs,PostgreSQL',
            'Machine Learning,TensorFlow,NLP',
            'React,JavaScript,CSS',
            'Java,Spring Boot,Docker',
            'Python,SQL,Statistics',
            'Docker,Kubernetes,Jenkins',
            'React,Node.js,MongoDB',
            'AWS,Cloud Architecture'
        ],

        'experience_years': [5, 3, 4, 6, 5, 4, 3, 7]
    })

    # ------------------------------------------------------------
    # JOBS
    # ------------------------------------------------------------

    jobs = pd.DataFrame({
        'job_id': [
            'J001', 'J002', 'J003', 'J004',
            'J005', 'J006', 'J007', 'J008'
        ],

        'job_title': [
            'Senior Python Developer',
            'ML Engineer',
            'React Frontend Developer',
            'Java Backend Engineer',
            'Data Scientist',
            'DevOps Engineer',
            'Full Stack Developer',
            'Cloud Architect'
        ],

        'job_description': [
            'Python Django REST APIs PostgreSQL AWS',
            'Machine Learning TensorFlow PyTorch NLP',
            'React JavaScript CSS Vue.js',
            'Java Spring Boot Microservices Docker',
            'Python SQL Statistics Tableau',
            'Docker Kubernetes Jenkins CI/CD',
            'React Node.js MongoDB Express',
            'AWS Cloud Architecture Azure GCP'
        ],

        'required_skills': [
            'Python,Django,REST APIs,PostgreSQL',
            'Machine Learning,TensorFlow,PyTorch',
            'React,JavaScript,CSS',
            'Java,Spring Boot,Microservices',
            'Python,SQL,Statistics',
            'Docker,Kubernetes,Jenkins',
            'React,Node.js,MongoDB',
            'AWS,Cloud Architecture'
        ],

        'experience_required': [5, 3, 4, 6, 5, 4, 3, 7]
    })

    # ------------------------------------------------------------
    # POSITIVE MATCHES
    # ------------------------------------------------------------

    positive_matches = pd.DataFrame({
        'candidate_id': [
            'C001', 'C002', 'C003', 'C004',
            'C005', 'C006', 'C007', 'C008'
        ],

        'job_id': [
            'J001', 'J002', 'J003', 'J004',
            'J005', 'J006', 'J007', 'J008'
        ],

        'match_score': [
            0.95, 0.88, 0.92, 0.90,
            0.87, 0.89, 0.91, 0.93
        ],

        'is_match': [1, 1, 1, 1, 1, 1, 1, 1]
    })

    # ------------------------------------------------------------
    # NEGATIVE MATCHES
    # ------------------------------------------------------------

    negative_matches = pd.DataFrame({
        'candidate_id': [
            'C001', 'C002', 'C003', 'C004',
            'C005', 'C006', 'C007', 'C008'
        ],

        'job_id': [
            'J005', 'J006', 'J007', 'J008',
            'J001', 'J002', 'J003', 'J004'
        ],

        'match_score': [
            0.20, 0.15, 0.25, 0.18,
            0.22, 0.12, 0.28, 0.10
        ],

        'is_match': [0, 0, 0, 0, 0, 0, 0, 0]
    })

    # ------------------------------------------------------------
    # COMBINE MATCHES
    # ------------------------------------------------------------

    matches = pd.concat(
        [positive_matches, negative_matches],
        ignore_index=True
    )

    # Shuffle rows
    matches = matches.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # SAVE FILES
    # ------------------------------------------------------------

    Path('data').mkdir(exist_ok=True)

    resumes.to_csv('data/resumes.csv', index=False)
    jobs.to_csv('data/jobs.csv', index=False)
    matches.to_csv('data/matches.csv', index=False)

    print("✓ Sample data generated successfully")
    print(f"✓ Total matches: {len(matches)}")
    print("\nClass Distribution:")
    print(matches['is_match'].value_counts())

    return resumes, jobs, matches


if __name__ == '__main__':
    generate_sample_data()

