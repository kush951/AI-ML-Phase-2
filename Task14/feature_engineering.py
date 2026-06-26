import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re

class SkillExtractor:
    """Extract and normalize skills from text"""
    
    def __init__(self):
        self.skill_ontology = {
            'programming_languages': ['python', 'java', 'javascript', 'cpp', 'csharp', 'go', 'rust', 'kotlin'],
            'frameworks': ['django', 'flask', 'spring', 'spring boot', 'react', 'vue', 'angular', 'express', 'fastapi'],
            'databases': ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'kubernetes', 'docker'],
            'ml_tools': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'xgboost'],
            'data_tools': ['pandas', 'numpy', 'spark', 'hadoop', 'tableau'],
            'devops': ['jenkins', 'gitlab', 'github', 'ci/cd', 'docker', 'kubernetes'],
            'soft_skills': ['communication', 'leadership', 'teamwork', 'problem-solving']
        }
        
        self.flattened_skills = {skill for skills in self.skill_ontology.values() for skill in skills}
    
    def extract_skills(self, text):
        """Extract recognized skills from text"""
        text_lower = text.lower()
        extracted = set()
        
        for skill in self.flattened_skills:
            if skill in text_lower:
                extracted.add(skill)
        
        return list(extracted)
    
    def normalize_skill(self, skill):
        """Normalize skill name"""
        skill = skill.lower().strip()
        skill = re.sub(r'[^\w\s-]', '', skill)
        return skill
    
    def skill_overlap(self, skills1, skills2):
        """Calculate skill overlap ratio"""
        set1 = set(self.normalize_skill(s) for s in skills1)
        set2 = set(self.normalize_skill(s) for s in skills2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class FeatureEngineer:
    """Generate features for matching model"""
    
    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    
    def create_features(self, resumes_df, jobs_df):
        """Create feature matrix for all resume-job pairs"""
        
        features_list = []
        
        for _, resume in resumes_df.iterrows():
            for _, job in jobs_df.iterrows():
                features = self._create_pair_features(resume, job)
                features['candidate_id'] = resume['candidate_id']
                features['job_id'] = job['job_id']
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _create_pair_features(self, resume, job):
        """Create features for a single resume-job pair"""
        
        resume_skills = self.skill_extractor.extract_skills(resume['verified_skills'])
        job_skills = self.skill_extractor.extract_skills(job['required_skills'])
        
        # Skill-based features
        skill_overlap = self.skill_extractor.skill_overlap(resume_skills, job_skills)
        required_skills_match = len(set(resume_skills) & set(job_skills)) / max(len(set(job_skills)), 1)
        
        # Experience features
        exp_diff = abs(resume['experience_years'] - job['experience_required'])
        exp_match = 1.0 if exp_diff <= 1 else max(0.0, 1.0 - (exp_diff / 10.0))
        
        # Text similarity
        resume_text = resume['verified_skills']
        job_text = job['required_skills']
        
        try:
            tfidf_matrix = self.tfidf.fit_transform([resume_text, job_text])
            text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            text_similarity = 0.0
        
        return {
            'skill_overlap': skill_overlap,
            'required_skills_match': required_skills_match,
            'exp_match': exp_match,
            'text_similarity': text_similarity,
            'exp_required': job['experience_required'],
            'num_required_skills': len(set(job_skills)),
            'num_candidate_skills': len(set(resume_skills))
        }


def prepare_training_data(resumes_df, jobs_df, matches_df):
    """Prepare complete training dataset"""
    
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_features(resumes_df, jobs_df)
    
    # Merge with ground truth
    training_data = features_df.merge(
        matches_df[['candidate_id', 'job_id', 'is_match', 'match_score']],
        on=['candidate_id', 'job_id'],
        how='left'
    )
    
    return training_data, feature_engineer
