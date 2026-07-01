"""
PlaceMux - Feature Store
Centralized feature management and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os
from typing import Dict, List, Tuple

class FeatureStore:
    """
    Feature Store: A centralized repository for all features used in ML models.
    Ensures consistency between training and inference.
    """
    
    def __init__(self, store_path='feature_store/'):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_metadata = {}
        self.categorical_features = []
        self.numerical_features = []
        
    def define_features(self, df: pd.DataFrame) -> Dict:
        """
        Define which columns are features and their types
        """
        self.numerical_features = [
            'years_experience', 'gpa', 'certifications', 'projects_completed',
            'aptitude_score', 'company_rating', 'salary_range_min', 'salary_range_max',
            'min_experience', 'min_gpa', 'required_certifications'
        ]
        
        # Add all skill columns
        skill_cols = [col for col in df.columns if col.startswith('skill_')]
        self.numerical_features.extend(skill_cols)
        
        # Get features that exist in dataframe
        self.feature_columns = [f for f in self.numerical_features if f in df.columns]
        
        self.feature_metadata = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'all_features': self.feature_columns,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        return self.feature_metadata
    
    def preprocess_features(self, df: pd.DataFrame, fit=True) -> pd.DataFrame:
        """
        Preprocess features: handle missing values, scale, etc.
        """
        df_processed = df.copy()
        
        # Handle missing values
        for col in self.feature_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Scale numerical features
        if fit:
            df_processed[self.feature_columns] = self.scaler.fit_transform(
                df_processed[self.feature_columns]
            )
        else:
            df_processed[self.feature_columns] = self.scaler.transform(
                df_processed[self.feature_columns]
            )
        
        return df_processed
    
    def create_feature_vector(self, student_data: Dict, job_data: Dict) -> np.ndarray:
        """
        Create a feature vector for a student-job pair during inference
        """
        features = []
        
        for col in self.feature_columns:
            # Try to get from student data first, then job data
            if col in student_data:
                features.append(student_data[col])
            elif col in job_data:
                features.append(job_data[col])
            else:
                features.append(0)  # Default if missing
        
        # Scale the features
        features_array = np.array(features).reshape(1, -1)
        scaled_features = self.scaler.transform(features_array)
        
        return scaled_features[0]
    
    def save_feature_store(self):
        """Save feature store to disk"""
        # Save scaler
        with open(f'{self.store_path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        with open(f'{self.store_path}metadata.pkl', 'wb') as f:
            pickle.dump(self.feature_metadata, f)
        
        print(f"✓ Feature store saved to {self.store_path}")
        print(f"  - Total features: {len(self.feature_columns)}")
    
    def load_feature_store(self):
        """Load feature store from disk"""
        if not os.path.exists(f'{self.store_path}scaler.pkl'):
            raise FileNotFoundError("Feature store not found. Train the model first.")
        
        with open(f'{self.store_path}scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{self.store_path}metadata.pkl', 'rb') as f:
            self.feature_metadata = pickle.load(f)
        
        self.feature_columns = self.feature_metadata.get('all_features', [])
        self.numerical_features = self.feature_metadata.get('numerical_features', [])
        self.categorical_features = self.feature_metadata.get('categorical_features', [])
        
        print(f"✓ Feature store loaded from {self.store_path}")
        print(f"  - Total features: {len(self.feature_columns)}")
    
    def get_features_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and labels for training
        Returns X (features) and y (labels)
        """
        self.define_features(df)
        
        X = self.preprocess_features(df[self.feature_columns], fit=True)
        y = df['label'] if 'label' in df.columns else None
        
        return X, y
    
    def get_features_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get features for inference (transform only, no fitting)
        """
        X = self.preprocess_features(df[self.feature_columns], fit=False)
        return X


if __name__ == '__main__':
    # Test feature store
    from data_generator import DataGenerator
    
    print("Generating test data...")
    generator = DataGenerator(n_students=100, n_jobs=50)
    generator.generate_students()
    generator.generate_jobs()
    generator.generate_matches()
    training_data = generator.create_training_data()
    
    print("\nInitializing feature store...")
    fs = FeatureStore()
    X, y = fs.get_features_for_training(training_data)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Labels: {y.value_counts().to_dict()}")
    print(f"  - Features: {fs.feature_columns[:5]}... (total: {len(fs.feature_columns)})")
    
    fs.save_feature_store()
