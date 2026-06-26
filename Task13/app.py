"""
PlaceMux Proctoring FP Reduction - Flask API
REST API for serving FP reduction model predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from ml_pipeline import FPReductionPipeline
import os

app = Flask(__name__)
CORS(app)

# Initialize pipeline
pipeline = FPReductionPipeline()
print("Training model...")
best_name, best_model, best_metrics = pipeline.train_pipeline()

# Store metrics for reporting
MODEL_METRICS = {
    'model_name': best_name,
    'validation_metrics': pipeline.best_metrics,
    'test_metrics': pipeline.test_metrics,
}

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_METRICS['model_name'],
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for a proctoring session
    
    Expected JSON:
    {
        "skill_match": 0.85,
        "session_duration": 0.6,
        "camera_available": 1,
        "env_quality": 0.9,
        "verification_confidence": 0.88,
        "completion_pct": 0.95,
        "answer_consistency": 0.92,
        "device_stability": 0.87
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = pipeline.feature_names
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields. Expected: {required_fields}'
            }), 400
        
        # Get explanation
        explanation = pipeline.explain_prediction(data)
        
        return jsonify({
            'success': True,
            'prediction': explanation['prediction'],
            'fraud_probability': explanation['fraud_probability'],
            'confidence': explanation['confidence'],
            'risk_factors': explanation['risk_factors'],
            'positive_factors': explanation['positive_factors'],
            'recommended_action': 'FLAG FOR REVIEW' if explanation['fraud_probability'] > 0.6 else 'ACCEPT',
            'risk_level': 'HIGH' if explanation['fraud_probability'] > 0.7 else 
                         'MEDIUM' if explanation['fraud_probability'] > 0.4 else 'LOW'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict fraud probability for multiple sessions"""
    try:
        data = request.get_json()
        
        if 'sessions' not in data:
            return jsonify({'error': 'Expected "sessions" array in request'}), 400
        
        results = []
        for session in data['sessions']:
            explanation = pipeline.explain_prediction(session)
            results.append({
                'session_id': session.get('session_id', 'unknown'),
                'prediction': explanation['prediction'],
                'fraud_probability': explanation['fraud_probability'],
                'confidence': explanation['confidence'],
                'risk_level': 'HIGH' if explanation['fraud_probability'] > 0.7 else 
                             'MEDIUM' if explanation['fraud_probability'] > 0.4 else 'LOW'
            })
        
        return jsonify({
            'success': True,
            'total_sessions': len(results),
            'flagged_sessions': sum(1 for r in results if r['risk_level'] == 'HIGH'),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    return jsonify({
        'model_name': MODEL_METRICS['model_name'],
        'validation': {
            'precision': float(MODEL_METRICS['validation_metrics']['precision']),
            'recall': float(MODEL_METRICS['validation_metrics']['recall']),
            'f1_score': float(MODEL_METRICS['validation_metrics']['f1']),
            'roc_auc': float(MODEL_METRICS['validation_metrics']['roc_auc']),
            'false_positive_rate': float(MODEL_METRICS['validation_metrics']['false_positive_rate']),
            'false_negative_rate': float(MODEL_METRICS['validation_metrics']['false_negative_rate']),
            'true_positives': int(MODEL_METRICS['validation_metrics']['true_positives']),
            'false_positives': int(MODEL_METRICS['validation_metrics']['false_positives']),
            'true_negatives': int(MODEL_METRICS['validation_metrics']['true_negatives']),
            'false_negatives': int(MODEL_METRICS['validation_metrics']['false_negatives']),
            'accuracy': float(MODEL_METRICS['validation_metrics']['accuracy'])
        },
        'test': {
            'precision': float(MODEL_METRICS['test_metrics']['precision']),
            'recall': float(MODEL_METRICS['test_metrics']['recall']),
            'f1_score': float(MODEL_METRICS['test_metrics']['f1']),
            'roc_auc': float(MODEL_METRICS['test_metrics']['roc_auc']),
            'false_positive_rate': float(MODEL_METRICS['test_metrics']['false_positive_rate']),
            'false_negative_rate': float(MODEL_METRICS['test_metrics']['false_negative_rate']),
            'accuracy': float(MODEL_METRICS['test_metrics']['accuracy'])
        }
    })

@app.route('/api/feature-info', methods=['GET'])
def feature_info():
    """Get information about features used by the model"""
    return jsonify({
        'features': pipeline.feature_names,
        'feature_descriptions': {
            'skill_match': 'Overlap between verified skills and job requirements (0-1)',
            'session_duration': 'Time spent in proctoring session normalized (0-1)',
            'camera_available': 'Whether camera was available (0/1)',
            'env_quality': 'Quality of environment for proctoring (0-1)',
            'verification_confidence': 'System confidence in verification (0-1)',
            'completion_pct': 'Percentage of session completed (0-1)',
            'answer_consistency': 'Consistency of answers with past behavior (0-1)',
            'device_stability': 'Stability of device fingerprint (0-1)'
        },
        'total_features': len(pipeline.feature_names)
    })

@app.route('/api/example-cases', methods=['GET'])
def example_cases():
    """Get example test cases"""
    return jsonify({
        'cases': [
            {
                'id': 'case_1_legitimate',
                'description': 'High-confidence legitimate session',
                'data': {
                    'skill_match': 0.85,
                    'session_duration': 0.6,
                    'camera_available': 1,
                    'env_quality': 0.9,
                    'verification_confidence': 0.88,
                    'completion_pct': 0.95,
                    'answer_consistency': 0.92,
                    'device_stability': 0.87
                }
            },
            {
                'id': 'case_2_suspicious',
                'description': 'Suspicious session with fraud indicators',
                'data': {
                    'skill_match': 0.3,
                    'session_duration': 0.15,
                    'camera_available': 0,
                    'env_quality': 0.7,
                    'verification_confidence': 0.75,
                    'completion_pct': 0.85,
                    'answer_consistency': 0.25,
                    'device_stability': 0.6
                }
            },
            {
                'id': 'case_3_borderline',
                'description': 'Borderline case requiring human review',
                'data': {
                    'skill_match': 0.65,
                    'session_duration': 0.45,
                    'camera_available': 1,
                    'env_quality': 0.65,
                    'verification_confidence': 0.55,
                    'completion_pct': 0.88,
                    'answer_consistency': 0.65,
                    'device_stability': 0.72
                }
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
