# PlaceMux Deployment Guide

## Quick Start (5 minutes)

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser

### Step 1: Install Dependencies
```bash
cd placemux_project
pip install -r requirements.txt
```

Expected packages:
- flask==2.3.2
- flask-cors==4.0.0
- scikit-learn==1.3.0
- numpy==1.24.3
- pandas==2.0.3
- reportlab==4.0.4

### Step 2: Start the ML Backend
```bash
python ml_pipeline.py
```

This will:
1. Generate 2000 synthetic proctoring sessions
2. Train 3 models (Baseline, Random Forest, Gradient Boosting)
3. Evaluate and select the best model
4. Display comprehensive metrics

Expected output includes:
```
======================================================================
PlaceMux Proctoring FP Reduction - Model Training Pipeline
======================================================================

1. Generating synthetic proctoring data...
   - Total samples: 2000
   - Fraud cases: 285 (14.2%)

2. Preparing data...
   - Training set: 1280 samples
   - Validation set: 320 samples
   - Test set: 400 samples

3. Training models...

Building Baseline Model (Logistic Regression):
  Model: Baseline (Logistic Regression)
  Precision:    0.8920
  Recall:       0.8510
  F1 Score:     0.8713
  ...

Building Random Forest Model:
  [metrics]

Building Gradient Boosting Model:
  [metrics]

======================================================================
BEST MODEL SELECTED: GRADIENT_BOOSTING
======================================================================
False Positive Rate: 0.0210
False Negatives: 0.1150
F1 Score: 0.9120
...

4. Final evaluation on test set...
```

**Note:** This script can be run once to train the model, or repeatedly to validate consistency.

### Step 3: Start the Flask API Server
```bash
python app.py
```

Expected output:
```
[INFO] Training model...
... [model training output] ...
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

**Important:** Keep this terminal open while using the system.

### Step 4: Open the Web Dashboard
Open your browser and navigate to:
```
file:///path/to/placemux_project/index.html
```

Or better, serve via HTTP in another terminal:
```bash
# Terminal 3
cd placemux_project
python -m http.server 8000
```

Then navigate to:
```
http://localhost:8000/index.html
```

You should see:
- PlaceMux header
- Session Analysis form with 8 sliders
- Analysis Results panel
- Model Performance Metrics

---

## System Architecture for Deployment

```
User's Browser
    ↓
index.html (Frontend)
    ↓ (fetch API calls)
    ↓
http://localhost:5000
    ↓
Flask API (app.py)
    ↓ (calls methods)
    ↓
ML Pipeline (ml_pipeline.py)
    ↓ (trained models)
    ↓
Prediction + Explanation
    ↓ (JSON response)
    ↓
Browser displays results
```

---

## Testing the System

### Manual Testing

1. **Use Example Cases**
   - Click "Examples" tab in frontend
   - Click any case to load it
   - System auto-analyzes
   - Compare results to expected outcomes

2. **Manipulate Inputs**
   - Drag sliders to extreme values
   - Observe how predictions change
   - Verify factors match your input changes

3. **Check API Directly**
   ```bash
   # In another terminal
   curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{
       "skill_match": 0.85,
       "session_duration": 0.6,
       "camera_available": 1,
       "env_quality": 0.9,
       "verification_confidence": 0.88,
       "completion_pct": 0.95,
       "answer_consistency": 0.92,
       "device_stability": 0.87
     }'
   ```
   
   Expected response:
   ```json
   {
     "success": true,
     "prediction": "LEGITIMATE",
     "fraud_probability": 0.05,
     "confidence": 0.90,
     "risk_level": "LOW",
     "recommended_action": "ACCEPT",
     "risk_factors": [],
     "positive_factors": [...]
   }
   ```

4. **Check Metrics**
   ```bash
   curl http://localhost:5000/api/metrics | python -m json.tool
   ```

---

## Production Deployment

### Moving Beyond Development

1. **Use Production Web Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Add Authentication**
   ```python
   # In app.py, add before routes:
   from functools import wraps
   from flask import request
   
   def require_api_key(f):
       @wraps(f)
       def decorated_function(*args, **kwargs):
           api_key = request.headers.get('X-API-Key')
           if not api_key or api_key != os.environ.get('API_KEY'):
               return {'error': 'Invalid API key'}, 401
           return f(*args, **kwargs)
       return decorated_function
   
   @app.route('/api/predict', methods=['POST'])
   @require_api_key
   def predict():
       # ... existing code ...
   ```

3. **Add Database Persistence**
   ```python
   # Log all predictions to database
   from sqlalchemy import create_engine
   
   engine = create_engine('postgresql://user:pass@localhost/placemux')
   
   # In predict():
   prediction_log = {
       'timestamp': datetime.now(),
       'input_data': data,
       'prediction': prediction,
       'fraud_probability': probability
   }
   engine.execute(PredictionLog.__table__.insert(), prediction_log)
   ```

4. **Monitor Model Performance**
   ```python
   # Track metrics over time
   def log_metrics():
       """Compare current metrics to baseline"""
       current = evaluate_on_fresh_data()
       if current['fp_rate'] > baseline['fp_rate'] * 1.1:
           send_alert("Model FP rate degraded!")
   ```

5. **Add Monitoring & Alerting**
   - Use DataDog, Prometheus, or New Relic
   - Track: API latency, error rate, prediction volume
   - Alert on: FP rate change, model drift, API downtime

6. **Containerize with Docker**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 5000
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

   Build and run:
   ```bash
   docker build -t placemux-fp-reduction .
   docker run -p 5000:5000 placemux-fp-reduction
   ```

7. **Implement Offer Signing**
   ```python
   from cryptography.hazmat.primitives import hashes
   from cryptography.hazmat.primitives.asymmetric import rsa, padding
   
   @app.route('/api/sign-offer', methods=['POST'])
   def sign_offer():
       offer_data = request.json
       
       # Only sign if FP reduced prediction
       if offer_data['model_confidence'] > 0.85:
           signature = sign_with_private_key(offer_data)
           persist_to_ledger(offer_data, signature)
           return {'signature': signature, 'verified': True}
   ```

8. **Interview Scheduling Integration**
   ```python
   @app.route('/api/schedule-interview', methods=['POST'])
   def schedule_interview():
       candidate = request.json['candidate']
       job = request.json['job']
       
       # Get FP-reduced prediction
       prediction = get_prediction(candidate, job)
       
       if prediction['fraud_probability'] < 0.05:
           # High confidence - auto-schedule
           schedule_interview_auto(candidate, job, slots='next_week')
           return {'scheduled': True, 'confidence': 'HIGH'}
       elif prediction['fraud_probability'] < 0.15:
           # Medium confidence - flag for review
           flag_for_human_review(candidate)
           return {'scheduled': False, 'confidence': 'MEDIUM', 'action': 'FLAG_FOR_REVIEW'}
   ```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
pip install flask flask-cors
```

### Problem: "Address already in use" when starting Flask

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use a different port
python app.py --port=5001
```

### Problem: CORS errors in browser console

**Solution:**
Ensure Flask-CORS is installed and initialized:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
```

### Problem: Predictions are inconsistent

**Solution:**
Check that you're using the same `random_state`:
```python
pipeline = FPReductionPipeline(random_state=42)
```

### Problem: API returns 500 error

**Solution:**
Check Flask terminal for error output. Common causes:
- Missing input fields
- Out-of-range values
- Model not trained yet

---

## Performance Optimization

### For High-Volume Predictions

1. **Batch Processing**
   ```bash
   POST /api/batch-predict
   ```
   Can handle 100+ predictions in single request.

2. **Caching**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   
   @app.route('/api/metrics')
   @cache.cached(timeout=300)
   def get_metrics():
       # Cached for 5 minutes
   ```

3. **Model Quantization**
   - Save model in compressed format
   - Reduces memory footprint
   - Faster inference on edge devices

4. **Load Balancing**
   - Deploy multiple instances
   - Use nginx or AWS Load Balancer
   - Handle thousands of requests/second

---

## File Reference

| File | Purpose |
|------|---------|
| `ml_pipeline.py` | Core ML pipeline - trains and evaluates models |
| `app.py` | Flask REST API server |
| `index.html` | Interactive web dashboard |
| `README.md` | Comprehensive documentation |
| `REPORT.pdf` | Performance metrics and technical analysis |
| `requirements.txt` | Python dependencies |
| `generate_report.py` | PDF report generator |

---

## Support

For issues:
1. Check README.md for detailed documentation
2. Review REPORT.pdf for performance metrics
3. Check Flask terminal output for error messages
4. Review code comments for implementation details

---

**PlaceMux is ready for deployment! 🚀**
