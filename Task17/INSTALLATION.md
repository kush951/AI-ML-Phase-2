# PlaceMux Installation & Setup Guide

## Quick Installation (5 minutes)

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python main.py
```

This will:
- ✓ Generate sample data (100 students, 50 jobs, 200 matches)
- ✓ Train 4 models (Baseline, Logistic, Random Forest, Gradient Boost)
- ✓ Evaluate on test data with proper train/val/test split
- ✓ Generate recommendations samples
- ✓ Create PDF report
- ✓ Save artifacts

### 3. Start API Server

```bash
python fastapi_server.py
# Server running at: http://localhost:8000
```

### 4. Access Dashboard

Open in browser: **http://localhost:8000/dashboard**

---

## Detailed Setup

### Step 1: Environment Setup

```bash
# Create directory
mkdir placemux-system
cd placemux-system

# Clone or copy files
cp placement_recommendation_pipeline.py .
cp fastapi_server.py .
cp frontend_dashboard.jsx .
cp pdf_report_generator.py .
cp main.py .
cp requirements.txt .

# Create Python environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```python
# Test imports
python -c "
import numpy, pandas, sklearn, fastapi, reportlab
print('✓ All dependencies installed')
"
```

### Step 3: Run Training

```bash
# Generate data, train models, and create report
python main.py --students 100 --jobs 50 --matches 200

# Options:
# --students N       Number of sample students
# --jobs N          Number of sample jobs
# --matches N       Number of student-job matches
# --output-dir PATH Output directory for artifacts
```

**Output files created:**
- `model_comparison.json` - Model metrics comparison
- `experiment_log.json` - Training history
- `deployment_summary.txt` - Deployment checklist
- `PlaceMux_Recommendation_Report_v1.pdf` - Full evaluation report

### Step 4: Start API Server

```bash
# Terminal 1: Start server
python fastapi_server.py

# Output:
# INFO:     Started server process [xxxxx]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Check health:**
```bash
curl http://localhost:8000/health
```

### Step 5: Run Tests & Demos

```bash
# Terminal 2: Run demo
python demo_and_integration.py --demo all

# Or run specific demo:
python demo_and_integration.py --demo basic      # Basic workflow
python demo_and_integration.py --demo batch      # Batch recommendations
python demo_and_integration.py --demo metrics    # Model comparison
python demo_and_integration.py --demo performance  # Performance testing
```

### Step 6: Frontend Integration

#### Option A: React App Integration

```jsx
// In your React component
import PlaceMuxDashboard from './frontend_dashboard.jsx';

export default function App() {
  return (
    <div>
      <PlaceMuxDashboard />
    </div>
  );
}
```

#### Option B: Standalone HTML

```html
<!-- Access directly in browser -->
<a href="http://localhost:8000/dashboard">Open PlaceMux Dashboard</a>
```

#### Option C: Embed in Existing Dashboard

```html
<iframe 
  src="http://localhost:8000/dashboard" 
  width="100%" 
  height="800px"
  title="PlaceMux Placement Dashboard">
</iframe>
```

---

## Configuration

### API Server Settings

Edit `fastapi_server.py`:

```python
# Change port
uvicorn.run(app, host="0.0.0.0", port=8001)

# Enable CORS for cross-origin requests
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/recommend", dependencies=[Depends(security)])
async def get_recommendation(...):
    ...
```

### Model Configuration

Edit `placement_recommendation_pipeline.py`:

```python
# Use specific models
recommender = HybridRecommender(
    use_models=['random_forest', 'gradient_boost']
    # Or: ['baseline', 'logistic', 'random_forest', 'gradient_boost']
)

# Adjust hyperparameters
model_configs = {
    'random_forest': RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,      # Deeper trees
        random_state=42
    )
}
```

### Data Configuration

Edit `main.py`:

```python
# Generate more data for better training
pipeline.step_1_generate_data(
    n_students=500,   # 5x more students
    n_jobs=200,       # 4x more jobs
    n_matches=5000    # 25x more matches
)

# Use real CSV data
import pandas as pd
student_data = pd.read_csv('students.csv')
job_data = pd.read_csv('jobs.csv')
matches = pd.read_csv('matches.csv')
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Issue: "Port 8000 already in use"

```bash
# Use different port
python fastapi_server.py --port 8001

# Or kill existing process
lsof -i :8000  # Find process ID
kill -9 <PID>
```

### Issue: "Memory error with large dataset"

```python
# Generate smaller dataset
python main.py --students 50 --jobs 25 --matches 100

# Or use chunked training
recommender.fit(
    student_data, 
    job_data, 
    matches,
    batch_size=32  # Process in batches
)
```

### Issue: "Frontend not loading"

```bash
# Check if server is running
curl http://localhost:8000/health

# Check browser console for CORS errors
# If CORS issue, enable in fastapi_server.py

# Try accessing directly
http://localhost:8000/dashboard
```

### Issue: "Model not found" error

```python
# Verify training completed
import json
with open('outputs/model_comparison.json') as f:
    data = json.load(f)
    print(f"Best model: {data['best_model']}")

# Retrain if needed
python main.py --output-dir ./fresh_outputs
```

---

## Performance Optimization

### For Faster Training

```python
# Use fewer models
recommender = HybridRecommender(
    use_models=['gradient_boost']  # Only best model
)

# Reduce data
n_samples = 100  # Instead of 200

# Use simpler models first
models = ['baseline', 'logistic']  # Skip ensemble for exploration
```

### For Faster Inference

```python
# Load model once
import pickle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Batch multiple requests
recommendations = []
for (student, job) in student_job_pairs:
    rec = recommender.get_recommendation(...)
    recommendations.append(rec)
```

### For Better Scaling

```python
# Use Redis for caching
from redis import Redis
redis = Redis()

def get_recommendation_cached(student_id, job_id):
    key = f"{student_id}:{job_id}"
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    # ... generate recommendation
    redis.setex(key, 3600, json.dumps(result))  # Cache 1 hour
    return result
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY placement_recommendation_pipeline.py .
COPY fastapi_server.py .
COPY pdf_report_generator.py .
COPY main.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  placemux-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - LOG_LEVEL=INFO
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped

  # Optional: Add frontend service
  placemux-frontend:
    image: node:16
    working_dir: /app
    volumes:
      - ./frontend:/app
    ports:
      - "3000:3000"
    command: npm start
```

### Deploy with Docker

```bash
# Build image
docker build -t placemux:v1 .

# Run container
docker run -p 8000:8000 placemux:v1

# Or use docker-compose
docker-compose up -d

# Check logs
docker logs <container_id>

# Stop container
docker stop <container_id>
```

---

## Monitoring & Logging

### Enable Debug Logging

```bash
# Verbose output
LOGLEVEL=DEBUG python fastapi_server.py

# Save to file
python fastapi_server.py > logs/server.log 2>&1 &
```

### Monitor Metrics

```bash
# Check system health
curl http://localhost:8000/health | jq

# Get model metrics
curl http://localhost:8000/metrics | jq

# Monitor API usage (in logs)
tail -f logs/server.log | grep "POST /recommend"
```

### Performance Monitoring

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper

@timing_decorator
def get_recommendation(...):
    ...
```

---

## Production Checklist

Before deploying to production:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Training pipeline completed successfully
- [ ] Models evaluated on test data
- [ ] PDF report generated
- [ ] API server tested locally
- [ ] Dashboard UI verified
- [ ] Error handling tested
- [ ] Performance benchmarks met (<100ms per request)
- [ ] CORS settings configured
- [ ] Logging enabled
- [ ] Database/storage configured (if using)
- [ ] SSL/HTTPS configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy in place
- [ ] Documentation complete

---

## Quick Commands Reference

```bash
# Installation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Training
python main.py

# API Server
python fastapi_server.py

# Demo
python demo_and_integration.py --demo all

# Testing
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Docker
docker build -t placemux:v1 .
docker run -p 8000:8000 placemux:v1

# Cleanup
deactivate
rm -rf venv/
rm -rf __pycache__ *.pyc
```

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs: `logs/server.log`
3. Check model comparison: `outputs/model_comparison.json`
4. Review PDF report: `outputs/PlaceMux_Recommendation_Report_v1.pdf`

---

**Ready to deploy? Start with: `python main.py` then `python fastapi_server.py`**
