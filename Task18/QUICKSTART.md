# 🚀 PlaceMux Quick Start Guide

## 5-Minute Setup

### Step 1: Start Backend (Terminal 1)
```bash
cd placemux_project/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r ../requirements.txt

# Run server
python main.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start Frontend (Terminal 2)
```bash
cd placemux_project/frontend

# Install dependencies
npm install

# Start dev server
npm start
```

**Browser opens to:** `http://localhost:3000`

### Step 3: Access Admin Console
1. Navigate to http://localhost:3000
2. Job Matching tab loads with sample data (John Doe → Senior Python Developer)
3. Click "🚀 Predict Match"
4. View prediction: 82% match with explanation

## API Quick Test

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Available Models
```bash
curl http://localhost:8000/models/available
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/match/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student": {
      "student_id": "ST_001",
      "name": "Alice Smith",
      "verified_skills": ["Python", "React", "Node.js"],
      "skill_scores": {"Python": 0.9, "React": 0.85, "Node.js": 0.8},
      "gpa": 3.9,
      "experience_years": 2,
      "college_id": "College_A"
    },
    "job": {
      "job_id": "JB_002",
      "title": "Full Stack Developer",
      "company": "Tech Startup",
      "required_skills": ["Python", "React", "Node.js"],
      "required_exp_years": 2,
      "salary_range": "$120000-$160000",
      "college_id": "College_A"
    }
  }'
```

### View Metrics
```bash
curl http://localhost:8000/metrics/comparison
```

## Files Overview

### Backend Structure
- `main.py` - FastAPI application with all endpoints
- `models/trainer.py` - Multi-model training logic
- `models/explainability.py` - Explanation generation
- `data/sample_data.py` - Sample data generation
- `config.py` - Configuration settings

### Frontend Structure
- `src/components/AdminConsole.jsx` - Main UI component
- `src/components/AdminConsole.css` - Styling
- `src/App.jsx` - React app wrapper

### Documentation
- `README.md` - Complete documentation
- `MODEL_EVALUATION_REPORT.pdf` - Detailed metrics report
- `generate_report.py` - PDF report generator

## Common Tasks

### Train Models Fresh
```python
from backend.data.sample_data import load_sample_data
from backend.models.trainer import ModelTrainer

X_train, X_test, y_train, y_test, features = load_sample_data(n_samples=500)
trainer = ModelTrainer(X_train, y_train, X_test, y_test, features)
print(f"Best Model: {trainer.best_model_name}")
print(f"Best F1 Score: {trainer.model_metrics[trainer.best_model_name]['f1_score']:.4f}")
```

### Generate PDF Report
```bash
python generate_report.py
# Output: /mnt/user-data/outputs/MODEL_EVALUATION_REPORT.pdf
```

### Run API Tests
```bash
pytest backend/tests/ -v
```

## Troubleshooting

**Issue: Port 8000 already in use**
```bash
lsof -i :8000  # Find process
kill -9 <PID>   # Kill it
# Or specify different port:
python main.py --port 8001
```

**Issue: Frontend can't connect to backend**
- Check CORS in `backend/main.py` line ~45
- Ensure API_BASE in AdminConsole.jsx matches your backend URL
- Check browser console for specific errors

**Issue: Models not training**
- Ensure scikit-learn installed: `pip install scikit-learn numpy pandas`
- Check data loading in `backend/data/sample_data.py`

## Next Steps

1. ✅ Understand the architecture (see README.md)
2. ✅ Review metrics in MODEL_EVALUATION_REPORT.pdf
3. ✅ Test different student/job profiles in admin console
4. ✅ Check explainability explanations
5. ✅ Deploy to production (Docker config in README)

## Support

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Full Documentation**: See README.md
- **Model Details**: See MODEL_EVALUATION_REPORT.pdf

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: June 28, 2024
