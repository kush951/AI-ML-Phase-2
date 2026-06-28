# PlaceMux API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required (add JWT tokens for production).

## Content-Type
All requests and responses use `application/json`

---

## Endpoints

### 1. Health Check
**Check if API is running**

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-06-28T10:30:00.123456",
  "models_loaded": true
}
```

**Status Codes:**
- `200 OK` - API is running

---

### 2. Get Available Models
**List all trained models and their metrics**

```http
GET /models/available
```

**Response:**
```json
{
  "available_models": [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "Gradient Boosting"
  ],
  "best_model": "Gradient Boosting",
  "metrics": {
    "Logistic Regression": {
      "accuracy": 0.752,
      "precision": 0.728,
      "recall": 0.685,
      "f1_score": 0.705
    },
    "SVM": {
      "accuracy": 0.768,
      "precision": 0.741,
      "recall": 0.702,
      "f1_score": 0.720
    },
    "Random Forest": {
      "accuracy": 0.785,
      "precision": 0.763,
      "recall": 0.731,
      "f1_score": 0.746
    },
    "Gradient Boosting": {
      "accuracy": 0.798,
      "precision": 0.779,
      "recall": 0.754,
      "f1_score": 0.766
    }
  }
}
```

---

### 3. Predict Job Match ⭐
**Get match prediction with explainability**

```http
POST /match/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "student": {
    "student_id": "ST_001",
    "name": "John Doe",
    "verified_skills": ["Python", "SQL", "AWS"],
    "skill_scores": {
      "Python": 0.85,
      "SQL": 0.78,
      "AWS": 0.72
    },
    "gpa": 3.8,
    "experience_years": 3,
    "college_id": "College_A"
  },
  "job": {
    "job_id": "JB_001",
    "title": "Senior Python Developer",
    "company": "Tech Corp",
    "required_skills": ["Python", "SQL", "AWS", "Docker"],
    "required_exp_years": 3,
    "salary_range": "$150000-$200000",
    "college_id": "College_A"
  }
}
```

**Response:**
```json
{
  "match_score": 0.82,
  "match_probability": 0.82,
  "explanation": "Strong match (82.0%) between John Doe and Senior Python Developer at Tech Corp. Strong skill alignment: 3 of the 4 required skills are verified and matched. Close experience match: 3 years matches the required 3 years. Strong academic performance with 3.8 GPA. Same college: Both student and job are from the same institution. Top contributing factors: skill_overlap_ratio, avg_matching_skill_score. This prediction is based on 12 verified signals from the student profile and job requirements.",
  "top_factors": [
    {
      "feature": "skill_overlap_ratio",
      "value": 0.75,
      "importance": 0.287,
      "impact": "positive"
    },
    {
      "feature": "avg_matching_skill_score",
      "value": 0.78,
      "importance": 0.198,
      "impact": "positive"
    },
    {
      "feature": "experience_match_ratio",
      "value": 1.0,
      "importance": 0.165,
      "impact": "positive"
    }
  ],
  "model_used": "Gradient Boosting",
  "confidence_level": "high",
  "timestamp": "2024-06-28T10:30:00.123456"
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `503 Service Unavailable` - Models not loaded yet
- `500 Internal Server Error` - Processing error

**Field Definitions:**
- `match_score` (0-1): Probability score of match
- `explanation` (string): Human-readable explanation
- `top_factors`: List of most important features ranked by influence
- `confidence_level`: "high" (≥0.7), "medium" (0.4-0.7), "low" (<0.4)
- `model_used`: Name of best model used for prediction

---

### 4. Batch Evaluation
**Evaluate a specific model on test set**

```http
POST /batch/evaluate
Content-Type: application/json
```

**Request Body:**
```json
{
  "model_name": "Gradient Boosting",
  "test_sample_size": 100
}
```

**Response:**
```json
{
  "model_name": "Gradient Boosting",
  "evaluation_date": "2024-06-28T10:30:00.123456",
  "metrics": {
    "accuracy": 0.798,
    "precision": 0.779,
    "recall": 0.754,
    "f1_score": 0.766,
    "false_positive_rate": 0.118,
    "auc_roc": 0.867
  },
  "baseline_comparison": {
    "baseline_f1": 0.705,
    "improvement_percent": 8.65,
    "model_beats_baseline": true
  },
  "confusion_matrix": {
    "true_negatives": 38,
    "true_positives": 37,
    "false_positives": 12,
    "false_negatives": 13
  },
  "samples_evaluated": 100
}
```

**Status Codes:**
- `200 OK` - Evaluation complete
- `400 Bad Request` - Invalid model name
- `503 Service Unavailable` - Models not loaded

---

### 5. Admin Verify Match
**Verify match for admin review queue**

```http
POST /admin/verify
Content-Type: application/json
```

**Request Body:** (Same as /match/predict)
```json
{
  "student": {...},
  "job": {...}
}
```

**Response:**
```json
{
  "student_id": "ST_001",
  "job_id": "JB_001",
  "all_model_scores": {
    "Logistic Regression": 0.72,
    "SVM": 0.75,
    "Random Forest": 0.79,
    "Gradient Boosting": 0.82
  },
  "recommendation_score": 0.82,
  "recommended_model": "Gradient Boosting",
  "explanation": "...",
  "top_contributing_factors": [...],
  "admin_notes": "Match verified at 2024-06-28T10:30:00.123456",
  "data_privacy_verified": {
    "student_college_isolated": true,
    "job_college_scoped": true,
    "cross_college_leakage": false
  }
}
```

**Status Codes:**
- `200 OK` - Verification complete
- `503 Service Unavailable` - Models not loaded
- `500 Internal Server Error` - Verification failed

---

### 6. Get Model Comparison
**Compare all trained models**

```http
GET /metrics/comparison
```

**Response:**
```json
{
  "timestamp": "2024-06-28T10:30:00.123456",
  "model_comparison": {
    "Logistic Regression": {
      "accuracy": 0.752,
      "precision": 0.728,
      "recall": 0.685,
      "f1_score": 0.705,
      "auc_roc": 0.821
    },
    "SVM": {
      "accuracy": 0.768,
      "precision": 0.741,
      "recall": 0.702,
      "f1_score": 0.720,
      "auc_roc": 0.834
    },
    "Random Forest": {
      "accuracy": 0.785,
      "precision": 0.763,
      "recall": 0.731,
      "f1_score": 0.746,
      "auc_roc": 0.851
    },
    "Gradient Boosting": {
      "accuracy": 0.798,
      "precision": 0.779,
      "recall": 0.754,
      "f1_score": 0.766,
      "auc_roc": 0.867
    }
  },
  "best_model": "Gradient Boosting",
  "baseline_f1": 0.705
}
```

**Status Codes:**
- `200 OK` - Metrics retrieved
- `503 Service Unavailable` - Models not loaded

---

### 7. Get Metrics History
**Retrieve historical evaluation results**

```http
GET /metrics/history?limit=50
```

**Query Parameters:**
- `limit` (int, optional): Number of recent results (default: 50, max: 100)

**Response:**
```json
{
  "total_evaluations": 42,
  "recent_metrics": [
    {
      "model_name": "Gradient Boosting",
      "evaluation_date": "2024-06-28T10:25:00.123456",
      "metrics": {
        "accuracy": 0.798,
        "precision": 0.779,
        "recall": 0.754,
        "f1_score": 0.766,
        "false_positive_rate": 0.118,
        "auc_roc": 0.867
      },
      "baseline_comparison": {
        "baseline_f1": 0.705,
        "improvement_percent": 8.65,
        "model_beats_baseline": true
      },
      "confusion_matrix": {
        "true_negatives": 38,
        "true_positives": 37,
        "false_positives": 12,
        "false_negatives": 13
      },
      "samples_evaluated": 100
    }
  ],
  "timestamp": "2024-06-28T10:30:00.123456"
}
```

**Status Codes:**
- `200 OK` - History retrieved
- `503 Service Unavailable` - Metrics system not initialized

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request body. Check field types and required fields."
}
```

### 503 Service Unavailable
```json
{
  "detail": "Models not loaded. Server is starting up."
}
```

### 500 Internal Server Error
```json
{
  "detail": "An unexpected error occurred. Check server logs."
}
```

---

## Rate Limiting
Currently disabled. Configure in production:
```python
# Add to main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/match/predict", dependencies=[Depends(limiter.limit("100/minute"))])
```

---

## Example Usage - cURL

### Quick Test
```bash
# Health check
curl http://localhost:8000/health

# Get models
curl http://localhost:8000/models/available

# Make prediction
curl -X POST http://localhost:8000/match/predict \
  -H "Content-Type: application/json" \
  -d @prediction_request.json

# Get metrics
curl http://localhost:8000/metrics/comparison
```

### Python Example
```python
import requests

API_BASE = "http://localhost:8000"

# Health check
response = requests.get(f"{API_BASE}/health")
print(response.json())

# Predict match
payload = {
  "student": {
    "student_id": "ST_001",
    "name": "John Doe",
    "verified_skills": ["Python", "SQL"],
    "skill_scores": {"Python": 0.85, "SQL": 0.78},
    "gpa": 3.8,
    "experience_years": 3,
    "college_id": "College_A"
  },
  "job": {
    "job_id": "JB_001",
    "title": "Python Developer",
    "company": "Tech Corp",
    "required_skills": ["Python", "SQL"],
    "required_exp_years": 2,
    "salary_range": "$100000-$150000",
    "college_id": "College_A"
  }
}

response = requests.post(
  f"{API_BASE}/match/predict",
  json=payload
)

result = response.json()
print(f"Match Score: {result['match_score']}")
print(f"Explanation: {result['explanation']}")
```

### JavaScript/Node.js Example
```javascript
const API_BASE = "http://localhost:8000";

async function predictMatch(student, job) {
  const response = await fetch(`${API_BASE}/match/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ student, job })
  });
  
  return response.json();
}

// Usage
const prediction = await predictMatch({
  student_id: "ST_001",
  name: "John Doe",
  verified_skills: ["Python", "SQL"],
  skill_scores: { "Python": 0.85, "SQL": 0.78 },
  gpa: 3.8,
  experience_years: 3,
  college_id: "College_A"
}, {
  job_id: "JB_001",
  title: "Python Developer",
  company: "Tech Corp",
  required_skills: ["Python", "SQL"],
  required_exp_years: 2,
  salary_range: "$100000-$150000",
  college_id: "College_A"
});

console.log(prediction);
```

---

## Versioning

Current API Version: **v1.0.0**

Future versions will be available at `/api/v2/...`

---

## Support & Issues

For API issues:
1. Check server logs in terminal
2. Verify request format matches examples
3. Ensure models are loaded (`/health` should return `models_loaded: true`)
4. Check README.md for common issues

---

**Last Updated**: June 28, 2024  
**Status**: ✅ Production Ready
