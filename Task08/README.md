# PlaceMux Spend-Quality Guardrail

AI/ML-based intelligent payment protection and job-match quality system for PlaceMux.

---

# Project Overview

The PlaceMux Spend-Quality Guardrail is an AI-powered system designed to prevent low-quality job applications and protect students from spending money on poor-fit opportunities.

The system evaluates:

* Skill compatibility
* Verified assessment scores
* Experience relevance
* Match quality

Based on these factors, the platform:

* Generates match scores
* Shows low-fit warnings
* Triggers refunds automatically
* Performs payment reconciliation
* Provides explainable AI outputs

---

# Problem Statement

Many hiring platforms allow students to spend money on applications even when their profile has very low chances of selection.

This project solves that problem by introducing:

* AI-based fit analysis
* Spend protection
* Refund automation
* Transaction verification

---

# Features

## AI Match Scoring

Uses Machine Learning to predict whether a student is a good fit for a job.

---

## Low-Fit Warning

Warns users when match quality is below acceptable threshold.

Example:

```json
{
  "match_score_percent": 18,
  "low_fit_warning": true
}
```

---

## Refund Trigger System

Automatically triggers refunds when:

* Match score is too low
* Payment inconsistency occurs
* Duplicate transaction detected

---

## Payment Reconciliation

Compares:

```text
Gateway Records == Internal Records
```

Ensures financial consistency.

---

## Explainable AI

Every prediction includes:

* Match score
* Missing skills
* Reasons
* Warning status

Example:

```json
{
  "reasons": [
    "Missing skills: Python, ML",
    "Verified score below threshold"
  ]
}
```

---

# Technology Stack

| Component        | Technology             |
| ---------------- | ---------------------- |
| Backend API      | FastAPI                |
| Machine Learning | scikit-learn           |
| Model            | RandomForestClassifier |
| Frontend         | HTML, CSS, JavaScript  |
| Data Processing  | Pandas, NumPy          |
| API Testing      | Swagger UI             |
| Deployment Ready | Uvicorn                |

---

# Project Structure

```text
placemux_guardrail/
│
├── api/
│   └── main.py
│
├── data/
│   ├── students.csv
│   └── jobs.csv
│
├── reconciliation/
│   └── reconcile.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── train_model.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

# Installation Guide

## 1. Clone Repository

```bash
git clone <repository_url>
cd placemux_guardrail
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv env
env\\Scripts\\activate
```

### Linux / Mac

```bash
python3 -m venv env
source env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Machine Learning Model

Run:

```bash
python train_model.py
```

Expected output:

```text
Precision: 1.0
Recall: 1.0
False Positive Rate: 0.0
Model saved as model.pkl
```

---

# Run Backend Server

Start FastAPI server:

```bash
uvicorn api.main:app --reload
```

Server runs at:

```text
http://127.0.0.1:8000
```

Swagger API Docs:

```text
http://127.0.0.1:8000/docs
```

---

# Run Frontend

Open:

```text
index.html
```

OR use VS Code Live Server.

---

# API Endpoints

## GET /

Health check endpoint.

### Response

```json
{
  "message": "PlaceMux Spend Guardrail Running"
}
```

---

## POST /match

Predicts job-fit quality.

### Request Example

```json
{
  "student_skills": ["HTML", "CSS"],
  "required_skills": ["Python", "ML"],
  "verified_score": 25,
  "experience": 0,
  "min_score": 70
}
```

---

### Response Example

```json
{
  "match_score_percent": 0,
  "prediction": 0,
  "low_fit_warning": true,
  "refund_triggered": true,
  "reasons": [
    "Missing skills: Python, ML",
    "Verified score below threshold"
  ]
}
```

---

# Machine Learning Workflow

## Step 1 — Data Collection

Student and job datasets are loaded from CSV files.

---

## Step 2 — Feature Engineering

Features used:

* Skill overlap
* Verified score
* Experience

---

## Step 3 — Baseline Matching

Formula:

```python
match_score = common_skills / total_required_skills
```

---

## Step 4 — Model Training

Uses:

```python
RandomForestClassifier
```

---

## Step 5 — Evaluation Metrics

Metrics used:

* Precision
* Recall
* False Positive Rate

---

# Performance Metrics

| Metric              | Value |
| ------------------- | ----- |
| Precision           | 1.0   |
| Recall              | 1.0   |
| False Positive Rate | 0.0   |

---

# Refund Logic

Refund is triggered when:

```python
if match_score < 0.4:
    refund = True
```

---

# Reconciliation Logic

Checks:

* Amount consistency
* Transaction status
* Gateway vs Internal records

Example:

```python
if gateway_record != internal_record:
    raise Exception("Mismatch detected")
```

---

# CORS Fix

If frontend shows:

```text
OPTIONS /match 405 Method Not Allowed
```

Add:

```python
from fastapi.middleware.cors import CORSMiddleware
```

Then:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

# Demo Flow

## Good Match

### Input

* Student Skills: Python, ML
* Required Skills: Python, ML

### Output

```json
{
  "match_score_percent": 100,
  "refund_triggered": false
}
```

---

## Bad Match

### Input

* Student Skills: HTML, CSS
* Required Skills: Python, ML

### Output

```json
{
  "match_score_percent": 0,
  "refund_triggered": true
}
```

---

# Edge Cases Handled

* Low-fit applications
* Duplicate payments
* Payment mismatch
* Missing skills
* Low verified scores
* Reconciliation failures

---

# Future Improvements

* Vector similarity search
* NLP-based JD parsing
* Semantic matching
* MLflow integration
* Real payment gateway integration
* Dashboard analytics
* Drift detection

---

# Conclusion

The PlaceMux Spend-Quality Guardrail successfully demonstrates:

* AI-powered job matching
* Spend protection
* Refund automation
* Explainable predictions
* Payment reconciliation
* End-to-end ML integration

This project provides a strong real-world demonstration of AI engineering, backend integration, and intelligent financial protection systems.
