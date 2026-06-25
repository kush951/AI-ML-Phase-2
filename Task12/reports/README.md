
# PlaceMux Task 12 — E-Sign Integration & Tamper-Evidence

## Features
- Resume Parsing v0
- JD Parsing
- Multi-model matching
- TF-IDF + Jaccard + Weighted ensemble
- RSA-PSS digital signing
- SHA-256 tamper verification
- FastAPI backend
- Frontend integration
- Experiment logging
- Explainable AI matching

## Models Used
- Logistic Regression
- Random Forest
- SVM
- Ensemble Matching

## APIs
- /parse_resume
- /parse_jd
- /match
- /sign_offer
- /verify_offer

## Run Backend
```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --reload
```

## Frontend
Open frontend/index.html

## Demo Flow
1. Upload Resume
2. Upload JD
3. Generate Match
4. Generate Offer
5. Sign Offer
6. Modify Payload
7. Verify Tampering

## Metrics
- Precision
- Recall
- F1 Score
- Accuracy
- False Positive Rate

