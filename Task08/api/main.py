from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI(title="PlaceMux Spend Guardrail API")

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

class MatchRequest(BaseModel):
    student_skills: list
    required_skills: list
    verified_score: int
    experience: int
    min_score: int

@app.get("/")
def home():
    return {
        "message": "PlaceMux Spend Guardrail Running"
    }

@app.post("/match")
def match(req: MatchRequest):

    overlap = len(
        set(req.student_skills).intersection(
            set(req.required_skills)
        )
    )

    score = overlap / len(req.required_skills)

    features = [[
        score,
        req.verified_score,
        req.experience
    ]]

    prediction = int(model.predict(features)[0])

    low_fit = score < 0.4

    reasons = []

    missing = list(
        set(req.required_skills) - set(req.student_skills)
    )

    if missing:
        reasons.append(
            f"Missing skills: {', '.join(missing)}"
        )

    if req.verified_score < req.min_score:
        reasons.append(
            "Verified score below threshold"
        )

    refund = low_fit

    return {
        "match_score_percent": round(score * 100, 2),
        "prediction": prediction,
        "low_fit_warning": low_fit,
        "refund_triggered": refund,
        "reasons": reasons
    }