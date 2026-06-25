from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.esign.pipeline import ESignPipeline

# ------------------------------------------------

app = FastAPI()

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------

pipeline = ESignPipeline()

# ---------------- REQUEST MODEL ----------------

class MatchRequest(BaseModel):
    resume_skills: list
    jd_skills: list

# ------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "running"
    }

# ------------------------------------------------

@app.post("/match")
def match(req: MatchRequest):

    resume_skills = set([
        s.strip().lower()
        for s in req.resume_skills
    ])

    jd_skills = set([
        s.strip().lower()
        for s in req.jd_skills
    ])

    matched = list(resume_skills & jd_skills)

    missing = list(jd_skills - resume_skills)

    score = len(matched) / max(len(jd_skills), 1)

    return {
        "match_score": round(score * 100, 2),
        "matched_skills": matched,
        "missing_skills": missing,
        "explanation":
        f"Matched {len(matched)} out of {len(jd_skills)} required skills."
    }

# ------------------------------------------------

@app.post("/sign_offer")
def sign_offer(payload: dict):

    signed = pipeline.sign_offer(payload)

    return signed

# ------------------------------------------------

@app.post("/verify_offer")
def verify_offer(payload: dict):

    result = pipeline.verify_offer(payload)

    return result