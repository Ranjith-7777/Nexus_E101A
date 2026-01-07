from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json

from src.pipeline import main as run_pipeline


app = FastAPI(title="ImpactLens v2 API")

# Hackathon-friendly CORS (frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    scenario: str

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/scenarios")
def scenarios():
    base = os.path.join("data", "demo")
    if not os.path.exists(base):
        return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

@app.post("/run")
def run(req: RunRequest):
    data_dir = os.path.join("data", "demo", req.scenario)
    run_pipeline(data_dir, output_path="output.json")
    with open("output.json", "r", encoding="utf-8") as f:
        return json.load(f)
