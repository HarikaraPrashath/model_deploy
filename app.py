from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model_schema.schema import StudentInput
from service.Career_preddiction import predict_career_service

#files are import from service folder
from service.career_guide_service import predict_career

#this is Prevent the CORS issues
app = FastAPI(title="Career Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#this is the root path
@app.get("/")
def root():
    return {"status": "Career Prediction API running"}

@app.post("/predict")
def predict(inp: StudentInput):
    return predict_career(inp)

@app.post("/predict")
def predict_career(student: StudentInput):
    print("✅ Request received for Random Forest prediction")
    return predict_career_service(student)

@app.post("/debug")
def debug_endpoint(student: StudentInput):
    return {"received": student.dict()}