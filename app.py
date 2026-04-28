from __future__ import annotations

import os
import sys
from typing import Any
from contextlib import asynccontextmanager
from routes.usersRoute import router
from database.database import Base, engine
from typing import Optional
from pydantic import BaseModel
from service.personality_career.constants import INTERVIEW_QUESTIONS
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from model_schema.schema import StudentInput
from model_schema.student_model import StudentInputGuide
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from service.Career_preddiction import predict_career_service

#files are import from service folder
from service.career_guide_service import run_prediction


from service.career_market import cv_service

from service.career_market.auth_endpoints_service import (
    signup_service,
    login_service,
    forgot_password_service,
)

from service.career_market.profile_service import (
    get_profile_service,
    put_profile_service,
)
from service.career_market.jobs_service import (
    get_jobs_service,
    get_job_file_service,
    refresh_jobs_service,
)
from service.career_market.trends_service import (
    get_trend_history_service,
    get_trends_service,
    seed_trends_service,
)
from service.career_market.analysis_service import analyse_service
from service.personality_career.interview_analysis_service import analyze_interview_service
from service.career_market.root_health_service import health_service
from service.career_market.ranked_service import (
    get_ranked_service,
    get_ranked_summary_service,
)

from service.career_market.utils.config import (
    ANALYSIS_BACKEND_DIR,
    CV_EXTRACTOR_DIR,
    MAX_FILE_SIZE,
    ensure_storage_dirs,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Create tables for the main schema and for the library models
        from lib.database.models import Base as LibBase

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # ensure tables declared in lib/database/models.py are created too
            await conn.run_sync(LibBase.metadata.create_all)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"⚠️  Failed to create database tables: {str(e)}")
        print("ℹ️  App will continue running (using local SQLite)")
    
    yield
    
    # Shutdown
    await engine.dispose()



#this is Prevent the CORS issues
app = FastAPI(title="Career Prediction", lifespan=lifespan)

# only allow the front-end host (or list of hosts) when cookies/credentials are used
# Wildcard (*) is not permitted when credentials=True.  Pull from env or default to localhost:3000.
frontend_origins = os.getenv("FRONTEND_ORIGINS")
if frontend_origins:
    origins_list = [o.strip() for o in frontend_origins.split(",") if o.strip()]
else:
    # Allow both localhost and 127.0.0.1 (and IPv6 loopback) during development
    origins_list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
#this is the root path
@app.get("/")
def root():
    return {"status": "Career Prediction API running"}

@app.post("/predict-career")
def predict_career_route(inp: StudentInputGuide):
    return run_prediction(inp)

@app.post("/predict")
def predict_career(student: StudentInput):
    print("✅ Request received for Random Forest prediction")
    return predict_career_service(student)

@app.post("/debug")
def debug_endpoint(student: StudentInput):
    return {"received": student.dict()}

#---career-market---------------------------------------------------------------------------------------
#------------temp--------------------
@app.post("/auth/signup")
async def signup_endpoint(payload: dict[str, Any]) -> JSONResponse:
    return signup_service(payload)

@app.post("/auth/login")
async def login_endpoint(payload: dict[str, Any]) -> JSONResponse:
    return login_service(payload)

@app.post("/auth/forgot-password")
async def forgot_password_endpoint(payload: dict[str, Any]) -> JSONResponse:
    return forgot_password_service(payload)
#----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return health_service()

@app.post("/parse")
def parse_cv_endpoint(file: UploadFile = File(...)):
    return cv_service.parse_cv(file, max_file_size=MAX_FILE_SIZE)

@app.get("/cv")
def get_latest_cv_endpoint(request: Request):
    return cv_service.get_latest_cv_response(request)

@app.get("/cv/file")
def get_cv_file_endpoint(id: str, request: Request):
    return cv_service.get_cv_file_response(id, request)

@app.get("/profile")
def get_profile_endpoint(request: Request) -> JSONResponse:
    return get_profile_service(request)

@app.put("/profile")
async def put_profile_endpoint(payload: dict[str, Any], request: Request) -> JSONResponse:
    return put_profile_service(payload, request)

@app.get("/jobs")
def get_jobs_endpoint() -> JSONResponse:
    return get_jobs_service()

@app.get("/jobs/file")
def get_job_file_endpoint(name: str) -> FileResponse:
    return get_job_file_service(name)

@app.post("/jobs/refresh")
async def refresh_jobs_endpoint(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    return refresh_jobs_service(request, payload)

@app.get("/trends/history")
def get_trend_history_endpoint() -> JSONResponse:
    return get_trend_history_service()

@app.get("/trends")
def get_trends_endpoint() -> JSONResponse:
    return get_trends_service()

@app.post("/trends/seed")
async def seed_trends_endpoint(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    return seed_trends_service(request, payload)

@app.get("/ranked")
def get_ranked_endpoint() -> JSONResponse:
    return get_ranked_service()

@app.get("/ranked/summary")
def get_ranked_summary_endpoint() -> JSONResponse:
    return get_ranked_summary_service()

@app.post("/analyse")
async def analyse_endpoint(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    return await analyse_service(request, payload)

@app.post("/api/analyze")
async def analyze_interview_endpoint(payload: dict[str, Any]) -> JSONResponse:
    """Analyze interview emotions and predict career based on emotional patterns."""
    return analyze_interview_service(payload)


# sample questions route pulls from shared constants so it matches
# the rest of the backend's data definitions.

@app.get("/api/questions")
def get_questions():
    """Return a list of interview questions stored in constants.

    The frontend relies on this to populate the interview UI.  Keeping the
    data in a central constants module makes it easier to edit or extend
    without touching the route code.
    """
    # the constant is a list of dicts matching the Question type
    return {"questions": INTERVIEW_QUESTIONS}

# WebSocket route used by front-end analyze feature
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """Websocket handler for the analyze feature.

    When a client connects we immediately send the current set of
    questions (mirroring the GET /api/questions route).  Afterwards we echo
    any text received back to the client.  The client was previously closing
    the socket quickly because it never received any data and assumed the
    connection had failed; sending a welcome message/questions gives it
    something to act on.

    You should adapt the logic below to push live analysis updates instead
    of the simple echo implementation.
    """
    await websocket.accept()
    print("🔌 WebSocket /ws/analyze accepted")

    # send the initial question list so the frontend can render them
    # (replace this with a DB call, service call, etc. as needed)
    try:
        questions_payload = get_questions()  # returns {'questions': [...]}
        await websocket.send_json(questions_payload)
    except Exception as e:
        print(f"⚠️ failed to send initial questions: {e}")

    try:
        while True:
            text = await websocket.receive_text()
            print("📥 received from client:", text)
            # echo for now; real code would feed into analyse_service
            await websocket.send_text(f"received: {text}")
    except WebSocketDisconnect:
        print("🛑 WebSocket /ws/analyze disconnected")

# Authentication
app.include_router(router, prefix="/api/users")