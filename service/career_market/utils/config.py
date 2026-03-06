from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
CAREER_MARKET_DIR = BASE_DIR / "service" / "career_market"
CV_EXTRACTOR_DIR = CAREER_MARKET_DIR / "cv_extractor"
SCR_OUTPUT_DIR = CAREER_MARKET_DIR / "scr_output" / "topjobs_ads"
STORAGE_DIR = CAREER_MARKET_DIR / "storage"
CV_STORAGE_DIR = STORAGE_DIR / "cvs"
PROFILES_DIR = STORAGE_DIR / "profiles"
ANALYSIS_BACKEND_DIR = CAREER_MARKET_DIR / "analysis_pipeline"
ANALYSIS_OUTPUT_DIR = STORAGE_DIR / "analysis_output"

PROFILE_PATH = STORAGE_DIR / "profile.json"
CV_INDEX_PATH = STORAGE_DIR / "cv_index.json"
LAST_QUERY_PATH = STORAGE_DIR / "last_query.json"
USERS_PATH = STORAGE_DIR / "users.json"
TREND_HISTORY_PATH = STORAGE_DIR / "trends_history.json"

SCRAPER_PATH = CAREER_MARKET_DIR / "job_analyse" / "scrapper" / "TopJobs_scraper_t2.py"
PIPELINE_PATH = CAREER_MARKET_DIR / "job_analyse" / "job_skill_pipeline.py"
SKILLS_PATH = CV_EXTRACTOR_DIR / "skills.txt"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
JOB_STORAGE_PREFIX = "jobs"

MAX_FILE_SIZE = 20 * 1024 * 1024
TREND_WINDOW_DAYS = 7
TREND_HISTORY_DAYS = 30
TREND_MIN_COUNT = 2
TREND_RISE_THRESHOLD = 0.3
TREND_DECLINE_THRESHOLD = 0.3


def ensure_storage_dirs() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CV_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
