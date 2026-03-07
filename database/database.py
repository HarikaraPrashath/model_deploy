import os
import ssl
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Use SQLite only if DATABASE_URL is missing
if not DATABASE_URL:
    print("⚠️ Using SQLite (no DATABASE_URL found)")
    DATABASE_URL = "sqlite+aiosqlite:///./fastapi_dev.db"
else:
    print("✅ Using PostgreSQL database")

# Convert PostgreSQL URL to async format
if DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    # Remove SSL and channel binding parameters from URL (handled via connect_args)
    DATABASE_URL = DATABASE_URL.replace("&channel_binding=require", "")
    DATABASE_URL = DATABASE_URL.replace("?sslmode=require", "")

# Engine configuration
if "sqlite" in DATABASE_URL:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False}
    )
else:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        connect_args={
            "timeout": 30,
            "command_timeout": 30,
            "ssl": ssl_context,
            "server_settings": {
                "application_name": "fastapi_app"
            }
        }
    )

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()