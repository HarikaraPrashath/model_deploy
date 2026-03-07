import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv 

from database.database import SessionLocal
from schema.models import User
from database_model_schema.schemas import (
    RegisterSchema,
    LoginSchema,
    ForgetPasswordSchema,
    ResetPasswordSchema
)
from auth import create_token

load_dotenv()

router = APIRouter()

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

SECRET = os.getenv("SECRET")

if not SECRET:
    raise RuntimeError("SECRET environment variable not set")


# Database Dependency
async def get_db():
    async with SessionLocal() as session:
        yield session


# ================= REGISTER =================
@router.post("/register")
async def register_user(
    data: RegisterSchema,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    try:
        print(f"📝 Register request: name={data.name}, email={data.email}")
        
        if len(data.password) < 8:
            raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long"
    )

        # Check existing user
        result = await db.execute(
            select(User).where(User.email == data.email)
        )
        existing = result.scalars().first()

        if existing:
            raise HTTPException(status_code=400, detail="Email already exists")

        hashed_password = pwd_context.hash(data.password)

        new_user = User(
            name=data.name,
            email=data.email,
            password=hashed_password
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        token = create_token({"id": new_user.id})

        response.set_cookie(
            key="token",
            value=token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
    )

        return {
            "success": True,
            "user": {
                "id": new_user.id,
                "name": new_user.name,
                "email": new_user.email
            },
            "token": token
        }
    except HTTPException as e:
        print(f"❌ Register error: {e.detail}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ================= LOGIN =================
@router.post("/login")
async def login_user(
    data: LoginSchema,
    response: Response,
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User).where(User.email == data.email)
    )
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not pwd_context.verify(data.password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = create_token({"id": user.id})

    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax"
    )

    return {
        "success": True,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        },
        "token": token
    }


# ================= FORGET PASSWORD =================
@router.post("/reset-password")
async def forget_password(
    data: ForgetPasswordSchema,
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User).where(User.email == data.email)
    )
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = jwt.encode(
        {
        "email": user.email,
        "exp": datetime.utcnow() + timedelta(hours=1)
         },
        SECRET,
        algorithm="HS256"
    )

    reset_url = f"{os.getenv('CLIENT_URL')}/reset-password/{token}"

    msg = MIMEText(f"Reset password here: {reset_url}")
    msg["Subject"] = "Reset Your Password"
    msg["From"] = os.getenv("MY_GMAIL")
    msg["To"] = user.email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(
            os.getenv("MY_GMAIL"),
            os.getenv("MY_PASSWORD")
        )
        server.send_message(msg)

    return {"message": "Reset link sent"}


# ================= RESET PASSWORD =================
@router.post("/reset-password/{token}")
async def reset_password(
    token: str,
    data: ResetPasswordSchema,
    db: AsyncSession = Depends(get_db)
):

    try:
        decoded = jwt.decode(
            token,
            SECRET,
            algorithms=["HS256"]
        )
        email = decoded["email"]

    except JWTError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired token"
        )

    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if len(data.password) < 8 or len(data.password) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password must be between 8 and 72 characters"
        )

    user.password = pwd_context.hash(data.password)

    await db.commit()

    return {"message": "Password reset successfully"}