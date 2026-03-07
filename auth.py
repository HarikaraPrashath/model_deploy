import os
from jose import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SECRET = os.getenv("SECRET")
ALGORITHM = "HS256"


def create_token(data: dict, days: int = 3):
    payload = data.copy()

    expire = datetime.utcnow() + timedelta(days=days)

    payload.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })

    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)