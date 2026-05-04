import os
import asyncio
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from lib.database.models import Profile, User
from lib.database.db import _normalize_db_url

async def check_db():
    db_url = _normalize_db_url("postgresql+psycopg2://postgres:AlEx%4020020226%40@db.vestlmclwcbcfajnqwtf.supabase.co:5432/postgres")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        # Check users
        users = session.execute(select(User)).scalars().all()
        print(f"Total Users: {len(users)}")
        for u in users:
            print(f" - User: {u.email}")
            
        # Check profiles
        profiles = session.execute(select(Profile)).scalars().all()
        print(f"\nTotal Profiles: {len(profiles)}")
        for p in profiles:
            p_json = p.profile_json if isinstance(p.profile_json, dict) else {}
            has_guide = "careerGuide" in p_json and p_json["careerGuide"]
            has_prep = "careerPrep" in p_json and p_json["careerPrep"]
            has_emotion = "careerEmotion" in p_json and p_json["careerEmotion"]
            has_market = "careerMarket" in p_json and p_json["careerMarket"]
            print(f" - Profile for {p.email}: Guide={bool(has_guide)}, Prep={bool(has_prep)}, Emotion={bool(has_emotion)}, Market={bool(has_market)}")

if __name__ == "__main__":
    asyncio.run(check_db())
