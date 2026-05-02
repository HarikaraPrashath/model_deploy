
import os
from sqlalchemy import text
from lib.database.db import engine

def check_columns():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'job_metadata'"))
        columns = [row[0] for row in result.fetchall()]
        print(f"Current columns in 'job_metadata' table: {columns}")
        
        missing = [
            'text_full', 'skills_found', 'must_have_skills', 'nice_to_have_skills',
            'core_skills', 'role_tags', 'source_keyword', 'scraped_at', 'extraction_metadata'
        ]
        
        for col in missing:
            if col not in columns:
                print(f"Adding '{col}' column...")
                col_type = "JSONB" if col in ['skills_found', 'must_have_skills', 'nice_to_have_skills', 'core_skills', 'role_tags', 'extraction_metadata'] else "TEXT"
                if col == 'scraped_at': col_type = "TIMESTAMP WITH TIME ZONE"
                
                conn.execute(text(f"ALTER TABLE job_metadata ADD COLUMN {col} {col_type}"))
                print(f"Column '{col}' added.")
        
        conn.commit()
        print("Schema check/update complete.")

if __name__ == "__main__":
    check_columns()
