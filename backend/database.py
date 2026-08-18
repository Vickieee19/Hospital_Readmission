"""
backend/database.py
───────────────────
Database configuration and session management for CareGrid.
Uses SQLite for lightweight, file-based persistence without external server requirements.
"""

from __future__ import annotations

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database file location in the backend directory (resolved to absolute path)
BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = (BACKEND_DIR / "caregrid.db").resolve()

# Handle DATABASE_URL from environment or fallback to absolute SQLite DB path
env_db_url = os.getenv("DATABASE_URL")
if env_db_url and not env_db_url.startswith("sqlite:///backend"):
    SQLALCHEMY_DATABASE_URL = env_db_url
else:
    # Always use the absolute path formatted for SQLite
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

# SQLite connection args (check_same_thread=False needed for FastAPI multithreaded requests)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db_schema():
    """
    Initialize SQLite tables and automatically run lightweight schema migrations.
    """
    Base.metadata.create_all(bind=engine)

    # Ensure initial_password column exists if DB was created previously
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # Check existing columns in users table
            result = conn.execute(text("PRAGMA table_info(users)")).fetchall()
            columns = [row[1] for row in result]
            if "initial_password" not in columns and len(columns) > 0:
                conn.execute(text("ALTER TABLE users ADD COLUMN initial_password VARCHAR(255)"))
                conn.commit()
                print("[Database] Migrated schema: added 'initial_password' column.")
            if "designation" not in columns and len(columns) > 0:
                conn.execute(text("ALTER TABLE users ADD COLUMN designation VARCHAR(100) DEFAULT 'Staff'"))
                conn.commit()
                print("[Database] Migrated schema: added 'designation' column.")
            if "permissions" not in columns and len(columns) > 0:
                conn.execute(text("ALTER TABLE users ADD COLUMN permissions VARCHAR(255) DEFAULT 'standard'"))
                conn.commit()
                print("[Database] Migrated schema: added 'permissions' column.")
    except Exception as e:
        print(f"[Database] Migration notice: {e}")


def get_db():
    """
    FastAPI dependency that yields a database session and closes it after the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
