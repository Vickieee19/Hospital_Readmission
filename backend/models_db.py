"""
backend/models_db.py
────────────────────
SQLAlchemy Database Models for CareGrid User Management and Authentication.
"""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from backend.database import Base


class User(Base):
    """
    SQLAlchemy User model for staff/nurse and administrator accounts.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="nurse")  # "nurse", "doctor", "sub_admin", "admin"
    full_name = Column(String(100), nullable=False)
    # Professional job title / designation (e.g. Doctor, Charge Nurse, Clinical Supervisor)
    designation = Column(String(100), nullable=True, default="Staff")
    # Selective permissions for sub-admins (e.g. "create_users,view_reports")
    permissions = Column(String(255), nullable=True, default="standard")
    # Stored temporary/assigned password visible to administrators for staff credential management
    initial_password = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return f"<User id={self.id} username='{self.username}' role='{self.role}' active={self.is_active}>"
