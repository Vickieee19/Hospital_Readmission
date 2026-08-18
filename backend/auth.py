"""
backend/auth.py
───────────────
Authentication and Authorization Service for CareGrid.
Provides:
- Bcrypt password hashing & verification
- 8-hour JWT token generation & payload verification
- Role-Based Access Control (Nurse vs Admin)
- Automatic initial admin seeding
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import User

# ── JWT & Security Configuration ─────────────────────────────────────────────
# In production, ensure JWT_SECRET_KEY is stored in a secure secret store or .env
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "caregrid-healthcare-jwt-secret-key-change-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "8"))

# Default Admin Credentials for First Run / Local Development
# IMPORTANT NOTE FOR PRODUCTION:
# These credentials are provided for initial system provisioning and local demonstration.
# In a production healthcare deployment, change DEFAULT_ADMIN_PASSWORD via environment variables.
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin@123")
DEFAULT_ADMIN_NAME = os.getenv("DEFAULT_ADMIN_NAME", "System Administrator")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ── Password Hashing Helpers ────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a plaintext password using native bcrypt."""
    import bcrypt
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify that a plaintext password matches its bcrypt hash."""
    import bcrypt
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# ── JWT Token Helpers ───────────────────────────────────────────────────────
def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token with an expiration timestamp."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

    try:
        from jose import jwt
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except ImportError:
        import jwt
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT access token."""
    try:
        try:
            from jose import jwt, JWTError
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except ImportError:
            import jwt
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
    except Exception:
        return None


# ── Pydantic Schemas ────────────────────────────────────────────────────────
class UserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    designation: str | None = "Staff"
    permissions: str | None = "standard"
    is_active: bool
    initial_password: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int = ACCESS_TOKEN_EXPIRE_HOURS
    user: UserResponse


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=4, max_length=100)
    role: str = Field("nurse", pattern=r"^(nurse|doctor|sub_admin|admin)$")
    designation: str | None = Field(default="Staff", max_length=100)
    permissions: str | None = Field(default="standard", max_length=255)


class UserStatusUpdateRequest(BaseModel):
    is_active: bool


class UserPasswordUpdateRequest(BaseModel):
    new_password: str = Field(..., min_length=4, max_length=100)


# ── Authentication Dependencies ─────────────────────────────────────────────
def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency that authenticates the user from the Bearer JWT token.
    Raises 401 Unauthorized if the token is missing, invalid, expired, or the user is deactivated.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please log in.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception

    username: str | None = payload.get("sub")
    if not username:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Please contact an administrator.",
        )

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency enforcing Administrator role.
    Raises 403 Forbidden for non-admin accounts.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Full Administrator role required.",
        )
    return current_user


def require_admin_or_subadmin(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency enforcing Administrator or Sub-Administrator role.
    """
    if current_user.role not in ["admin", "sub_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Administrator or Sub-Administrator role required.",
        )
    return current_user


# ── Database Initialization / Default Admin Seeding ─────────────────────────
def seed_default_admin(db: Session) -> None:
    """
    Ensures a default system administrator exists upon system startup.
    """
    admin_exists = db.query(User).filter(User.role == "admin").first()
    if not admin_exists:
        hashed_pwd = hash_password(DEFAULT_ADMIN_PASSWORD)
        default_admin = User(
            username=DEFAULT_ADMIN_USERNAME,
            hashed_password=hashed_pwd,
            initial_password=DEFAULT_ADMIN_PASSWORD,
            role="admin",
            designation="Chief System Administrator",
            permissions="all",
            full_name=DEFAULT_ADMIN_NAME,
            is_active=True,
        )
        db.add(default_admin)
        db.commit()
        db.refresh(default_admin)
        print(f"[Auth] Seeded default administrator account: username='{DEFAULT_ADMIN_USERNAME}'")
