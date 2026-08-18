"""
backend/routes_auth.py
──────────────────────
Authentication routes for CareGrid (Staff/Nurse and Admin Login, Logout, and Profile).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth import (
    LoginRequest,
    TokenResponse,
    UserResponse,
    create_access_token,
    get_current_user,
    verify_password,
)
from backend.database import get_db
from backend.models_db import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse, summary="Login for Staff/Nurse and Administrator Accounts")
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user credentials (username and password) and issue an 8-hour JWT access token.
    Both nurse and admin users authenticate through this endpoint.
    """
    user = db.query(User).filter(User.username == login_data.username.strip()).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Please contact an administrator.",
        )

    # Issue 8-hour JWT token containing subject and role
    token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
            "name": user.full_name,
            "id": user.id,
        }
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post("/logout", summary="Logout current session")
def logout(current_user: User = Depends(get_current_user)):
    """
    Logout confirmation endpoint. Invalidate client token session.
    """
    return {
        "message": f"Successfully logged out {current_user.username}.",
        "status": "success",
    }


@router.get("/me", response_model=UserResponse, summary="Get Current Authenticated User Profile")
def get_me(current_user: User = Depends(get_current_user)):
    """
    Return the profile details and role of the currently logged-in user.
    """
    return UserResponse.model_validate(current_user)
