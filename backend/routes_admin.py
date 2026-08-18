"""
backend/routes_admin.py
───────────────────────
Admin Management Routes for CareGrid.
Provides administrator-only endpoints to:
- List all staff and admin accounts (with stored initial/temporary passwords visible to admins)
- Create new nurse / staff accounts
- Edit and reset user passwords
- Delete staff accounts permanently
- Deactivate / reactivate user accounts
- Protects against self-deletion and non-admin access (403 Forbidden)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth import (
    UserCreateRequest,
    UserPasswordUpdateRequest,
    UserResponse,
    UserStatusUpdateRequest,
    get_current_user,
    hash_password,
    require_admin,
    require_admin_or_subadmin,
)
from backend.database import get_db
from backend.models_db import User

router = APIRouter(prefix="/admin", tags=["Admin User Management"])


@router.get("/users", response_model=list[UserResponse], summary="List All Staff Accounts (Admin & Sub-Admin)")
def list_users(
    current_admin: User = Depends(require_admin_or_subadmin),
    db: Session = Depends(get_db),
):
    """
    List all user accounts in the system. Accessible by administrators and authorized sub-administrators.
    Includes initial/temporary passwords and designations for credential provisioning.
    """
    users = db.query(User).order_by(User.id.asc()).all()
    return [UserResponse.model_validate(u) for u in users]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Create New Staff / Sub-Admin Account")
def create_user(
    new_user_data: UserCreateRequest,
    current_admin: User = Depends(require_admin_or_subadmin),
    db: Session = Depends(get_db),
):
    """
    Create a new user account (Nurse, Doctor, Sub-Admin, or Admin).
    - Sub-admins can create clinical staff (Nurse, Doctor), but cannot create Super Admin accounts.
    - Super Admins can create all roles including Sub-Admins with selective permissions and designations.
    """
    # Permission check: Sub-admins cannot create full admins or sub-admins
    if current_admin.role == "sub_admin" and new_user_data.role in ["admin", "sub_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sub-administrators are only authorized to provision clinical staff (Nurses/Doctors).",
        )

    normalized_username = new_user_data.username.strip().lower()

    # Check for existing username
    existing_user = db.query(User).filter(User.username == normalized_username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An account with username '{normalized_username}' already exists.",
        )

    # Hash the password securely with bcrypt
    raw_pwd = new_user_data.password.strip()
    hashed_pwd = hash_password(raw_pwd)

    user = User(
        username=normalized_username,
        hashed_password=hashed_pwd,
        initial_password=raw_pwd,
        role=new_user_data.role,
        full_name=new_user_data.full_name.strip(),
        designation=(new_user_data.designation or "Staff").strip(),
        permissions=(new_user_data.permissions or "standard").strip(),
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}/password", response_model=UserResponse, summary="Edit & Reset User Password")
def update_user_password(
    user_id: int,
    pwd_data: UserPasswordUpdateRequest,
    current_admin: User = Depends(require_admin_or_subadmin),
    db: Session = Depends(get_db),
):
    """
    Update / reset the password for a user account.
    Sub-admins cannot modify super admin passwords.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    if current_admin.role == "sub_admin" and user.role in ["admin", "sub_admin"] and user.id != current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sub-administrators cannot reset passwords of higher or equal tier accounts.",
        )

    raw_new_pwd = pwd_data.new_password.strip()
    if len(raw_new_pwd) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 4 characters long.",
        )

    user.hashed_password = hash_password(raw_new_pwd)
    user.initial_password = raw_new_pwd
    db.commit()
    db.refresh(user)

    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}", summary="Delete User Account")
def delete_user(
    user_id: int,
    current_admin: User = Depends(require_admin_or_subadmin),
    db: Session = Depends(get_db),
):
    """
    Permanently delete a staff account from the database.
    Prevents deletion of self and prevents sub-admins from deleting super admins.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    if user.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security Protection: You cannot delete your own active administrator account.",
        )

    if current_admin.role == "sub_admin" and user.role in ["admin", "sub_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sub-administrators cannot delete administrator or sub-admin accounts.",
        )

    deleted_username = user.username
    db.delete(user)
    db.commit()

    return {
        "message": f"User '@{deleted_username}' has been permanently deleted.",
        "status": "success",
    }


@router.patch("/users/{user_id}/status", response_model=UserResponse, summary="Update User Active Status")
def update_user_status(
    user_id: int,
    status_update: UserStatusUpdateRequest,
    current_admin: User = Depends(require_admin_or_subadmin),
    db: Session = Depends(get_db),
):
    """
    Activate or deactivate a user account.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    if user.id == current_admin.id and not status_update.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security Protection: You cannot deactivate your own active administrator account.",
        )

    if current_admin.role == "sub_admin" and user.role in ["admin", "sub_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sub-administrators cannot modify status of administrator accounts.",
        )

    user.is_active = status_update.is_active
    db.commit()
    db.refresh(user)

    return UserResponse.model_validate(user)
