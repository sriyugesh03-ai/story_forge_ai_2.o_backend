from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from app.core.db import get_db_connection
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token
import sqlite3

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# Pydantic Schemas
class RegisterSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class LoginSchema(BaseModel):
    username: str
    password: str

# Dependency to fetch and validate active user session
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
         raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Invalid or expired session token",
             headers={"WWW-Authenticate": "Bearer"},
         )
    
    username = payload["sub"]
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, username, email, created_at FROM users WHERE username = ?", 
        (username,)
    ).fetchone()
    conn.close()
    
    if not user:
         raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Authenticated user record does not exist",
         )
    return dict(user)

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterSchema):
    conn = get_db_connection()
    
    # Check duplicate username
    user_exists = conn.execute("SELECT id FROM users WHERE username = ?", (payload.username,)).fetchone()
    if user_exists:
        conn.close()
        raise HTTPException(status_code=400, detail="Username is already registered")
        
    # Check duplicate email
    email_exists = conn.execute("SELECT id FROM users WHERE email = ?", (payload.email,)).fetchone()
    if email_exists:
        conn.close()
        raise HTTPException(status_code=400, detail="Email address is already registered")
        
    # Save user
    hashed = hash_password(payload.password)
    try:
        conn.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (payload.username, payload.email, hashed)
        )
        conn.commit()
    except sqlite3.Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database write failure: {e}")
        
    conn.close()
    return {"message": "Account created successfully. You can now log in."}

@router.post("/login")
def login(payload: LoginSchema):
    conn = get_db_connection()
    
    # Locate user by username or email
    user = conn.execute(
        "SELECT id, username, email, hashed_password, created_at FROM users WHERE username = ? OR email = ?",
        (payload.username, payload.username)
    ).fetchone()
    conn.close()
    
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    # Generate Token
    token = create_access_token(data={"sub": user["username"], "email": user["email"]})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "email": user["email"],
            "joined": user["created_at"]
        }
    }

@router.get("/me")
def get_profile(current_user: dict = Depends(get_current_user)):
    return current_user
