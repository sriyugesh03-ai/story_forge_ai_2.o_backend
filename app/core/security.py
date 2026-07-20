import hashlib
import os
import secrets
import jwt
from datetime import datetime, timedelta, timezone

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "story_forge_super_secure_vault_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  # Iterations count
    )
    return f"{salt.hex()}:{key.hex()}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its PBKDF2 hash."""
    try:
        salt_hex, key_hex = hashed_password.split(':')
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        
        test_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return secrets.compare_digest(test_key, expected_key)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Generate a JWT token containing data and expiration timestamp."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict | None:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None
