import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings

logger = logging.getLogger(__name__)

ACCESS_TOKEN_TYPE = "access"


def jwt_secret() -> str:
    secret = settings.APP_JWT_SECRET
    if not secret:
        # Development convenience only — never rely on this in production.
        secret = secrets.token_hex(32)
        logger.warning(
            "APP_JWT_SECRET is not set; using an ephemeral dev secret. "
            "Set APP_JWT_SECRET in .env for production."
        )
    return secret


def create_access_token(user_id: str) -> str:
    """Create a short-lived application access token for the given user.

    Claims are minimal (no PII): sub (user id), type, jti, iat, exp.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": ACCESS_TOKEN_TYPE,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, jwt_secret(), algorithm="HS256")


def decode_access_token(token: str) -> dict | None:
    """Validate an application access token; returns claims or None."""
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("type") != ACCESS_TOKEN_TYPE:
        return None
    return payload


def generate_refresh_token() -> str:
    """Generate a cryptographically secure random refresh token."""
    return secrets.token_urlsafe(64)


def hash_refresh_token(token: str) -> str:
    """Hash a refresh token before it is stored in MongoDB."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()