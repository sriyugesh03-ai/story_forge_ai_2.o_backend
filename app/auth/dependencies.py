import logging
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.tokens import decode_access_token
from app.repo.profiles_repo import ProfilesRepository
from app.repo.users_repo import UsersRepository

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

users_repo = UsersRepository()
profiles_repo = ProfilesRepository()


def serialize_user(user: dict, profile: dict | None = None) -> dict:
    """Sanitized user payload returned to the frontend (no sensitive fields)."""
    return {
        "id": str(user.get("_id", "")),
        "clerk_user_id": user.get("clerk_user_id", ""),
        "email": user.get("email", ""),
        "is_verified": user.get("is_verified", False),
        "is_banned": user.get("is_banned", False),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
        "profile": {
            "first_name": (profile or {}).get("first_name", ""),
            "last_name": (profile or {}).get("last_name", ""),
            "phone": (profile or {}).get("phone", ""),
            "profile_image": (profile or {}).get("profile_image", ""),
            "bio": (profile or {}).get("bio", ""),
            "location": (profile or {}).get("location", ""),
        },
    }


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Resolve the authenticated app user from the application access token.

    The identity always comes from the verified token (never from the client).
    Server-side status (is_banned) is enforced here for every protected route.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await users_repo.find_by_id(payload["sub"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.get("is_banned"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been banned.",
        )

    profile = await profiles_repo.find_by_user_id(user["_id"])
    return serialize_user(user, profile)


class _RateLimiter:
    """Tiny in-memory sliding-window rate limiter, keyed by client IP."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and bucket[0] <= now - self.window:
            bucket.pop(0)
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


clerk_exchange_limiter = _RateLimiter(limit=10, window_seconds=60)
refresh_limiter = _RateLimiter(limit=30, window_seconds=60)
github_start_limiter = _RateLimiter(limit=10, window_seconds=60)
github_callback_limiter = _RateLimiter(limit=10, window_seconds=60)
github_tool_limiter = _RateLimiter(limit=30, window_seconds=60)