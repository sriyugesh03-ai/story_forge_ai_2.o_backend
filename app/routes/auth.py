from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.auth.clerk import (
    ClerkVerificationError,
    aget_clerk_user_email,
    verify_clerk_token,
)
from app.auth.dependencies import (
    clerk_exchange_limiter,
    get_current_user,
    profiles_repo,
    refresh_limiter,
    serialize_user,
    users_repo,
)
from app.auth.schemas import ClerkExchangeIn, MessageResponse, TokenResponse
from app.auth.tokens import (
    create_access_token,
    generate_refresh_token,
    hash_refresh_token,
)
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])


def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    """Set the HttpOnly refresh-token cookie.

    HttpOnly prevents JavaScript access; Secure restricts it to HTTPS in
    production; SameSite=strict prevents cross-site sends (the app is served
    from the same origin as the API — Option A). The raw token only ever
    lives in the browser cookie; Mongo only stores its SHA-256 hash.
    """
    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.COOKIE_NAME, path="/")


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/clerk", response_model=TokenResponse)
async def clerk_exchange(payload: ClerkExchangeIn, request: Request, response: Response):
    """Exchange a verified Clerk session token for application tokens.

    The Clerk token is verified against Clerk's JWKS (signature, issuer,
    expiration). The user's identity is always derived from the verified
    token's `sub` claim — never from anything the frontend sends directly.
    """
    if not clerk_exchange_limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")

    try:
        claims = verify_clerk_token(payload.clerk_token)
    except ClerkVerificationError as err:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(err))

    clerk_user_id = claims.get("sub")
    if not clerk_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clerk token missing user id")

    email = claims.get("email") or await aget_clerk_user_email(clerk_user_id) or ""
    user = await users_repo.upsert_clerk_user(clerk_user_id, email, is_verified=True)

    if user.get("is_banned"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account has been banned.")

    profile = await profiles_repo.ensure_profile(user["_id"])

    # Fresh refresh token on every exchange; only the hash is stored.
    raw_refresh = generate_refresh_token()
    await users_repo.set_refresh_token(user["_id"], hash_refresh_token(raw_refresh))
    _set_refresh_cookie(response, raw_refresh)

    access = create_access_token(str(user["_id"]))
    return TokenResponse(access_token=access, token_type="bearer", user=serialize_user(user, profile))


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response):
    """Rotate the refresh token and issue a fresh access token.

    The presented cookie is hashed and matched against the stored hash.
    Reusing an already-rotated token is treated as suspicious and revokes
    the entire session.
    """
    if not refresh_limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

    raw = request.cookies.get(settings.COOKIE_NAME)
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

    token_hash = hash_refresh_token(raw)
    user = await users_repo.find_by_refresh_hash(token_hash)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    # Reuse detection: a rotated-out token must never work again.
    if user.get("refresh_token_prev_hash") == token_hash:
        await users_repo.revoke_refresh_token(user["_id"])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token reuse detected; session revoked.",
        )

    expires_at = user.get("refresh_token_expires_at")
    if not expires_at:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has been revoked")
    try:
        expired = datetime.fromisoformat(expires_at) < datetime.now(timezone.utc)
    except ValueError:
        expired = True
    if expired:
        await users_repo.revoke_refresh_token(user["_id"])
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has expired")

    if user.get("is_banned"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account has been banned.")

    new_raw = generate_refresh_token()
    await users_repo.rotate_refresh_token(user["_id"], token_hash, hash_refresh_token(new_raw))
    _set_refresh_cookie(response, new_raw)

    access = create_access_token(str(user["_id"]))
    return TokenResponse(access_token=access, token_type="bearer", user=serialize_user(user))


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, response: Response):
    """Revoke the refresh token server-side and clear the cookie."""
    raw = request.cookies.get(settings.COOKIE_NAME)
    if raw:
        await users_repo.revoke_by_hash(hash_refresh_token(raw))
    _clear_refresh_cookie(response)
    return MessageResponse(message="Logged out successfully")


@router.get("/me")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user + profile (protected by access token)."""
    return current_user