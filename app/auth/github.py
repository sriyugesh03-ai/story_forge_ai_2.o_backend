"""GitHub OAuth — owns the OAuth responsibilities of the GitHub integration.

This module is the ONLY place that touches GitHub's OAuth endpoints and the
encrypted token storage. It never logs tokens or authorization codes.
"""

import base64
import json
import logging
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.fernet import Fernet

from app.auth.tokens import jwt_secret
from app.core.config import settings

logger = logging.getLogger(__name__)

STATE_TTL_SECONDS = 600  # 10 minutes

# Every call to these GitHub REST endpoints MUST send a User-Agent header.
_USER_AGENT = "story-forge-ai/1.0"


class GitHubOAuthError(Exception):
    """Raised for any GitHub OAuth / token-handling failure."""


class GitHubNotConfiguredError(GitHubOAuthError):
    """GitHub client id/secret/redirect are missing from the environment."""


class GitHubStateError(GitHubOAuthError):
    """The state parameter is missing, invalid, or expired."""


# ── Configuration guard ────────────────────────────────────────────────────


def _require_configured() -> None:
    missing = [
        name
        for name in ("GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "GITHUB_REDIRECT_URI")
        if not getattr(settings, name, "")
    ]
    if missing:
        raise GitHubNotConfiguredError(
            "GitHub OAuth is not configured. Set " + ", ".join(missing) + " in the environment."
        )


def _token_cipher() -> Fernet:
    """Return a Fernet cipher for encrypting GitHub tokens at rest."""
    key = settings.GITHUB_TOKEN_ENCRYPTION_KEY
    if not key:
        raise GitHubOAuthError(
            "GITHUB_TOKEN_ENCRYPTION_KEY is not set. Generate one with: "
            "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode())


# ── State parameter (CSRF protection) ──────────────────────────────────────


def create_state(user_id: str) -> str:
    """Create a signed, short-lived, single-use state bound to the user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "github_state",
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=STATE_TTL_SECONDS)).timestamp()),
    }
    return jwt.encode(payload, jwt_secret(), algorithm="HS256")


def verify_state(state: str) -> str | None:
    """Return the user_id encoded in a valid state, or None if invalid/expired."""
    try:
        payload = jwt.decode(state, jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("type") != "github_state" or not payload.get("sub"):
        return None
    return str(payload["sub"])


# ── Authorization URL ──────────────────────────────────────────────────────


def build_authorize_url(state: str) -> str:
    """Build the GitHub authorization URL the browser is redirected to."""
    _require_configured()
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
        "scope": settings.GITHUB_SCOPES,
        "state": state,
    }
    return f"{settings.GITHUB_AUTH_URL}?{urllib.parse.urlencode(params)}"


# ── Code → Token exchange (server-to-server only) ─────────────────────────


def exchange_github_code(code: str) -> tuple[str, list[str]]:
    """Exchange an authorization code for an access token + granted scopes."""
    _require_configured()
    data = urllib.parse.urlencode(
        {
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": settings.GITHUB_REDIRECT_URI,
        }
    ).encode()
    req = urllib.request.Request(
        settings.GITHUB_TOKEN_URL,
        data=data,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.URLError as err:
        logger.warning("GitHub token exchange network error: %s", err.reason)
        raise GitHubOAuthError("Could not reach GitHub while exchanging the authorization code.")

    try:
        payload = json.loads(body)
    except ValueError:
        raise GitHubOAuthError("Unexpected response from GitHub token endpoint.")

    if "error" in payload:
        # Never log the error details that could include the code.
        logger.warning("GitHub token exchange failed with error='%s'.", payload.get("error_description", "unknown"))
        raise GitHubOAuthError("GitHub did not approve the authorization request.")

    token = payload.get("access_token")
    if not token:
        raise GitHubOAuthError("GitHub did not return an access token.")

    scopes = [s for s in (payload.get("scope", "") or "").split(",") if s]
    return token, scopes


# ── GitHub user identity ───────────────────────────────────────────────────


def fetch_github_user(access_token: str) -> dict:
    """Return {github_user_id, github_username} for the token's owner."""
    req = urllib.request.Request(
        f"{settings.GITHUB_API_URL}/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        logger.warning("GitHub /user lookup failed with HTTP %s.", err.code)
        raise GitHubOAuthError("GitHub rejected the access token.")
    except urllib.error.URLError as err:
        logger.warning("GitHub /user lookup network error: %s", err.reason)
        raise GitHubOAuthError("Could not reach GitHub while verifying the account.")

    try:
        payload = json.loads(body)
        return {
            "github_user_id": int(payload.get("id") or 0),
            "github_username": payload.get("login", ""),
        }
    except (ValueError, TypeError):
        raise GitHubOAuthError("Unexpected response from GitHub.")


# ── Revoke (disconnect) ───────────────────────────────────────────────────


def revoke_github_token(access_token: str) -> None:
    """Revoke a GitHub OAuth App token server-side.

    Uses Basic auth (client_id:client_secret) — the client secret travels only
    in this Authorization header to GitHub and is never exposed to the browser.
    """
    if not settings.GITHUB_CLIENT_ID or not settings.GITHUB_CLIENT_SECRET:
        raise GitHubOAuthError("GitHub OAuth is not configured.")
    basic = base64.b64encode(
        f"{settings.GITHUB_CLIENT_ID}:{settings.GITHUB_CLIENT_SECRET}".encode()
    ).decode()
    req = urllib.request.Request(
        f"{settings.GITHUB_API_URL}/applications/{settings.GITHUB_CLIENT_ID}/token",
        data=b"",
        headers={
            "Authorization": f"Basic {basic}",
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except urllib.error.HTTPError as err:
        # 404 means the token no longer exists — treat as success.
        if err.code != 404:
            logger.warning("GitHub token revocation failed with HTTP %s.", err.code)
            raise GitHubOAuthError("GitHub could not revoke the access token.")
    except urllib.error.URLError as err:
        logger.warning("GitHub token revocation network error: %s", err.reason)
        raise GitHubOAuthError("Could not reach GitHub while revoking the access token.")


# ── Token encryption at rest ───────────────────────────────────────────────


def encrypt_token(access_token: str) -> str:
    return _token_cipher().encrypt(access_token.encode()).decode()


def decrypt_token(encrypted: str) -> str:
    try:
        return _token_cipher().decrypt(encrypted.encode()).decode()
    except Exception:
        raise GitHubOAuthError("Stored GitHub token could not be decrypted.")


# ── Small time helper for expiry comparisons ───────────────────────────────


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def is_pending_state_expired(expires_at: str | None) -> bool:
    if not expires_at:
        return True
    try:
        return datetime.fromisoformat(expires_at) < now_utc()
    except ValueError:
        return True