import asyncio
import json
import logging
import urllib.error
import urllib.request

import jwt
from jwt import PyJWKClient

from app.core.config import settings

logger = logging.getLogger(__name__)

CLERK_LEEWAY_SECONDS = 30


class ClerkVerificationError(Exception):
    """Raised when a Clerk token cannot be trusted."""


class ClerkVerifier:
    """Verifies Clerk session JWTs offline using the instance's public JWKS.

    The token signature is validated against Clerk's published keys, and the
    issuer + expiration are enforced. The audience/azp claims are enforced
    only when an audience is configured, because Clerk token formats vary.
    """

    def __init__(self, jwks_url: str, issuer: str, audience: str = ""):
        self._jwks_client = PyJWKClient(jwks_url, cache_keys=True)
        self.issuer = issuer
        self.audience = audience

    def verify(self, token: str) -> dict:
        if not token:
            raise ClerkVerificationError("Missing Clerk token")

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.issuer,
                leeway=CLERK_LEEWAY_SECONDS,
                options={"require": ["sub", "iss", "exp"]},
            )
        except jwt.PyJWTError as err:
            raise ClerkVerificationError(f"Invalid or expired Clerk token: {err}")

        if self.audience:
            azp = payload.get("azp")
            if azp and azp != self.audience:
                raise ClerkVerificationError("Clerk token audience mismatch (azp)")
            aud = payload.get("aud")
            if isinstance(aud, str):
                aud = [aud]
            if aud and self.audience not in aud:
                raise ClerkVerificationError("Clerk token audience mismatch (aud)")

        return payload


def _build_verifier() -> ClerkVerifier:
    if not settings.CLERK_ISSUER or not settings.CLERK_JWKS_URL:
        raise ClerkVerificationError(
            "Clerk is not configured (missing CLERK_ISSUER / CLERK_JWKS_URL)."
        )
    return ClerkVerifier(settings.CLERK_JWKS_URL, settings.CLERK_ISSUER, settings.CLERK_AUDIENCE)


clerk_verifier = _build_verifier() if settings.CLERK_ISSUER and settings.CLERK_JWKS_URL else None


def verify_clerk_token(token: str) -> dict:
    """Verify a Clerk session token and return its verified claims."""
    if clerk_verifier is None:
        raise ClerkVerificationError("Clerk verifier is not configured")
    return clerk_verifier.verify(token)


def get_clerk_user_email(clerk_user_id: str) -> str | None:
    """Fetch the user's primary email from Clerk's Backend API.

    Only needed when the session token does not carry an email claim.
    Returns None if the secret key is missing or the API call fails.
    """
    if not settings.CLERK_SECRET_KEY or not clerk_user_id:
        return None
    url = f"https://api.clerk.com/v1/users/{clerk_user_id}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {settings.CLERK_SECRET_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as err:
        logger.warning("Clerk API user fetch failed: %s", err)
        return None

    emails = data.get("email_addresses") or []
    primary_id = data.get("primary_email_address_id")
    for item in emails:
        if item.get("id") == primary_id:
            return item.get("email_address")
    if emails:
        return emails[0].get("email_address")
    return None


async def aget_clerk_user_email(clerk_user_id: str) -> str | None:
    """Async wrapper for the Clerk Backend API email lookup."""
    return await asyncio.to_thread(get_clerk_user_email, clerk_user_id)