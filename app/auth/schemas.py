from pydantic import BaseModel


class ClerkExchangeIn(BaseModel):
    """Payload for POST /auth/clerk — the Clerk session token from the frontend."""

    clerk_token: str


class TokenResponse(BaseModel):
    """Success payload for the Clerk exchange and refresh endpoints."""

    access_token: str
    token_type: str = "bearer"
    user: dict


class MessageResponse(BaseModel):
    message: str