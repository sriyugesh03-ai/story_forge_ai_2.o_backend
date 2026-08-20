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


# ── GitHub OAuth ──────────────────────────────────────────────────────────


class GitHubAuthUrlResponse(BaseModel):
    """Response for GET /auth/github — the URL the frontend redirects to."""

    auth_url: str


class GitHubStatusResponse(BaseModel):
    """Public GitHub connection status (never exposes the token)."""

    connected: bool = False
    github_username: str | None = None
    github_user_id: int | None = None
    scopes: list[str] | None = None


class GitHubToolIn(BaseModel):
    """Payload for POST /github/mcp/tool."""

    tool: str
    arguments: dict = {}


class GitHubToolOut(BaseModel):
    tool: str
    via: str  # "mcp" | "rest"
    data: dict