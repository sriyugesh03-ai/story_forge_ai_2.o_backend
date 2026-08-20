import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse

from app.auth import github as github_auth
from app.auth.dependencies import (
    get_current_user,
    github_callback_limiter,
    github_start_limiter,
    github_tool_limiter,
)
from app.auth.schemas import (
    GitHubAuthUrlResponse,
    GitHubStatusResponse,
    GitHubToolIn,
    GitHubToolOut,
    MessageResponse,
)
from app.core.config import settings
from app.repo.github_repo import GithubRepository
from app.tools.github_tool import (
    GithubNotConnectedError,
    call_github_tool,
)

router = APIRouter(tags=["github"])

github_repo = GithubRepository()

_LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def _redirect_base(request: Request) -> str:
    """Where the browser is sent after the OAuth callback.

    Option A: the SPA is served from the same origin as this API, so the
    callback redirects back to the request's own origin. Hostnames outside an
    allow-list fall back to FRONTEND_URL to avoid open-redirect misuse.
    """
    base = str(request.base_url).rstrip("/")
    host = request.url.hostname or ""
    frontend_host = urllib.parse.urlparse(settings.FRONTEND_URL).hostname or ""
    if host in _LOCAL_HOSTS or (frontend_host and host == frontend_host):
        return base
    return settings.FRONTEND_URL.rstrip("/")


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("/auth/github", response_model=GitHubAuthUrlResponse)
async def github_oauth_start(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Generate the GitHub authorization URL for the logged-in user.

    The returned URL already contains a signed, single-use state that is bound
    to this user. The frontend simply redirects the browser to it.
    """
    if not github_start_limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

    state = github_auth.create_state(current_user["id"])
    await github_repo.set_pending_state(current_user["id"], state)

    try:
        auth_url = github_auth.build_authorize_url(state)
    except github_auth.GitHubNotConfiguredError as err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err))

    return GitHubAuthUrlResponse(auth_url=auth_url)


@router.get("/auth/github/callback")
async def github_oauth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """GitHub redirects the browser here after the user authorizes (or denies).

    Validates the one-time state, exchanges the code for a token, encrypts it,
    stores it, and sends the browser back to the SPA.
    """
    if not github_callback_limiter.allow(_client_ip(request)):
        return RedirectResponse(f"{_redirect_base(request)}/settings?github_error=rate_limited")

    if error:
        return RedirectResponse(f"{_redirect_base(request)}/settings?github_error=denied")

    if not code or not state:
        return RedirectResponse(f"{_redirect_base(request)}/settings?github_error=missing_params")

    user_id = github_auth.verify_state(state)
    if not user_id:
        return RedirectResponse(f"{_redirect_base(request)}/settings?github_error=invalid_state")

    pending = await github_repo.find_by_user_id(user_id)
    stored_state = pending.get("pending_state") if pending else None
    if not stored_state or stored_state != state or github_auth.is_pending_state_expired(
        pending.get("pending_state_expires_at")
    ):
        return RedirectResponse(f"{_redirect_base(request)}/settings?github_error=state_mismatch")

    try:
        token, scopes = github_auth.exchange_github_code(code)
        identity = github_auth.fetch_github_user(token)
    except github_auth.GitHubOAuthError:
        return RedirectResponse(f"{_redirect_base(request)}/settings?github_error=token_exchange")

    encrypted = github_auth.encrypt_token(token)
    await github_repo.save_connection(
        user_id,
        github_user_id=identity["github_user_id"],
        github_username=identity["github_username"],
        encrypted_token=encrypted,
        scopes=scopes,
    )
    return RedirectResponse(f"{_redirect_base(request)}/settings?github=connected")


@router.get("/auth/github/status", response_model=GitHubStatusResponse)
async def github_status(current_user: dict = Depends(get_current_user)):
    """Return whether the user has GitHub connected (never the token)."""
    conn = await github_repo.find_by_user_id(current_user["id"])
    if not conn or not conn.get("access_token_enc"):
        return GitHubStatusResponse(connected=False)
    return GitHubStatusResponse(
        connected=True,
        github_username=conn.get("github_username"),
        github_user_id=conn.get("github_user_id"),
        scopes=conn.get("scopes"),
    )


@router.post("/auth/github/disconnect", response_model=MessageResponse)
async def github_disconnect(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Revoke the GitHub token server-side and delete the connection."""
    conn = await github_repo.find_by_user_id(current_user["id"])
    if conn and conn.get("access_token_enc"):
        try:
            token = github_auth.decrypt_token(conn["access_token_enc"])
            await github_auth.revoke_github_token(token)  # blocking call
        except Exception:
            pass  # revocation is best-effort; local deletion still proceeds
    await github_repo.disconnect(current_user["id"])
    return MessageResponse(message="GitHub disconnected.")


@router.post("/github/mcp/tool", response_model=GitHubToolOut)
async def github_tool_endpoint(
    payload: GitHubToolIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Run a GitHub tool (via remote MCP, falling back to REST) for the user."""
    if not github_tool_limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

    try:
        result = await call_github_tool(current_user["id"], payload.tool, payload.arguments)
    except GithubNotConnectedError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err))

    return GitHubToolOut(tool=payload.tool, via=result["via"], data=result["data"])