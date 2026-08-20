"""GitHub tool node for the AI agent.

Mirrors app/tools/wikipedia_tool.py: a tool that fetches external GitHub data
and returns it as structured context for the LLM. Data path:
  1. MCP-first: GitHub's hosted remote MCP server (api.githubcopilot.com/mcp/).
  2. REST fallback: direct api.github.com calls, same tool interface.
"""

import asyncio
import logging

from app.auth import github as github_auth
from app.core.config import settings
from app.repo.github_repo import GithubRepository
from app.services.github_client import GitHubApiError, GithubRestClient
from app.services.github_mcp import McpCallError, mcp_call_tool

logger = logging.getLogger(__name__)

github_repo = GithubRepository()


class GithubNotConnectedError(Exception):
    pass


def allowed_tools() -> set[str]:
    return {t.strip() for t in settings.GITHUB_MCP_ALLOWED_TOOLS.split(",") if t.strip()}


async def get_user_token(user_id: str) -> str:
    """Load a user's GitHub connection and return the DECRYPTED token.

    The token exists in memory only for the duration of the caller's request.
    """
    conn = await github_repo.get_connected_token(user_id)
    if not conn:
        raise GithubNotConnectedError("GitHub is not connected for this account.")
    return github_auth.decrypt_token(conn["access_token_enc"])


async def call_github_tool(user_id: str, tool: str, arguments: dict) -> dict:
    """Execute a GitHub tool for a user, MCP-first with REST fallback.

    Returns {"tool", "via", "data"} where via is "mcp" or "rest".
    """
    if tool not in allowed_tools():
        raise GithubNotConnectedError(f"Tool '{tool}' is not allowed.")

    token = await get_user_token(user_id)

    # 1. MCP-first
    if settings.GITHUB_MCP_URL:
        try:
            result = await mcp_call_tool(token, tool, arguments)
            if not result.get("is_error"):
                return {"tool": tool, "via": "mcp", "data": result}
            logger.warning("GitHub MCP tool '%s' reported an error; trying REST.", tool)
        except McpCallError:
            logger.warning("GitHub MCP transport failed for tool '%s'; falling back to REST.", tool)

    # 2. REST fallback
    try:
        data = await asyncio.to_thread(_rest_call, token, tool, arguments)
        return {"tool": tool, "via": "rest", "data": data}
    except GitHubApiError as err:
        raise GithubNotConnectedError(
            f"GitHub tool '{tool}' failed: {err}" if not err.status else
            f"GitHub returned HTTP {err.status} for tool '{tool}'."
        )


def _rest_call(token: str, tool: str, arguments: dict) -> dict:
    client = GithubRestClient(token)
    return {"result": client.call(tool, arguments)}