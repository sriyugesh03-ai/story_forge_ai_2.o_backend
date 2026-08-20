"""Remote MCP GitHub client — the primary GitHub data path.

Connects to GitHub's HOSTED remote MCP server (api.githubcopilot.com/mcp/)
over Streamable HTTP using the official `mcp` Python SDK. GitHub runs and
maintains this server — nothing runs on our own infrastructure.

Auth: the user's OAuth access token is sent as an `Authorization: Bearer`
header for the duration of the call, exactly like a PAT. It is never logged.
"""

import logging
from typing import Any

import httpx2
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from app.core.config import settings

logger = logging.getLogger(__name__)

_MCP_ACCEPT = "application/json, text/event-stream"


class McpCallError(Exception):
    """Raised when the MCP transport itself fails (auth, network, protocol)."""


def _extract_result(result: types.CallToolResult) -> dict:
    """Flatten an MCP CallToolResult into a plain dict for the tool layer."""
    text_parts: list[str] = []
    structured: dict | None = None

    for item in result.content or []:
        if hasattr(item, "text"):
            text_parts.append(item.text)
        elif isinstance(item, dict):
            if item.get("text"):
                text_parts.append(str(item["text"]))
            elif item.get("structuredContent"):
                structured = item["structuredContent"]
            else:
                text_parts.append(str(item))

    data: dict = {}
    if text_parts:
        data["text"] = "\n".join(text_parts)
    if structured is not None:
        data["structured"] = structured
    if result.structured_content is not None:
        data["structured"] = result.structured_content
    data["is_error"] = bool(result.is_error)
    return data


async def mcp_list_tools(token: str) -> list[str]:
    """List the tool names exposed by the remote GitHub MCP server."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": _MCP_ACCEPT,
        "User-Agent": "story-forge-ai/1.0",
    }
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0)) as client:
        async with streamable_http_client(url=settings.GITHUB_MCP_URL, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [t.name for t in tools.tools]


async def mcp_call_tool(token: str, tool: str, arguments: dict[str, Any]) -> dict:
    """Call a single GitHub MCP tool on the remote server.

    Raises McpCallError for transport/auth failures so the caller can fall
    back to the REST client. Tool-level errors are returned in the payload.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": _MCP_ACCEPT,
        "User-Agent": "story-forge-ai/1.0",
    }
    try:
        async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(60.0)) as client:
            async with streamable_http_client(url=settings.GITHUB_MCP_URL, http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, arguments or {}, read_timeout_seconds=60)
                    return _extract_result(result)
    except McpCallError:
        raise
    except Exception as err:
        logger.warning("GitHub MCP call to tool '%s' failed: %s", tool, type(err).__name__)
        raise McpCallError(str(err)) from err