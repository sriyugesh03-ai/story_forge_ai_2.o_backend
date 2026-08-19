import asyncio
import logging
from dataclasses import dataclass

import wikipediaapi
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 5000
USER_AGENT = "StoryForgeAI/1.0 (sports storytelling assistant)"

_wiki_client = None


@dataclass
class WikipediaResult:
    ok: bool
    title: str = ""
    url: str = ""
    content: str = ""
    error: str = ""


def _get_client():
    """Lazy, shared Wikipedia client."""
    global _wiki_client
    if _wiki_client is None:
        _wiki_client = wikipediaapi.Wikipedia(USER_AGENT, language="en")
    return _wiki_client


def _truncate_paragraph(text: str, limit: int) -> str:
    """Truncate text at a paragraph boundary without cutting mid-sentence."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind("\n")
    if boundary > limit * 0.5:
        return cut[:boundary]
    return cut.rstrip() + "..."


def _search_best_title(client, name: str) -> str | None:
    """Find the best matching Wikipedia page title via full-text search."""
    try:
        results = client.search(name, limit=5)
        titles = list(results.pages.keys())
    except Exception as err:
        logger.warning("Wikipedia search failed for %r: %s", name, err)
        return None

    name_tokens = set(name.lower().split())
    best = None
    for title in titles:
        if not best:
            best = title
        title_tokens = set(title.lower().split())
        if name_tokens.issubset(title_tokens) or title_tokens.issubset(name_tokens):
            return title
    return best


def fetch_wikipedia(player_name: str) -> WikipediaResult:
    """Fetch a player's Wikipedia page and return clean plain-text content."""
    if not player_name or not str(player_name).strip():
        return WikipediaResult(ok=False, error="Invalid player name provided to the Wikipedia tool.")

    client = _get_client()
    try:
        page = client.page(player_name)
        if not page.exists():
            best_title = _search_best_title(client, player_name)
            if not best_title:
                return WikipediaResult(
                    ok=False,
                    error=f"Could not find a Wikipedia page for '{player_name}'. Please check the name and try again.",
                )
            page = client.page(best_title)

        if not page.exists():
            return WikipediaResult(
                ok=False,
                error=f"Could not find a Wikipedia page for '{player_name}'. Please check the name and try again.",
            )

        content = (page.text or "").strip()
        if not content:
            return WikipediaResult(
                ok=False,
                error=f"Wikipedia returned no readable content for '{player_name}'.",
            )

        return WikipediaResult(
            ok=True,
            title=page.title,
            url=page.fullurl,
            content=_truncate_paragraph(content, MAX_CONTENT_CHARS),
        )
    except Exception as err:
        logger.warning("Wikipedia fetch failed for %r: %s", player_name, err)
        return WikipediaResult(
            ok=False,
            error="Wikipedia is currently unavailable. Please try again later.",
        )


@tool
def get_wikipedia_player(player_name: str) -> str:
    """Fetch biographical information about a sports player from Wikipedia.

    Use this tool ONLY when the user asks about a specific athlete or sports
    player whose information is NOT available in the local knowledge base.

    Do NOT use this tool:
    - If the player already exists in the local knowledge base.
    - For questions about the content of the local knowledge base itself.
    - For non-sports or general questions (e.g. programming, politics).

    Args:
        player_name: Full name of the player, e.g. "Virat Kohli".

    Returns:
        Structured text with the Wikipedia page title, URL and cleaned content,
        or a clear error message if the page cannot be found.
    """
    result = fetch_wikipedia(player_name)
    if not result.ok:
        return f"WIKIPEDIA_ERROR: {result.error}"
    return (
        f"Wikipedia page: {result.title}\n"
        f"URL: {result.url}\n\n"
        f"{result.content}"
    )


async def aget_wikipedia_player(player_name: str) -> WikipediaResult:
    """Async wrapper that runs the blocking Wikipedia call off the event loop."""
    return await asyncio.to_thread(fetch_wikipedia, player_name)
