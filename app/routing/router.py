import inspect
import json
import logging
import re
from dataclasses import dataclass

from app.rag.retriever import _find_best_player_match, _normalize_player_name
from app.services.llm_client import ask_llm
from app.services.retry_service import retry_call

logger = logging.getLogger(__name__)

ROUTE_RAG = "rag"
ROUTE_WIKIPEDIA = "wikipedia"
ROUTE_GENERAL = "general"

VALID_ROUTES = {ROUTE_RAG, ROUTE_WIKIPEDIA, ROUTE_GENERAL}

ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a sports storytelling assistant.

The assistant's knowledge base (RAG) currently contains information ONLY about the players listed under KNOWLEDGE BASE. Any other player is NOT available locally.

Decide which data source answers the user's question:

1. "rag" — The question asks about a player from the KNOWLEDGE BASE list, or asks about the content of our own documents/knowledge base (e.g. "Who is the player mentioned in this document?", "Which players do you have?"). Use this when the question refers to our stored knowledge or a player we already cover.

2. "wikipedia" — The question asks about a SPECIFIC real-world athlete or sports player that is NOT in the KNOWLEDGE BASE list, and reliable information would need to be fetched from Wikipedia. Use this ONLY when the query clearly names a specific person who is NOT in the list.

3. "general" — The question is NOT about a specific sports player (e.g. general programming, politics, medical advice, weather, anything unrelated). Never route to "wikipedia" for these.

Rules:
- Never pick "wikipedia" just because the query mentions sports in general.
- A famous player still gets "wikipedia" if they are NOT in the KNOWLEDGE BASE list.
- If in doubt about a sports question, prefer "rag".
- For questions about our own documents or knowledge base, pick "rag".

Respond with ONLY a JSON object and nothing else, in exactly this shape:
{"route": "rag" | "wikipedia" | "general", "player_name": "<specific player name or null>", "reason": "<short reason>"}"""


@dataclass
class RouteDecision:
    route: str = ROUTE_RAG
    player_name: str | None = None
    reason: str = ""


def _deterministic_match(query: str, all_players: list[str]) -> str | None:
    """Match the query to a stored player name without calling the LLM.

    Uses the existing fuzzy matcher first, then a lenient surname/token
    overlap check so short queries like "Tell me about Messi" still route
    to RAG deterministically.
    """
    match = _find_best_player_match(query, all_players)
    if match:
        return match

    query_tokens = set(re.findall(r"[a-z]+", query.lower()))
    if not query_tokens:
        return None

    for player in all_players:
        name_tokens = [
            token
            for token in re.findall(r"[a-z]+", _normalize_player_name(player))
            if len(token) >= 3
        ]
        if any(token in query_tokens for token in name_tokens):
            return player

    return None


def _extract_json(raw: str) -> dict | None:
    """Parse the classifier's response, tolerating markdown code fences."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


async def _llm_classify(query: str, available_players: list[str] | None = None) -> dict:
    """Tier-2 router: ask the LLM to classify the query into rag/wikipedia/general.

    The classifier is told exactly which players exist in the RAG knowledge
    base so it never assumes a famous player is already covered locally.
    Uses the existing retry wrapper to ride out transient API failures.
    """
    player_list = ", ".join(sorted(available_players or [])) or "empty"
    prompt = (
        f"User query:\n{query}\n\n"
        f"KNOWLEDGE BASE players: {player_list}\n\n"
        "Respond with ONLY the JSON object described in the system instructions."
    )
    result = await retry_call(
        lambda: ask_llm(prompt, system_prompt=ROUTER_SYSTEM_PROMPT),
        retries=2,
        delay=1.5,
    )
    raw = result["story"]
    parsed = _extract_json(raw) or {}
    route = parsed.get("route")
    if route not in VALID_ROUTES:
        route = ROUTE_RAG
    return {
        "route": route,
        "player_name": parsed.get("player_name"),
        "reason": parsed.get("reason") or "LLM router classification",
    }


async def route_query(query: str, retriever, classify=None) -> RouteDecision:
    """Decide which path serves the query: rag, wikipedia, or general.

    Tier 1 (deterministic, free): if the query mentions a player stored in
    the RAG knowledge base, route to RAG — no LLM call needed.
    Tier 2 (LLM classifier): otherwise classify the query with a single,
    small LLM call. The router NEVER generates the final answer.
    """
    query = (query or "").strip()
    if not query:
        return RouteDecision(route=ROUTE_GENERAL, reason="Empty query")

    all_players = await retriever.get_all_players()
    if all_players:
        match = _deterministic_match(query, all_players)
        if match:
            return RouteDecision(
                route=ROUTE_RAG,
                player_name=match,
                reason="Player present in RAG knowledge base",
            )

    classify = classify or _llm_classify
    try:
        result = classify(query, all_players)
        if inspect.isawaitable(result):
            result = await result
        route = result.get("route", ROUTE_RAG)
        if route not in VALID_ROUTES:
            route = ROUTE_RAG
        return RouteDecision(
            route=route,
            player_name=result.get("player_name"),
            reason=result.get("reason") or "LLM router classification",
        )
    except Exception as err:
        logger.warning("Router classification failed; defaulting to RAG: %s", err)
        return RouteDecision(route=ROUTE_RAG, reason="Router unavailable; defaulting to RAG")
