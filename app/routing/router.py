import inspect
import json
import logging
import re
from dataclasses import dataclass

from app.rag.retriever import _find_best_player_match, _normalize_player_name
from app.core.config import settings
from app.services.fallback_service import FALLBACK_MODEL
from app.services.llm_client import ask_llm
from app.services.retry_service import retry_call

logger = logging.getLogger(__name__)

ROUTE_RAG = "rag"
ROUTE_WIKIPEDIA = "wikipedia"
ROUTE_GENERAL = "general"
ROUTE_GITHUB = "github"

VALID_ROUTES = {ROUTE_RAG, ROUTE_WIKIPEDIA, ROUTE_GENERAL, ROUTE_GITHUB}

ROUTER_SYSTEM_PROMPT = """You are a query analyzer for a sports storytelling assistant.

The assistant has a local knowledge base (RAG) containing documents ONLY about the players listed under KNOWLEDGE BASE. Any other person is NOT covered locally.

Your job is to EXTRACT information from the user's question. You do NOT decide the routing — the system decides it from your extraction.

Return a JSON object with these fields:
1. "player_name": the specific athlete or sports figure the question is about, if one is named. Otherwise null.
2. "topic_class": one of:
   - "sports_player" — the question is about a SPECIFIC named athlete or sports figure (e.g. "Tell me about Virat Kohli", "What is Lionel Messi's net worth?").
   - "knowledge_base" — the question is about the assistant's own documents or stored knowledge (e.g. "Who is the player mentioned in this document?", "Which players do you have?").
   - "general" — anything else, including non-sports questions (programming, politics, weather, health) or sports questions that do NOT name a specific athlete.
3. "reason": a short explanation.

Rules:
- Only fill "player_name" with a real person's/athlete's name. Never fill it with programming languages, countries, teams, events, or concepts (e.g. Python, cricket, football, World Cup, Argentina).
- A famous athlete should be named in "player_name" even if they are NOT in the KNOWLEDGE BASE list.
- If the question names multiple athletes, fill "player_name" with the primary one.
- A technical or programming question is ALWAYS "general", never "knowledge_base" — the knowledge base only contains athlete biographies.

Examples:
- "How do I fix a Python syntax error?" -> {"player_name": null, "topic_class": "general"}
- "Tell me about Virat Kohli" -> {"player_name": "Virat Kohli", "topic_class": "sports_player"}
- "Who is the player mentioned in this document?" -> {"player_name": null, "topic_class": "knowledge_base"}

Respond with ONLY a JSON object and nothing else, in exactly this shape:
{"player_name": "<name or null>", "topic_class": "sports_player" | "knowledge_base" | "general", "reason": "<short reason>"}"""


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
    """Tier-2 router: extract the player name and topic class from the query.

    The classifier is told exactly which players exist in the RAG knowledge
    base so it never assumes a famous player is already covered locally.
    It only extracts facts — the routing decision is made deterministically
    by route_query using knowledge-base membership.
    """
    player_list = ", ".join(sorted(available_players or [])) or "empty"
    prompt = (
        f"User query:\n{query}\n\n"
        f"KNOWLEDGE BASE players: {player_list}\n\n"
        "Respond with ONLY the JSON object described in the system instructions."
    )
    result = await _classify_with_fallback(prompt)
    raw = result["story"]
    parsed = _extract_json(raw) or {}

    topic_class = parsed.get("topic_class")
    if topic_class not in {"sports_player", "knowledge_base", "general"}:
        # Backward compatibility: map the legacy "route" field to a topic class.
        route = parsed.get("route")
        if route == ROUTE_WIKIPEDIA:
            topic_class = "sports_player"
        elif route == ROUTE_GENERAL:
            topic_class = "general"
        else:
            topic_class = "knowledge_base"

    return {
        "topic_class": topic_class,
        "player_name": parsed.get("player_name"),
        "reason": parsed.get("reason") or "LLM router classification",
    }


async def _classify_with_fallback(prompt: str) -> dict:
    """Run the classifier through the primary model, falling back to Groq.

    The router needs an LLM to extract the player name; if Gemini is down,
    the same Groq fallback used for story generation keeps routing working.
    """
    last_err: Exception | None = None
    for model in (None, FALLBACK_MODEL):  # None -> DEFAULT_MODEL (Gemini)
        try:
            return await retry_call(
                lambda: ask_llm(prompt, system_prompt=ROUTER_SYSTEM_PROMPT, model=model),
                retries=2,
                delay=1.5,
            )
        except Exception as err:
            last_err = err
            logger.warning(
                "Router classifier failed on %s: %s",
                model or settings.DEFAULT_MODEL, err,
            )
    raise last_err


_SPORTS_SIGNALS = frozenset({
    "player", "players", "cricket", "football", "soccer", "tennis", "badminton",
    "athlete", "athletes", "sport", "sports", "match", "matches", "tournament",
    "championship", "championships", "olympic", "medal", "medals", "goal", "goals",
    "batsman", "bowler", "court", "pitch", "stadium", "trophy", "trophies",
    "career", "net worth", "jersey", "team", "captain", "coach", "score", "scored",
    "innings", "wicket", "ace", "grand slam", "world cup", "series", "t20", "odi",
    "league", "champion", "champions", "final", "finalist", "racket", "batting",
    "bowling", "fielding", "goalkeeper", "defender", "striker", "midfielder",
    "volley", "serve", "serving", "set point", "match point", "olympics", "paralympics",
})

_KB_TRIGGERS = frozenset({
    "document", "documents", "knowledge base", "which players", "players do you have",
    "player list", "your stories", "available players", "stored data",
})

_NON_SPORTS_SIGNALS = frozenset({
    "python", "javascript", "java", "html", "css", "sql", "code", "coding",
    "program", "programming", "syntax", "bug", "function", "variable", "class",
    "algorithm", "api", "software", "computer", "windows", "linux", "mac",
    "math", "chemistry", "physics", "biology", "recipe", "cooking", "weather",
    "politics", "president", "election", "medicine", "doctor", "medical", "symptom",
    "stock", "invest", "investing", "economy", "history of", "geography", "capital",
})

_GITHUB_INTENT_PATTERNS = [
    re.compile(r"github", re.IGNORECASE),
    re.compile(r"\brepositories\b|\brepository\b|\brepos?\b", re.IGNORECASE),
    re.compile(r"pull request", re.IGNORECASE),
    re.compile(r"readme", re.IGNORECASE),
    re.compile(r"(code search|search code)", re.IGNORECASE),
    # owner/repo (owner must start with a letter, avoids "183/183")
    re.compile(r"(?<![a-zA-Z0-9])[a-zA-Z][\w.-]*/[\w.-]+"),
]


def _has_github_intent(query: str) -> bool:
    return any(p.search(query) for p in _GITHUB_INTENT_PATTERNS)


def _contains_any(text_lower: str, keywords) -> bool:
    return any(kw in text_lower for kw in keywords)


async def route_query(query: str, retriever, classify=None) -> RouteDecision:
    """Decide which path serves the query: rag, wikipedia, or general.

    Tier 1 (deterministic, free): if the query mentions a player stored in
    the RAG knowledge base, route to RAG — no LLM call needed.
    Tier 1b (deterministic, free): clearly non-sports queries are declined
    before any LLM call — no random player content is served for them.
    Tier 2 (LLM classifier): otherwise extract the player name + topic class
    and route by knowledge-base membership. The router never generates the
    final answer.
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

    # GitHub tool intent: the query asks about the user's GitHub repos/code.
    # Checked before the non-sports prefilter so e.g. "search my code" reaches
    # the GitHub tool path instead of being declined as general.
    if _has_github_intent(query):
        return RouteDecision(
            route=ROUTE_GITHUB,
            reason="Query targets GitHub repositories or code",
        )

    query_lower = query.lower()
    has_sports_signal = _contains_any(query_lower, _SPORTS_SIGNALS)
    has_kb_trigger = _contains_any(query_lower, _KB_TRIGGERS)
    if not has_sports_signal and not has_kb_trigger and _contains_any(query_lower, _NON_SPORTS_SIGNALS):
        return RouteDecision(
            route=ROUTE_GENERAL,
            reason="Query is unrelated to sports (deterministic prefilter)",
        )

    classify = classify or _llm_classify
    try:
        result = classify(query, all_players)
        if inspect.isawaitable(result):
            result = await result
        topic_class = result.get("topic_class")
        player_name = result.get("player_name")
        reason = result.get("reason") or "LLM router classification"

        # Backward compatibility: map a legacy "route" field to a topic class.
        if topic_class not in {"sports_player", "knowledge_base", "general"}:
            legacy_route = result.get("route")
            if legacy_route == ROUTE_WIKIPEDIA:
                topic_class = "sports_player"
            elif legacy_route == ROUTE_GENERAL:
                topic_class = "general"
            else:
                topic_class = "knowledge_base"

        if topic_class == "general":
            return RouteDecision(route=ROUTE_GENERAL, reason=reason)

        if topic_class == "sports_player" and player_name:
            # Deterministic check: is the named athlete in the RAG knowledge base?
            kb_match = (
                _find_best_player_match(str(player_name), all_players)
                if all_players
                else None
            )
            if kb_match:
                return RouteDecision(
                    route=ROUTE_RAG,
                    player_name=kb_match,
                    reason=f"{reason} (player present in RAG knowledge base)",
                )
            return RouteDecision(
                route=ROUTE_WIKIPEDIA,
                player_name=str(player_name),
                reason=f"{reason} (player not in RAG knowledge base)",
            )

        # knowledge_base queries or anything ambiguous → RAG
        return RouteDecision(route=ROUTE_RAG, reason=reason)
    except Exception as err:
        logger.warning("Router classification failed; defaulting to RAG: %s", err)
        return RouteDecision(route=ROUTE_RAG, reason="Router unavailable; defaulting to RAG")
