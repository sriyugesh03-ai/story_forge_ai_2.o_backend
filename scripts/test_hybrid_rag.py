import asyncio
import os
import sys

# Configure stdout to handle UTF-8 encoding safely on Windows command prompt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from app.routing.router import (
    ROUTE_GENERAL,
    ROUTE_RAG,
    ROUTE_WIKIPEDIA,
    route_query,
)
from app.tools.wikipedia_tool import fetch_wikipedia, get_wikipedia_player

PLAYERS = ["Lionel_Messi", "Cristiano_Ronaldo", "MS_Dhoni", "Carlos_Alcaraz"]


class FakeRetriever:
    """In-memory retriever used to test the router without MongoDB."""

    def __init__(self, players, docs=None):
        self._players = players
        self._docs = docs or []

    async def get_all_players(self):
        return self._players

    async def ainvoke(self, query):
        return self._docs

    async def retrieve_docs_for_player(self, player_name, top_k=5):
        return self._docs


RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    status = "PASS" if cond else "FAIL"
    print(f"{status}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


async def test_router():
    # ── RAG route: player exists in RAG ────────────────────────────────
    d = await route_query("Tell me about Lionel Messi's career", FakeRetriever(PLAYERS))
    check("Router: player in RAG -> rag", d.route == ROUTE_RAG and d.player_name == "Lionel_Messi", str(d))

    d = await route_query("Tell me about Messi", FakeRetriever(PLAYERS))
    check("Router: short name 'Messi' -> rag", d.route == ROUTE_RAG and d.player_name == "Lionel_Messi", str(d))

    # ── Wikipedia route: player NOT in RAG ─────────────────────────────
    stub = lambda q, players=None: {"topic_class": "sports_player", "player_name": "Virat Kohli", "reason": "unknown player"}
    d = await route_query("Tell me about Virat Kohli's career", FakeRetriever(PLAYERS), classify=stub)
    check("Router: unknown player -> wikipedia", d.route == ROUTE_WIKIPEDIA and d.player_name == "Virat Kohli", str(d))

    # ── Deterministic guard: classifier says rag for a KB player -> rag ─
    stub = lambda q, players=None: {"topic_class": "sports_player", "player_name": "Lionel Messi", "reason": "sports"}
    d = await route_query("Tell me about Lionel Messi's career", FakeRetriever(PLAYERS), classify=stub)
    check("Router: named player IN knowledge base -> rag", d.route == ROUTE_RAG and d.player_name == "Lionel_Messi", str(d))

    # ── Legacy route field is still honoured ───────────────────────────
    stub = lambda q, players=None: {"route": "wikipedia", "player_name": "Virat Kohli", "reason": "legacy"}
    d = await route_query("Tell me about Virat Kohli's career", FakeRetriever(PLAYERS), classify=stub)
    check("Router: legacy 'route' field maps to wikipedia", d.route == ROUTE_WIKIPEDIA, str(d))

    # ── General route: unrelated query (no Wikipedia call) ─────────────
    stub = lambda q, players=None: {"topic_class": "general", "player_name": None, "reason": "unrelated"}
    d = await route_query("How does Python list comprehension work?", FakeRetriever(PLAYERS), classify=stub)
    check("Router: unrelated query -> general", d.route == ROUTE_GENERAL, str(d))

    # ── Document-content query -> rag ──────────────────────────────────
    stub = lambda q, players=None: {"topic_class": "knowledge_base", "player_name": None, "reason": "about documents"}
    d = await route_query("Who is the player mentioned in this document?", FakeRetriever(PLAYERS), classify=stub)
    check("Router: document question -> rag", d.route == ROUTE_RAG, str(d))

    # ── Classifier failure -> graceful default to rag ──────────────────
    d = await route_query("some query", FakeRetriever(PLAYERS), classify=lambda q, players=None: (_ for _ in ()).throw(RuntimeError("boom")))
    check("Router: classifier failure -> default rag", d.route == ROUTE_RAG, str(d))

    # ── Deterministic prefilter: clearly non-sports query declines w/o LLM ─
    d = await route_query("How do I fix a Python syntax error?", FakeRetriever(PLAYERS))
    check("Router: prefilter declines non-sports query", d.route == ROUTE_GENERAL, str(d))

    # ── Sports query is NOT caught by the prefilter (goes to LLM) ──────
    stub = lambda q, players=None: {"topic_class": "general", "player_name": None, "reason": "no specific athlete"}
    d = await route_query("Tell me about the cricket World Cup", FakeRetriever(PLAYERS), classify=stub)
    check("Router: sports query skips prefilter", d.route == ROUTE_GENERAL, str(d))


def test_wikipedia_tool():
    # ── Tool is a proper LangChain tool ────────────────────────────────
    check("Wikipedia tool is a LangChain tool", isinstance(get_wikipedia_player, BaseTool))

    # ── Known player fetch ─────────────────────────────────────────────
    r = fetch_wikipedia("Virat Kohli")
    check("Wikipedia tool: known player fetched", r.ok and "Kohli" in r.title and len(r.content) > 100, str(r)[:150])

    # ── Invalid input ──────────────────────────────────────────────────
    r = fetch_wikipedia("")
    check("Wikipedia tool: empty input -> error", not r.ok and bool(r.error), r.error)

    # ── Missing page ───────────────────────────────────────────────────
    r = fetch_wikipedia("Aqzxvqzx Unknown Player 987654321")
    graceful = not isinstance(r, Exception) and (r.ok or ("Could not find" in r.error or "unavailable" in r.error))
    check("Wikipedia tool: missing page handled gracefully", graceful, r.error)


async def test_generate_story():
    """Verify the orchestrator: final LLM receives query + correct context."""
    import app.services.chat as chat_mod
    import app.routing.router as router_mod

    captured = {}
    original_get_retriever = chat_mod.get_retriever
    original_ask_llm = chat_mod.ask_llm
    original_router_ask_llm = router_mod.ask_llm

    def fake_ask_llm(prompt, model=None, system_prompt=None):
        captured["prompt"] = prompt
        return "FAKE_STORY_ANSWER"

    def fake_router_ask_llm(prompt, model=None, system_prompt=None):
        return '{"topic_class": "general", "player_name": null, "reason": "unrelated"}'

    try:
        chat_mod.get_retriever = lambda: FakeRetriever(PLAYERS)
        chat_mod.ask_llm = fake_ask_llm
        router_mod.ask_llm = fake_router_ask_llm

        # ── RAG path: LLM receives RAG context ─────────────────────────
        docs = [Document(page_content="RAG_FAKE_CONTENT_MARKER_XYZ " * 5)]
        chat_mod.get_retriever = lambda: FakeRetriever(PLAYERS, docs=docs)
        resp = await chat_mod.generate_story("Tell me about Lionel Messi", "biography")
        check("RAG path: selected route=rag", resp.get("route") == ROUTE_RAG and resp.get("source") == "RAG", str(resp.get("route")))
        prompt = captured.get("prompt", "")
        check("RAG path: LLM received the user query", "Lionel Messi" in prompt)
        check("RAG path: LLM received the RAG context", "RAG_FAKE_CONTENT_MARKER_XYZ" in prompt)

        # ── Empty RAG result + known player -> Wikipedia fallback ──────
        captured.clear()
        chat_mod.get_retriever = lambda: FakeRetriever(PLAYERS, docs=[])  # empty retrieval
        resp = await chat_mod.generate_story("Tell me about Lionel Messi", "biography")
        check("Empty RAG: falls back to Wikipedia", resp.get("source") == "Wikipedia" and resp.get("route") == ROUTE_WIKIPEDIA, str(resp.get("source")))
        prompt = captured.get("prompt", "")
        check("Wikipedia fallback: LLM received the query", "Lionel Messi" in prompt)
        check("Wikipedia fallback: LLM received wikipedia content", len(prompt) > 500)

        # ── General query: graceful decline, no LLM story call ─────────
        captured.clear()
        chat_mod.get_retriever = lambda: FakeRetriever(PLAYERS, docs=[])
        resp = await chat_mod.generate_story("How does Python list comprehension work?", "biography")
        check("General query: graceful decline", resp.get("route") == ROUTE_GENERAL and "can't help" in resp.get("story", ""), str(resp.get("route")))
        check("General query: no story LLM call made", "prompt" not in captured)

        # ── Wikipedia path: LLM receives wikipedia content ─────────────
        captured.clear()
        chat_mod.get_retriever = lambda: FakeRetriever(PLAYERS, docs=[])
        router_mod.ask_llm = lambda prompt, model=None, system_prompt=None: (
            '{"topic_class": "sports_player", "player_name": "Virat Kohli", "reason": "unknown"}'
        )
        resp = await chat_mod.generate_story("Tell me about Virat Kohli's career", "timeline")
        check("Wikipedia path: selected route=wikipedia", resp.get("route") == ROUTE_WIKIPEDIA and resp.get("source") == "Wikipedia", str(resp.get("route")))
        prompt = captured.get("prompt", "")
        check("Wikipedia path: LLM received the query", "Virat Kohli" in prompt)
        check("Wikipedia path: LLM received wikipedia content", "born" in prompt.lower() or "cricketer" in prompt.lower())

    finally:
        chat_mod.get_retriever = original_get_retriever
        chat_mod.ask_llm = original_ask_llm
        router_mod.ask_llm = original_router_ask_llm


async def test_router_live_llm():
    """Optional live test: real Gemini classifier for an unknown player."""
    print("\n[Live] Running real LLM router classification for 'Virat Kohli'...")
    try:
        d = await route_query("Tell me about Virat Kohli's career", FakeRetriever(PLAYERS))
        print(f"  -> route={d.route}, player={d.player_name}, reason={d.reason}")
        check("Live router: Virat Kohli classified as wikipedia", d.route == ROUTE_WIKIPEDIA, str(d))
    except Exception as e:
        print(f"  [SKIP] live router test failed: {e}")


async def main():
    await test_router()
    test_wikipedia_tool()
    await test_generate_story()

    if "--live" in sys.argv:
        await test_router_live_llm()

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\nRESULT: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())