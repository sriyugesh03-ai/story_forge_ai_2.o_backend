import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import app.services.chat as chat
import app.services.github_planner as planner
from app.routing.router import (
    ROUTE_GENERAL,
    ROUTE_GITHUB,
    ROUTE_RAG,
    RouteDecision,
    route_query,
)
from app.tools.github_tool import allowed_tools

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


class FakeRetriever:
    def __init__(self, players=None):
        self._players = players or []

    async def get_all_players(self):
        return self._players

    async def ainvoke(self, query):
        return []

    async def retrieve_docs_for_player(self, player_name, top_k=5):
        return []


async def main():
    # ── Router: GitHub intent → ROUTE_GITHUB ─────────────────────────────
    r = await route_query("list my repositories", FakeRetriever())
    check("router: 'list my repositories' -> github", r.route == ROUTE_GITHUB, r.route)
    r = await route_query("show the README of pydantic/pydantic", FakeRetriever())
    check("router: 'README owner/repo' -> github", r.route == ROUTE_GITHUB, r.route)
    r = await route_query("list issues in pydantic/pydantic", FakeRetriever())
    check("router: 'issues owner/repo' -> github", r.route == ROUTE_GITHUB, r.route)
    r = await route_query("search code for retry logic", FakeRetriever())
    check("router: 'search code' -> github", r.route == ROUTE_GITHUB, r.route)

    # ── Router: non-GitHub queries unaffected ────────────────────────────
    stub_general = lambda q, players=None: {"topic_class": "general", "player_name": None, "reason": "stub"}
    r = await route_query("give me the match report", FakeRetriever(), classify=stub_general)
    check("router: 'match report' NOT github", r.route != ROUTE_GITHUB, r.route)
    r = await route_query("what is 183/183", FakeRetriever(), classify=stub_general)
    check("router: '183/183' NOT github", r.route != ROUTE_GITHUB, r.route)
    r = await route_query("how does python list comprehension work", FakeRetriever())
    check("router: python question still general", r.route == ROUTE_GENERAL, r.route)

    # ── Planner: validated tool selection with mocked LLM ────────────────
    def fake_llm_json(payload):
        planner.ask_llm = lambda prompt, **kw: payload
        return payload

    fake_llm_json('{"tool": "get_file_contents", "arguments": {"owner": "pydantic", "repo": "pydantic", "path": "README.md"}, "reason": "x"}')
    plan = await planner.plan_github_call("show README of pydantic/pydantic", "octo-test")
    check("planner: picks get_file_contents", plan and plan["tool"] == "get_file_contents" and plan["arguments"]["path"] == "README.md", str(plan))

    fake_llm_json('{"tool": "get_file_contents", "arguments": {"repo": "demo", "path": "pyproject.toml"}, "reason": "x"}')
    plan = await planner.plan_github_call("show pyproject of my demo repo", "octo-test")
    check("planner: fills default owner", plan and plan["arguments"].get("owner") == "octo-test", str(plan))

    fake_llm_json('{"tool": "list_repositories", "arguments": {}, "reason": "x"}')
    plan = await planner.plan_github_call("list my repos", "octo-test")
    check("planner: list_repositories", plan and plan["tool"] == "list_repositories", str(plan))

    fake_llm_json('{"tool": "none", "arguments": {}, "reason": "vague"}')
    plan = await planner.plan_github_call("what is github", "octo-test")
    check("planner: rejects 'none'", plan is None, str(plan))

    fake_llm_json('{"tool": "create_or_update_file", "arguments": {}, "reason": "x"}')
    plan = await planner.plan_github_call("write a file", "octo-test")
    check("planner: rejects disallowed tool", plan is None, str(plan))

    fake_llm_json('{"tool": "get_file_contents", "arguments": {"repo": "demo"}, "reason": "missing owner+default"}')
    plan = await planner.plan_github_call("read a file in demo", None)
    check("planner: unusable plan -> None", plan is None, str(plan))

    # ── generate_story: GITHUB path end-to-end (all deps mocked) ────────
    original = {
        "get_retriever": chat.get_retriever,
        "route_query": chat.route_query,
        "get_connected_username": chat.get_connected_username,
        "plan_github_call": chat.plan_github_call,
        "call_github_tool": chat.call_github_tool,
        "retry_call": chat.retry_call,
    }

    async def fake_route(*a, **k):
        return RouteDecision(route=ROUTE_GITHUB)

    async def fake_owner(uid):
        return "octo-test"

    async def fake_plan(q, owner):
        return {"tool": "list_repositories", "arguments": {}}

    async def fake_call(uid, tool, args):
        return {"via": "rest", "data": {"result": {"repositories": ["octo-test/demo"]}}}

    async def fake_retry(func, **kw):
        return {"story": "Here are your repositories.", "retry_count": 0, "fallback_used": False}

    chat.get_retriever = lambda: FakeRetriever()
    chat.route_query = fake_route
    chat.get_connected_username = fake_owner
    chat.plan_github_call = fake_plan
    chat.call_github_tool = fake_call
    chat.retry_call = fake_retry

    resp = await chat.generate_story("list my repositories", "biography", user_id="user-1")
    check("story: routes to github", resp["route"] == ROUTE_GITHUB, resp["route"])
    check("story: source labels via", resp["source"] == "GitHub/rest", resp["source"])
    check("story: answer generated", resp["story"] == "Here are your repositories.", resp["story"])
    check("story: context fed to LLM", resp["evaluation"]["retrieved_chunks"] == 1)

    # Not connected
    async def no_owner(uid):
        return None

    chat.get_connected_username = no_owner
    resp = await chat.generate_story("list my repositories", "biography", user_id="user-1")
    check("story: not connected -> connect message", "Connect your GitHub account" in resp["story"], resp["story"])

    # Planner could not map request
    chat.get_connected_username = fake_owner

    async def no_plan(q, owner):
        return None

    chat.plan_github_call = no_plan
    resp = await chat.generate_story("list my repositories", "biography", user_id="user-1")
    check("story: planner miss -> helpful message", "couldn't map" in resp["story"], resp["story"])

    # Missing user_id
    chat.plan_github_call = fake_plan
    resp = await chat.generate_story("list my repositories", "biography", user_id=None)
    check("story: no user -> login message", "log in" in resp["story"], resp["story"])

    for name, value in original.items():
        setattr(chat, name, value)

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\nRESULT: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())