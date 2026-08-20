"""GitHub tool planner — turns a user request into ONE GitHub tool call.

Mirrors app/routing/router.py's LLM-first-with-fallback style: a cheap LLM
classifier plans the call, and the plan is validated against the allowed tool
set so the agent can never invoke a disallowed or malformed tool.
"""

import json
import logging
import re

from app.core.config import settings
from app.services.fallback_service import FALLBACK_MODEL
from app.services.llm_client import ask_llm
from app.services.retry_service import retry_call
from app.tools.github_tool import allowed_tools

logger = logging.getLogger(__name__)

_PLANNER_PROMPT = """You plan a single GitHub API call that answers the user's request.
Choose exactly one tool from the available list and fill in its arguments.

Available tools and their arguments:
- search_repositories: {"query": "keywords to search GitHub repositories"}
- list_repositories: {}  (lists the logged-in user's own repositories)
- get_file_contents: {"owner": "repo owner", "repo": "repo name", "path": "file path"}
- list_issues: {"owner": "repo owner", "repo": "repo name", "state": "open|closed|all"}
- get_issue: {"owner": "repo owner", "repo": "repo name", "issue_number": <number>}

Rules:
- If the request names "<owner>/<repo>", extract both into owner and repo.
- Otherwise, when the tool needs an owner and the user means their own account, use the given default owner.
- If the request is too vague to identify a repo, file, or issue, prefer list_repositories or search_repositories.
- Never invent repositories or issue numbers.
- If no tool fits, respond {"tool": "none", "arguments": {}}.

Respond with ONLY JSON: {"tool": "<name or 'none'>", "arguments": {...}, "reason": "short explanation"}"""


def _extract_json(raw: str) -> dict | None:
    """Parse the planner's response, tolerating markdown code fences."""
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


def _fill_defaults(tool: str, args: dict, default_owner: str | None) -> dict | None:
    """Fill missing owner with the user's login; None means the plan is unusable."""
    args = dict(args or {})
    if tool in {"get_file_contents", "list_issues", "get_issue"}:
        if not args.get("owner"):
            if not default_owner:
                return None
            args["owner"] = default_owner
        if not args.get("repo"):
            return None
    if tool == "get_file_contents" and not args.get("path"):
        args["path"] = "README.md"
    if tool == "search_repositories" and not args.get("query"):
        return None
    if tool == "get_issue":
        try:
            args["issue_number"] = int(args["issue_number"])
        except (TypeError, ValueError):
            return None
    return args


async def _plan_with_model(prompt: str, model: str | None) -> dict | None:
    result = await retry_call(
        lambda: ask_llm(prompt, system_prompt=_PLANNER_PROMPT, model=model),
        retries=2,
        delay=1.5,
    )
    return _extract_json(result["story"])


async def plan_github_call(query: str, default_owner: str | None) -> dict | None:
    """Return {"tool", "arguments"} to answer a GitHub request, or None.

    The plan is validated against the allowed tool set and required arguments
    before it is returned, so downstream callers can trust it.
    """
    prompt = (
        f"User request:\n{query}\n\n"
        f"Default owner (the logged-in GitHub user): {default_owner or 'unknown'}\n\n"
        "Respond with ONLY the JSON described in the system instructions."
    )
    last_err: Exception | None = None
    for model in (None, FALLBACK_MODEL):
        try:
            plan = await _plan_with_model(prompt, model)
        except Exception as err:
            last_err = err
            logger.warning("GitHub planner failed on %s: %s", model or settings.DEFAULT_MODEL, err)
            continue
        if not plan:
            continue
        tool = plan.get("tool")
        if tool == "none" or tool not in allowed_tools():
            return None
        args = _fill_defaults(tool, plan.get("arguments"), default_owner)
        if args is None:
            return None
        return {"tool": tool, "arguments": args}
    if last_err:
        logger.warning("GitHub planner unavailable: %s", last_err)
    return None