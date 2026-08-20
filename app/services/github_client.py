"""GitHub REST API access (used as the fallback when the remote MCP call fails).

This module is the ONLY place that issues raw requests against api.github.com
for tool data. It never sees or logs the OAuth client secret.
"""

import json
import logging
import urllib.parse
import urllib.request

from app.core.config import settings

logger = logging.getLogger(__name__)

_USER_AGENT = "story-forge-ai/1.0"
_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": _USER_AGENT,
    "X-GitHub-Api-Version": "2022-11-28",
}


class GitHubApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class GithubRestClient:
    """Thin sync client over the GitHub REST API. Methods return plain dicts.

    Instances are cheap and hold a single user's token in memory only for the
    duration of one tool call (the token is passed by the caller).
    """

    def __init__(self, access_token: str):
        self._token = access_token

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{settings.GITHUB_API_URL}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                **_HEADERS,
                "Authorization": f"Bearer {self._token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            raise GitHubApiError("GitHub API request failed.", status=err.code)
        except urllib.error.URLError as err:
            logger.warning("GitHub REST network error: %s", err.reason)
            raise GitHubApiError("Could not reach the GitHub API.")
        try:
            return json.loads(body)
        except ValueError:
            raise GitHubApiError("Unexpected response from the GitHub API.")

    # ── Tool implementations (mirror the allowed MCP tool set) ──────────

    def list_repositories(self, owner: str | None = None, visibility: str | None = None) -> dict:
        if owner:
            return {"repositories": self._get(f"/users/{owner}/repos")}
        params = {"per_page": 30}
        if visibility:
            params["visibility"] = visibility
        return {"repositories": self._get("/user/repos", params)}

    def get_file_contents(self, owner: str, repo: str, path: str) -> dict:
        data = self._get(f"/repos/{owner}/{repo}/contents/{path.strip('/')}")
        if isinstance(data, dict) and data.get("content"):
            import base64

            try:
                data = {**data, "content": base64.b64decode(data["content"]).decode("utf-8", "replace")}
            except Exception:
                pass
        return {"file": data}

    def search_repositories(self, query: str, per_page: int = 10) -> dict:
        return {"results": self._get("/search/repositories", {"q": query, "per_page": per_page}).get("items", [])}

    def list_issues(self, owner: str, repo: str, state: str = "open") -> dict:
        return {"issues": self._get(f"/repos/{owner}/{repo}/issues", {"state": state, "per_page": 20})}

    def get_issue(self, owner: str, repo: str, issue_number: int) -> dict:
        return {"issue": self._get(f"/repos/{owner}/{repo}/issues/{int(issue_number)}")}

    def call(self, tool: str, arguments: dict) -> dict:
        import inspect

        mapping = {
            "list_repositories": self.list_repositories,
            "get_file_contents": self.get_file_contents,
            "search_repositories": self.search_repositories,
            "list_issues": self.list_issues,
            "get_issue": self.get_issue,
        }
        fn = mapping.get(tool)
        if fn is None:
            raise GitHubApiError(f"Unknown tool: {tool}")

        # Forward only arguments the REST method accepts, mapping MCP-style
        # names onto the REST equivalents so both tool paths stay compatible.
        aliases = {"limit": "per_page", "perPage": "per_page", "per_page": "per_page"}
        params = inspect.signature(fn).parameters
        kwargs = {}
        for key, value in (arguments or {}).items():
            mapped = aliases.get(key, key)
            if mapped in params:
                kwargs[mapped] = value
        return fn(**kwargs)