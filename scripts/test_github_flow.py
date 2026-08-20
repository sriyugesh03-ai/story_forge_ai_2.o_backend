import asyncio
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx
import urllib.parse

import app.auth.github as github_auth
import app.routes.auth as auth_routes
import app.tools.github_tool as github_tool
from app.db.mongo_db import connect_to_mongo, close_mongo_connection
from app.main import app
from app.repo.github_repo import GithubRepository
from app.services.github_mcp import McpCallError

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


def parse_query(url):
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


async def main():
    # Hermetic: dummy OAuth credentials (never real secrets) so the routes work.
    from app.core.config import settings as app_settings

    app_settings.GITHUB_CLIENT_ID = "test_client_id"
    app_settings.GITHUB_CLIENT_SECRET = "test_client_secret"
    app_settings.GITHUB_REDIRECT_URI = "http://test/auth/github/callback"

    # ── Hermetic: encryption + state ──────────────────────────────────
    enc = github_auth.encrypt_token("gho_test_token_123")
    check("crypto: encrypt != raw", enc != "gho_test_token_123")
    check("crypto: decrypt roundtrip", github_auth.decrypt_token(enc) == "gho_test_token_123")
    sid = github_auth.create_state("user_x")
    check("state: verify valid", github_auth.verify_state(sid) == "user_x")
    check("state: rejects tampered", github_auth.verify_state(sid[:-1] + ("A" if sid[-1] != "A" else "B")) is None)
    check("state: rejects garbage", github_auth.verify_state("garbage") is None)

    await connect_to_mongo()
    try:
        repo = GithubRepository()
        clerk_user_id = "clerk_gh_" + uuid.uuid4().hex[:10]
        fake_email = clerk_user_id + "@example.com"

        # Create the app user via the Clerk exchange (mocked verification).
        auth_routes.verify_clerk_token = lambda token: {
            "sub": clerk_user_id, "email": fake_email,
            "iss": "https://national-feline-6121.clerk.accounts.dev", "exp": 9999999999,
        }

        # Mock GitHub network calls.
        def fake_exchange(code):
            check("exchange: code forwarded", code == "fakecode")
            return "gho_mock_token", ["read:user", "repo"]

        def fake_fetch(token):
            check("fetch: used exchanged token", token == "gho_mock_token")
            return {"github_user_id": 42, "github_username": "octo-test"}

        revoked = {"called": False}

        def fake_revoke(token):
            revoked["called"] = True
            check("revoke: used decrypted token", token == "gho_mock_token")

        original_exchange = github_auth.exchange_github_code
        original_fetch = github_auth.fetch_github_user
        original_revoke = github_auth.revoke_github_token
        github_auth.exchange_github_code = fake_exchange
        github_auth.fetch_github_user = fake_fetch
        github_auth.revoke_github_token = fake_revoke

        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Unauthenticated access
                check("start: 401 without token", (await client.get("/auth/github")).status_code == 401)

                r = await client.post("/auth/clerk", json={"clerk_token": "fake.token"})
                access = r.json()["access_token"]
                hdr = {"Authorization": f"Bearer {access}"}

                # Start OAuth
                r = await client.get("/auth/github", headers=hdr)
                check("start: 200 + auth_url", r.status_code == 200 and r.json().get("auth_url", "").startswith("https://github.com/login/oauth/authorize"), str(r.text[:150]))
                auth_url = r.json()["auth_url"]
                qs = parse_query(auth_url)
                state = qs.get("state", [""])[0]
                check("start: auth_url has state", bool(state))
                check("start: auth_url has redirect_uri", qs.get("redirect_uri", [""])[0] == github_auth.settings.GITHUB_REDIRECT_URI)
                check("start: auth_url has client_id", bool(qs.get("client_id", [""])[0]))

                # Callback with valid state + code
                r = await client.get(f"/auth/github/callback?code=fakecode&state={state}", follow_redirects=False)
                loc = r.headers.get("location", "")
                check("callback: 307 redirect", r.status_code in (302, 307) and "settings?github=connected" in loc, loc)

                # Callback with a bad state
                r = await client.get("/auth/github/callback?code=x&state=badstate", follow_redirects=False)
                check("callback: invalid state", "github_error=invalid_state" in r.headers.get("location", ""))

                # Reuse of the same state must fail (single-use)
                r = await client.get(f"/auth/github/callback?code=fakecode2&state={state}", follow_redirects=False)
                check("callback: state reuse rejected", "github_error=state_mismatch" in r.headers.get("location", ""), r.headers.get("location", ""))

                # Status after connect
                r = await client.get("/auth/github/status", headers=hdr)
                st = r.json()
                check("status: connected", st["connected"] and st["github_username"] == "octo-test" and st["github_user_id"] == 42, str(st))

                # Stored token is encrypted
                conn = await repo.find_by_user_id((await client.get("/auth/me", headers=hdr)).json()["id"])
                check("storage: encrypted at rest", conn and conn["access_token_enc"] != "gho_mock_token", str(conn)[:200])

                # Tool endpoint without a connection (fresh user)
                r2 = await client.post("/auth/clerk", json={"clerk_token": "fake.token"})
                hdr2 = {"Authorization": f"Bearer {r2.json()['access_token']}"}
                r = await client.post("/github/mcp/tool", headers=hdr2, json={"tool": "search_repositories", "arguments": {"query": "pydantic"}})
                check("tool: 409 without connection", r.status_code == 409, str(r.text[:120]))

                # Tool endpoint with mocked MCP success
                async def fake_mcp_ok(token, tool, arguments):
                    check("mcp: called with user token", token == "gho_mock_token")
                    check("mcp: logical name mapped to server name", tool == "search_repositories")
                    return {"text": "[{full_name: octo-test/demo}]", "is_error": False}

                original_mcp = github_tool.mcp_call_tool
                github_tool.mcp_call_tool = fake_mcp_ok
                r = await client.post("/github/mcp/tool", headers=hdr, json={"tool": "search_repositories", "arguments": {"query": "demo"}})
                check("tool: via mcp", r.status_code == 200 and r.json()["via"] == "mcp" and "octo-test" in r.json()["data"]["text"], str(r.text[:160]))

                # list_repositories has no hosted-MCP equivalent -> always REST
                def fake_rest(token, tool, arguments):
                    return {"result": {"rest": "repos"}}

                original_rest = github_tool._rest_call
                github_tool._rest_call = fake_rest
                r = await client.post("/github/mcp/tool", headers=hdr, json={"tool": "list_repositories", "arguments": {"limit": 3}})
                check("tool: list_repositories REST-only", r.status_code == 200 and r.json()["via"] == "rest" and r.json()["data"] == {"result": {"rest": "repos"}}, str(r.text[:160]))

                # Tool endpoint with MCP transport failure -> REST fallback
                async def fake_mcp_fail(token, tool, arguments):
                    raise McpCallError("transport down")

                github_tool.mcp_call_tool = fake_mcp_fail
                r = await client.post("/github/mcp/tool", headers=hdr, json={"tool": "search_repositories", "arguments": {"query": "x"}})
                check("tool: falls back to REST", r.status_code == 200 and r.json()["via"] == "rest" and r.json()["data"] == {"result": {"rest": "repos"}}, str(r.text[:160]))
                github_tool.mcp_call_tool = original_mcp
                github_tool._rest_call = original_rest

                # Disallowed tool
                r = await client.post("/github/mcp/tool", headers=hdr, json={"tool": "create_or_update_file", "arguments": {}})
                check("tool: disallowed rejected", r.status_code == 409, str(r.text[:120]))

                # Disconnect
                r = await client.post("/auth/github/disconnect", headers=hdr)
                check("disconnect: 200", r.status_code == 200)
                check("disconnect: revoke called", revoked["called"])
                r = await client.get("/auth/github/status", headers=hdr)
                check("disconnect: status disconnected", r.json()["connected"] is False)

        finally:
            github_auth.exchange_github_code = original_exchange
            github_auth.fetch_github_user = original_fetch
            github_auth.revoke_github_token = original_revoke
    finally:
        await close_mongo_connection()

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\nRESULT: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


async def live_mcp_probe():
    """Optional: verify the hosted remote MCP endpoint accepts our PAT."""
    import app.services.github_mcp as m
    from app.core.config import settings

    token = os.getenv("GITHUB_PAT_TOKEN", "")
    if not token:
        print("SKIP: GITHUB_PAT_TOKEN not set")
        return
    try:
        tools = await m.mcp_list_tools(token)
        print(f"LIVE MCP OK: {len(tools)} tools exposed by {settings.GITHUB_MCP_URL}")
        print("sample:", sorted(tools)[:8])
    except Exception as e:
        print(f"LIVE MCP FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if "--live" in sys.argv:
        asyncio.run(live_mcp_probe())
    else:
        asyncio.run(main())