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

import app.routes.auth as auth_routes
from app.auth.tokens import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_refresh_token,
)
from app.db.mongo_db import connect_to_mongo, close_mongo_connection
from app.main import app
from app.repo.users_repo import UsersRepository

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


async def find_user(repo, clerk_id):
    return await repo.find_by_clerk_id(clerk_id)


async def main():
    # ── Hermetic token logic ─────────────────────────────────────────
    user_id = str(uuid.uuid4())
    token = create_access_token(user_id)
    payload = decode_access_token(token)
    check("tokens: access token decodes", payload is not None and payload["sub"] == user_id and payload["type"] == "access", str(payload))
    check("tokens: garbage token rejected", decode_access_token("not.a.token") is None)
    check("tokens: refresh token random & long", len(generate_refresh_token()) >= 80 and generate_refresh_token() != generate_refresh_token())
    raw = generate_refresh_token()
    h = hash_refresh_token(raw)
    check("tokens: refresh hashed (sha256 hex)", len(h) == 64 and h != raw)

    # ── Integration flow (mocked Clerk verification) ─────────────────
    await connect_to_mongo()
    try:
        repo = UsersRepository()
        clerk_user_id = "clerk_test_" + uuid.uuid4().hex[:10]
        fake_email = clerk_user_id + "@example.com"

        def fake_verify(token: str) -> dict:
            return {"sub": clerk_user_id, "email": fake_email, "iss": "https://national-feline-6121.clerk.accounts.dev", "exp": 9999999999}

        original_verify = auth_routes.verify_clerk_token
        auth_routes.verify_clerk_token = fake_verify
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Exchange: create user + profile + tokens
                r = await client.post("/auth/clerk", json={"clerk_token": "fake.token"})
                check("exchange: 200 with access token", r.status_code == 200 and bool(r.json().get("access_token")), str(r.text[:120]))
                cookie = client.cookies.get("sf_refresh")
                check("exchange: sets HttpOnly refresh cookie", bool(cookie), "cookie present")

                db_user = await find_user(repo, clerk_user_id)
                check("exchange: user created in Mongo", db_user is not None and db_user.get("is_banned") is False, str(db_user))
                stored_hash = db_user.get("refresh_token_hash") if db_user else None
                check("exchange: only hash stored, not raw", bool(stored_hash) and stored_hash != cookie and len(stored_hash) == 64, str(stored_hash))
                check("exchange: profile created", await _profile_exists(clerk_user_id))

                # /auth/me
                access = r.json()["access_token"]
                me = await client.get("/auth/me", headers={"Authorization": f"Bearer {access}"})
                check("me: returns user", me.status_code == 200 and me.json().get("email") == fake_email, str(me.text[:120]))
                check("me: missing token -> 401", (await client.get("/auth/me")).status_code == 401)
                check("me: garbage token -> 401", (await client.get("/auth/me", headers={"Authorization": "Bearer garbage"})).status_code == 401)

                # Refresh flow
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c2:
                    r = await c2.post("/auth/refresh")
                    check("refresh: missing cookie -> 401", r.status_code == 401)
                old_cookie = cookie
                r = await client.post("/auth/refresh")
                check("refresh: rotates & returns access", r.status_code == 200 and bool(r.json().get("access_token")), str(r.text[:120]))
                new_cookie = client.cookies.get("sf_refresh")
                check("refresh: cookie rotated", bool(new_cookie) and new_cookie != old_cookie)

                # Reuse detection (explicit Cookie header bypasses the jar)
                r = await client.post("/auth/refresh", headers={"Cookie": f"sf_refresh={old_cookie}"})
                check("refresh: old token reuse -> 401 + revoke", r.status_code == 401 and "reuse" in r.json().get("detail", "").lower(), str(r.text[:120]))
                db_user2 = await find_user(repo, clerk_user_id)
                check("refresh: session revoked after reuse", not db_user2.get("refresh_token_hash"), str(db_user2))

                # Logout
                r = await client.post("/auth/clerk", json={"clerk_token": "fake.token"})
                logout_cookie_set = r.headers.get("set-cookie") or ""
                r = await client.post("/auth/logout")
                cleared = "Max-Age=0" in (r.headers.get("set-cookie") or "") or client.cookies.get("sf_refresh") != logout_cookie_set
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c3:
                    check("logout: revokes refresh", (await c3.post("/auth/refresh")).status_code == 401)
                check("logout: clears refresh cookie", cleared)

                # Banned user
                banned_clerk = "clerk_banned_" + uuid.uuid4().hex[:10]

                def banned_verify(token: str) -> dict:
                    return {"sub": banned_clerk, "email": banned_clerk + "@e.com", "iss": "x", "exp": 9999999999}

                auth_routes.verify_clerk_token = banned_verify
                r = await client.post("/auth/clerk", json={"clerk_token": "fake.token"})
                banned_id = r.json()["user"]["id"]
                await repo.set_banned(banned_id, True)
                r = await client.post("/auth/clerk", json={"clerk_token": "fake.token"})
                check("banned: exchange -> 403", r.status_code == 403, str(r.text[:120]))

                auth_routes.verify_clerk_token = fake_verify
                banned_access = create_access_token(banned_id)
                me = await client.get("/auth/me", headers={"Authorization": f"Bearer {banned_access}"})
                check("banned: access token -> 403", me.status_code == 403, str(me.text[:120]))
        finally:
            auth_routes.verify_clerk_token = original_verify
    finally:
        await close_mongo_connection()

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\nRESULT: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


async def _profile_exists(clerk_id):
    from app.db.mongo_db import profile_collection

    user = await UsersRepository().find_by_clerk_id(clerk_id)
    if not user:
        return False
    return await profile_collection().find_one({"user_id": str(user["_id"])}) is not None


if __name__ == "__main__":
    asyncio.run(main())