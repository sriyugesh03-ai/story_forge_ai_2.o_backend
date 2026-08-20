import logging
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from app.auth.github import STATE_TTL_SECONDS
from app.db.mongo_db import github_connections_collection

logger = logging.getLogger(__name__)


class GithubRepository:
    """MongoDB persistence for GitHub connections + OAuth pending states.

    One document per user. The only GitHub-sensitive value stored is the
    Fernet-encrypted access token; the raw token is never persisted.
    """

    def _get_collection(self):
        return github_connections_collection()

    def _oid(self, user_id):
        try:
            return ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
        except Exception:
            return user_id

    async def find_by_user_id(self, user_id) -> dict | None:
        return await self._get_collection().find_one({"user_id": str(user_id)})

    async def find_by_github_user_id(self, github_user_id: int) -> dict | None:
        return await self._get_collection().find_one({"github_user_id": github_user_id})

    async def set_pending_state(self, user_id, state: str) -> None:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=STATE_TTL_SECONDS)).isoformat()
        await self._get_collection().update_one(
            {"user_id": str(user_id)},
            {
                "$set": {
                    "pending_state": state,
                    "pending_state_expires_at": expires_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )

    async def clear_pending_state(self, user_id) -> None:
        await self._get_collection().update_one(
            {"user_id": str(user_id)},
            {
                "$set": {
                    "pending_state": None,
                    "pending_state_expires_at": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    async def save_connection(self, user_id, github_user_id: int, github_username: str,
                              encrypted_token: str, scopes: list[str]) -> dict:
        """Store (or replace) a linked GitHub account."""
        now = datetime.now(timezone.utc).isoformat()
        await self._get_collection().update_one(
            {"user_id": str(user_id)},
            {
                "$set": {
                    "user_id": str(user_id),
                    "github_user_id": github_user_id,
                    "github_username": github_username,
                    "access_token_enc": encrypted_token,
                    "scopes": scopes,
                    "connected_at": now,
                    "pending_state": None,
                    "pending_state_expires_at": None,
                    "updated_at": now,
                }
            },
            upsert=True,
        )
        return await self.find_by_user_id(user_id)

    async def disconnect(self, user_id) -> None:
        """Remove the GitHub connection document entirely."""
        await self._get_collection().delete_one({"user_id": str(user_id)})

    async def get_connected_token(self, user_id) -> dict | None:
        """Return the connection with its encrypted token, or None."""
        doc = await self.find_by_user_id(user_id)
        if not doc or not doc.get("access_token_enc"):
            return None
        return doc