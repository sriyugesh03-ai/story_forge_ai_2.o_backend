import logging
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from app.core.config import settings
from app.db.mongo_db import user_collection

logger = logging.getLogger(__name__)


class UsersRepository:
    """Repository for the MongoDB `users` collection.

    Only the SHA-256 hash of a refresh token is ever stored — never the raw value.
    """

    def _get_collection(self):
        return user_collection()

    async def find_by_clerk_id(self, clerk_user_id: str) -> dict | None:
        return await self._get_collection().find_one({"clerk_user_id": clerk_user_id})

    async def find_by_id(self, user_id: str) -> dict | None:
        try:
            _id = ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
        except Exception:
            _id = user_id
        return await self._get_collection().find_one({"_id": _id})

    async def find_by_refresh_hash(self, token_hash: str) -> dict | None:
        """Find a user by its active *or* rotated-out refresh-token hash.

        Matching the rotated-out (prev) hash lets the API detect token reuse.
        """
        return await self._get_collection().find_one(
            {"$or": [{"refresh_token_hash": token_hash}, {"refresh_token_prev_hash": token_hash}]}
        )

    async def upsert_clerk_user(self, clerk_user_id: str, email: str, is_verified: bool) -> dict:
        """Create the app user on first Clerk sign-in, or update it afterwards."""
        collection = self._get_collection()
        now = datetime.now(timezone.utc).isoformat()
        user = await self.find_by_clerk_id(clerk_user_id)

        if user is None:
            user = {
                "clerk_user_id": clerk_user_id,
                "email": email or "",
                "is_verified": is_verified,
                "is_banned": False,
                "refresh_token_hash": None,
                "refresh_token_prev_hash": None,
                "refresh_token_expires_at": None,
                "created_at": now,
                "updated_at": now,
                "last_login_at": now,
            }
            result = await collection.insert_one(user)
            user["_id"] = result.inserted_id
            logger.info("Created app user for Clerk user %s", clerk_user_id)
        else:
            update = {
                "email": email or user.get("email", ""),
                "is_verified": is_verified,
                "updated_at": now,
                "last_login_at": now,
            }
            await collection.update_one({"_id": user["_id"]}, {"$set": update})
            user.update(update)
        return user

    async def set_refresh_token(self, user_id, token_hash: str) -> None:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        await self._get_collection().update_one(
            {"_id": user_id},
            {
                "$set": {
                    "refresh_token_hash": token_hash,
                    "refresh_token_prev_hash": None,
                    "refresh_token_expires_at": expires_at.isoformat(),
                    "updated_at": now.isoformat(),
                }
            },
        )

    async def rotate_refresh_token(self, user_id, current_hash: str, new_hash: str) -> None:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        await self._get_collection().update_one(
            {"_id": user_id},
            {
                "$set": {
                    "refresh_token_prev_hash": current_hash,
                    "refresh_token_hash": new_hash,
                    "refresh_token_expires_at": expires_at.isoformat(),
                    "updated_at": now.isoformat(),
                }
            },
        )

    async def revoke_refresh_token(self, user_id) -> None:
        await self._get_collection().update_one(
            {"_id": user_id},
            {
                "$set": {
                    "refresh_token_hash": None,
                    "refresh_token_prev_hash": None,
                    "refresh_token_expires_at": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    async def revoke_by_hash(self, token_hash: str) -> None:
        await self._get_collection().update_one(
            {"refresh_token_hash": token_hash},
            {
                "$set": {
                    "refresh_token_hash": None,
                    "refresh_token_prev_hash": None,
                    "refresh_token_expires_at": None,
                }
            },
        )

    async def set_banned(self, user_id, banned: bool) -> None:
        try:
            _id = ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
        except Exception:
            _id = user_id
        await self._get_collection().update_one(
            {"_id": _id},
            {"$set": {"is_banned": banned, "updated_at": datetime.now(timezone.utc).isoformat()}},
        )