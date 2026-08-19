import logging
from datetime import datetime, timezone

from app.db.mongo_db import profile_collection

logger = logging.getLogger(__name__)


class ProfilesRepository:
    """Repository for the MongoDB `profiles` collection (1:1 with users)."""

    def _get_collection(self):
        return profile_collection()

    async def find_by_user_id(self, user_id) -> dict | None:
        return await self._get_collection().find_one({"user_id": str(user_id)})

    async def ensure_profile(self, user_id) -> dict:
        """Return the user's profile, creating an empty one if it does not exist."""
        existing = await self.find_by_user_id(user_id)
        if existing:
            return existing

        profile = {
            "user_id": str(user_id),
            "first_name": "",
            "last_name": "",
            "phone": "",
            "profile_image": "",
            "bio": "",
            "location": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        result = await self._get_collection().insert_one(profile)
        profile["_id"] = result.inserted_id
        logger.info("Created profile for user %s", user_id)
        return profile

    async def update_profile(self, user_id, fields: dict) -> dict | None:
        if not fields:
            return await self.find_by_user_id(user_id)
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._get_collection().update_one(
            {"user_id": str(user_id)},
            {"$set": fields},
        )
        return await self.find_by_user_id(user_id)