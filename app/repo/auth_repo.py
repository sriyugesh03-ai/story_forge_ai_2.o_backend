import logging
from datetime import datetime, timezone
from bson import ObjectId
from app.db.mongo_db import user_collection

logger = logging.getLogger(__name__)


class AuthRepository:
    """Repository layer handling database CRUD operations on the MongoDB 'users' collection."""

    def __init__(self):
        pass

    def _get_collection(self):
        return user_collection()

    async def find_by_username(self, username: str) -> dict | None:
        """Find user document by username."""
        try:
            collection = self._get_collection()
            return await collection.find_one({"username": username})
        except Exception as e:
            logger.error(f"Error fetching user by username '{username}': {e}")
            return None

    async def find_by_email(self, email: str) -> dict | None:
        """Find user document by email."""
        try:
            collection = self._get_collection()
            return await collection.find_one({"email": email})
        except Exception as e:
            logger.error(f"Error fetching user by email '{email}': {e}")
            return None

    async def find_by_username_or_email(self, identifier: str) -> dict | None:
        """Find user document by username OR email."""
        try:
            collection = self._get_collection()
            return await collection.find_one({
                "$or": [
                    {"username": identifier},
                    {"email": identifier}
                ]
            })
        except Exception as e:
            logger.error(f"Error fetching user by identifier '{identifier}': {e}")
            return None

    async def find_by_id(self, user_id: str) -> dict | None:
        """Find user document by ObjectId string."""
        try:
            collection = self._get_collection()
            obj_id = ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
            return await collection.find_one({"_id": obj_id})
        except Exception as e:
            logger.error(f"Error fetching user by ID '{user_id}': {e}")
            return None

    async def create_user(self, user_data: dict) -> dict:
        """Insert a new user document into MongoDB."""
        collection = self._get_collection()
        if "created_at" not in user_data:
            user_data["created_at"] = datetime.now(timezone.utc).isoformat()

        result = await collection.insert_one(user_data)
        user_data["_id"] = str(result.inserted_id)
        return user_data
