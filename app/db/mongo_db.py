import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

# Setup Logger (Critical for Cloud Debugging)
logger = logging.getLogger("uvicorn")

class Database:
    client: AsyncIOMotorClient = None

db_instance = Database()

def get_database_client():
    return db_instance.client


def get_vector_collection():
    db_name = getattr(settings, "DB_NAME", None) or "rag_db"
    return db_instance.client[db_name]["vector_documents"]


def user_collection():
    db_name = getattr(settings, "DB_NAME", None) or "rag_db"
    return db_instance.client[db_name]["users"]


def profile_collection():
    db_name = getattr(settings, "DB_NAME", None) or "rag_db"
    return db_instance.client[db_name]["profiles"]


def github_connections_collection():
    db_name = getattr(settings, "DB_NAME", None) or "rag_db"
    return db_instance.client[db_name]["github_connections"]


async def ensure_user_indexes():
    """Ensure the MongoDB indexes required for authentication."""
    try:
        collection = user_collection()
        # Drop the legacy non-sparse unique indexes if present: a non-sparse
        # unique index on `username` would let only ONE Clerk-backed document
        # exist (they have no username field -> null), which is wrong.
        for legacy in ("username_1", "email_1"):
            try:
                await collection.drop_index(legacy)
                logger.info(f"⚠ Dropped legacy non-sparse unique index '{legacy}' (replacing with sparse).")
            except Exception:
                pass
        # Explicit names make this idempotent across restarts.
        await collection.create_index("username", unique=True, sparse=True, name="username_unique_sparse")
        await collection.create_index("email", unique=True, sparse=True, name="email_unique_sparse")
        await collection.create_index("clerk_user_id", unique=True, sparse=True, name="clerk_user_id_unique")
        await collection.create_index("refresh_token_hash", sparse=True, name="refresh_token_hash_sparse")
        logger.info("✅ Unique indexes created/verified on 'users' collection.")
    except Exception as e:
        logger.warning(f"Note on user collection index creation: {e}")

    try:
        profiles = profile_collection()
        try:
            await profiles.drop_index("user_id_1")
            logger.info("⚠ Dropped legacy auto-named index 'user_id_1' (replacing with explicit name).")
        except Exception:
            pass
        await profiles.create_index("user_id", unique=True, name="user_id_unique")
        logger.info("✅ Unique index on 'profiles.user_id' verified.")
    except Exception as e:
        logger.warning(f"Note on profiles index creation: {e}")

    try:
        gh = github_connections_collection()
        await gh.create_index("user_id", unique=True, name="user_id_unique")
        await gh.create_index("github_user_id", unique=True, sparse=True, name="github_user_id_unique")
        logger.info("✅ Indexes verified on 'github_connections' collection.")
    except Exception as e:
        logger.warning(f"Note on github_connections index creation: {e}")


async def connect_to_mongo():
    try:
        logger.info("⏳ Connecting to MongoDB...")
        db_instance.client = AsyncIOMotorClient(
            settings.MANGO_DB_URL,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True
        )
        
        # THE PING TEST (Crucial for Cloud)
        await db_instance.client.admin.command('ping')
        logger.info("✅ MongoDB Connected Successfully!")
        await ensure_user_indexes()
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise e


async def close_mongo_connection():
    if db_instance.client:
        db_instance.client.close()
        logger.info("🔒 MongoDB connection closed.")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        asyncio.run(connect_to_mongo())
    finally:
        asyncio.run(close_mongo_connection())
