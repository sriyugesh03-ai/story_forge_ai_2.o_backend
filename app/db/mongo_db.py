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


async def ensure_user_indexes():
    """Ensure unique indexes on username and email in the MongoDB users collection."""
    try:
        collection = user_collection()
        await collection.create_index("username", unique=True)
        await collection.create_index("email", unique=True)
        logger.info("✅ Unique indexes created/verified on 'users' collection.")
    except Exception as e:
        logger.warning(f"Note on user collection index creation: {e}")


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
