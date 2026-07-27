import sys
import os
import asyncio

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.indexer import RAGIndexer
from app.db.mongo_db import connect_to_mongo, close_mongo_connection

async def main():
    # Setup MongoDB connection
    await connect_to_mongo()
    try:
        indexer = RAGIndexer()
        # reset=True wipes the existing collection and re-indexes all PDFs from scratch
        await indexer.build_index(reset=True)
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(main())
