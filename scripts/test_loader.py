import sys
import os
import asyncio

# Configure stdout to handle UTF-8 encoding safely on Windows command prompt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import get_retriever
from app.db.mongo_db import connect_to_mongo, close_mongo_connection

async def main():
    # Setup MongoDB connection
    await connect_to_mongo()
    try:
        retriever = get_retriever()
        results = await retriever.retrieve("Tell me about Carlos Alcaraz")
        
        print("=" * 60)
        for index, chunk in enumerate(results, start=1):
            print(f"\nChunk {index}")
            print("-" * 40)
            try:
                print(chunk)
            except UnicodeEncodeError:
                print(chunk.encode("ascii", errors="replace").decode("ascii"))
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(main())
