import logging
import asyncio
from pymongo import UpdateOne
from langchain_core.vectorstores import VectorStore as LCVectorStore
from langchain_core.documents import Document
from app.db.mongo_db import get_vector_collection
from app.rag.embedder import EmbeddingService

logger = logging.getLogger("uvicorn")


class VectorStore(LCVectorStore):
    """
    Stores text chunks alongside their embeddings in MongoDB Atlas,
    and performs vector search using Atlas Vector Search ($vectorSearch).
    Subclasses LangChain VectorStore for full LCEL compatibility.
    """

    INDEX_NAME = "vector_index_v4"

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or EmbeddingService()

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        raise NotImplementedError("Use add or aadd_documents method.")

    def _get_collection(self):
        return get_vector_collection()

    async def aadd_documents(self, documents: list[Document], ids: list[str] = None) -> list[str]:
        """LangChain async document ingestion."""
        if not documents:
            return []
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding_model.aembed_documents(texts)
        if ids is None:
            ids = [
                doc.metadata.get("id") or f"doc_{i}"
                for i, doc in enumerate(documents)
            ]
        metadatas = [doc.metadata for doc in documents]
        await self.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return ids

    def add_documents(self, documents: list[Document], ids: list[str] = None) -> list[str]:
        """LangChain sync document ingestion."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.aadd_documents(documents, ids))
        except RuntimeError:
            return asyncio.run(self.aadd_documents(documents, ids))

    async def asimilarity_search(
        self, query: str, k: int = 4, filter_player: str | None = None, **kwargs
    ) -> list[Document]:
        """LangChain async similarity search returning Document objects."""
        query_embedding = await self.embedding_model.aembed_query(query)
        results = await self.query(query_embedding=query_embedding, top_k=k, filter_player=filter_player)
        return [
            Document(
                page_content=res["text"],
                metadata={"source": res["source"], "score": res["score"], "distance": res["distance"]}
            )
            for res in results
        ]

    def similarity_search(
        self, query: str, k: int = 4, filter_player: str | None = None, **kwargs
    ) -> list[Document]:
        """LangChain sync similarity search returning Document objects."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.asimilarity_search(query, k=k, filter_player=filter_player))
        except RuntimeError:
            return asyncio.run(self.asimilarity_search(query, k=k, filter_player=filter_player))

    async def get_existing_ids(self) -> set[str]:
        """Returns all document chunk IDs already present in MongoDB Atlas."""

        try:
            collection = self._get_collection()
            cursor = collection.find({}, {"_id": 1, "id": 1})
            docs = await cursor.to_list(length=50000)
            existing = set()
            for doc in docs:
                if "_id" in doc:
                    existing.add(str(doc["_id"]))
                if "id" in doc:
                    existing.add(str(doc["id"]))
            return existing
        except Exception as e:
            logger.error(f"Failed to fetch existing document IDs from MongoDB: {e}")
            return set()

    async def ensure_vector_index(self):
        """Automatically checks and creates the Atlas Vector Search index if missing."""
        try:
            collection = self._get_collection()
            cursor = collection.list_search_indexes()
            existing_indexes = await cursor.to_list(length=100)
            idx_names = [idx.get("name") for idx in existing_indexes]
            if self.INDEX_NAME not in idx_names:
                from pymongo.operations import SearchIndexModel
                index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 768,
                                "similarity": "cosine"
                            },
                            {
                                "type": "filter",
                                "path": "metadata.player"
                            }
                        ]
                    },
                    name=self.INDEX_NAME,
                    type="vectorSearch"
                )
                await collection.create_search_index(model=index_model)
                logger.info(f"✅ Automatically created Atlas Vector Search index '{self.INDEX_NAME}' on MongoDB Atlas.")
        except Exception as e:
            logger.debug(f"Atlas Search index check note: {e}")

    async def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        """Stores or updates a batch of document chunks in MongoDB Atlas."""
        if not ids:
            return

        collection = self._get_collection()
        operations = []

        for chunk_id, embedding, doc, meta in zip(ids, embeddings, documents, metadatas):
            doc_body = {
                "_id": chunk_id,
                "id": chunk_id,
                "text": doc,
                "embedding": embedding,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
            }
            operations.append(
                UpdateOne({"_id": chunk_id}, {"$set": doc_body}, upsert=True)
            )

        if operations:
            result = await collection.bulk_write(operations)
            logger.info(f"MongoDB Vector Store updated: {result.upserted_count} inserted, {result.modified_count} modified.")
            await self.ensure_vector_index()

    async def query(self, query_embedding: list[float], top_k: int = 4, filter_player: str | None = None) -> list[dict]:
        """Finds the `top_k` chunks most similar to the query embedding using MongoDB Atlas Vector Search."""
        collection = self._get_collection()

        vector_search_stage = {
            "index": self.INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": max(top_k * 10, 50),
            "limit": top_k,
        }

        if filter_player:
            vector_search_stage["filter"] = {"metadata.player": filter_player}

        pipeline = [
            {
                "$vectorSearch": vector_search_stage
            },
            {
                "$project": {
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            matches = []
            for doc in results:
                matches.append({
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "unknown"),
                    "score": doc.get("score", 0.0),
                    "distance": 1.0 - doc.get("score", 0.0),
                })
            return matches
        except Exception as e:
            logger.warning(
                f"MongoDB Vector Search query failed or 'vector_index' is not configured yet in Atlas UI: {e}"
            )
            return []

    async def get_by_id(self, chunk_id: str) -> dict | None:
        """Fetch a document by its ID."""
        try:
            collection = self._get_collection()
            return await collection.find_one({"_id": chunk_id})
        except Exception as e:
            logger.error(f"Failed to fetch document by ID {chunk_id}: {e}")
            return None

    async def count(self) -> int:
        """Count the number of documents in the collection."""
        try:
            collection = self._get_collection()
            return await collection.count_documents({})
        except Exception as e:
            logger.error(f"Failed to count documents in MongoDB: {e}")
            return 0

    async def get_distinct_players(self) -> list[str]:
        """Returns all unique player names from metadata."""
        try:
            collection = self._get_collection()
            return await collection.distinct("metadata.player")
        except Exception as e:
            logger.error(f"Failed to fetch distinct players: {e}")
            return []

    async def get_ingested_sources(self) -> set[str]:
        """Returns the set of already-ingested PDF stems (e.g. 'Carlos_Alcaraz').
        
        Much lighter than get_existing_ids() — fetches only one distinct name
        per document instead of one ID per chunk. Used by ingest() to skip
        already-processed files on server startup.
        """
        players = await self.get_distinct_players()
        return set(players)

    async def get_player_stats(self) -> list[dict]:
        """Returns the chunk count for each player."""
        try:
            collection = self._get_collection()
            pipeline = [
                {"$group": {"_id": "$metadata.player", "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=1000)
            return [
                {"player": doc["_id"] or "unknown", "chunks_indexed": doc["count"]}
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Failed to fetch player stats: {e}")
            return []

    async def get_player_chunk_count(self, player_name: str) -> int:
        """Count the number of chunks for a player."""
        try:
            collection = self._get_collection()
            return await collection.count_documents({"metadata.player": player_name})
        except Exception as e:
            logger.error(f"Failed to count chunks for player {player_name}: {e}")
            return 0

    async def get_all_for_player(self, player_name: str, limit: int = 50) -> list[str]:
        """Fetch indexed chunks for a given player."""
        try:
            collection = self._get_collection()
            cursor = collection.find({"metadata.player": player_name}, {"text": 1}).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [doc.get("text", "") for doc in docs]
        except Exception as e:
            logger.error(f"Failed to fetch chunks for player {player_name}: {e}")
            return []


# Keep class name compatibility with rest of codebase
VectorDatabase = VectorStore