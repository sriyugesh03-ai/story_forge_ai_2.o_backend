import logging
import asyncio
from typing import Any
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from app.rag.embedder import EmbeddingService
from app.rag.vectordb import VectorDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, shared by all routes and services.
# ---------------------------------------------------------------------------
_retriever_instance: "Retriever | None" = None


def get_retriever() -> "Retriever":
    """Return the shared Retriever singleton, creating it on first call."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


# ---------------------------------------------------------------------------
# Player-list cache
# ---------------------------------------------------------------------------
_player_cache: list[str] | None = None


def invalidate_player_cache() -> None:
    """Call this after re-indexing PDFs to force a cache refresh."""
    global _player_cache
    _player_cache = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_player_name(name: str) -> str:
    """
    Normalize a player name to match the metadata stored in MongoDB.
    """
    return name.strip().lower().replace("_", " ")


def _find_best_player_match(query_name: str, all_players: list[str]) -> str | None:
    """
    Find the best matching player name from the stored metadata players list.
    """
    normalized_query = _normalize_player_name(query_name)

    # Exact match
    for player in all_players:
        if _normalize_player_name(player) == normalized_query:
            return player

    # Partial match fallback
    query_words = set(normalized_query.split())
    for player in all_players:
        player_words = set(_normalize_player_name(player).split())
        if query_words.issubset(player_words) or player_words.issubset(query_words):
            return player

    return None


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever(BaseRetriever):
    """
    LangChain BaseRetriever implementation for retrieving documents from MongoDB Atlas Vector Store.
    """
    embedder: Any = Field(default_factory=EmbeddingService)
    vectordb: Any = Field(default_factory=VectorDatabase)
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    async def get_all_players(self) -> list[str]:
        """Return all unique player names stored in the collection metadata."""
        global _player_cache
        if _player_cache is not None:
            return _player_cache

        players = await self.vectordb.get_distinct_players()
        _player_cache = sorted(players)
        return _player_cache

    async def get_player_stats(self) -> list[dict]:
        """Return each player with their chunk count."""
        return await self.vectordb.get_player_stats()

    # ------------------------------------------------------------------
    # Retrieve all chunks for a specific player (debug endpoint)
    # ------------------------------------------------------------------

    async def retrieve_all_for_player(self, player_name: str, limit: int = 50) -> dict:
        """Return indexed chunks for the matched player from MongoDB."""
        all_players = await self.get_all_players()
        if not all_players:
            raise ValueError("The vector database is empty. Run the indexer first.")

        matched_player = _find_best_player_match(player_name, all_players)
        if not matched_player:
            raise ValueError(
                f"No player matching '{player_name}' found. "
                f"Available: {all_players}"
            )

        total = await self.vectordb.get_player_chunk_count(matched_player)
        chunks = await self.vectordb.get_all_for_player(matched_player, limit=limit)

        return {
            "player": matched_player,
            "total_chunks": total,
            "returned": len(chunks),
            "chunks": chunks,
        }

    # ------------------------------------------------------------------
    # LangChain BaseRetriever Interface
    # ------------------------------------------------------------------

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Async retrieval method for LangChain LCEL pipelines."""
        all_players = await self.get_all_players()
        matched_player = None
        if all_players:
            matched_player = _find_best_player_match(query, all_players)

        return await self.vectordb.asimilarity_search(
            query=query, k=self.top_k, filter_player=matched_player
        )

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Sync retrieval method for LangChain LCEL pipelines."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self._aget_relevant_documents(query, run_manager=run_manager))
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    # ------------------------------------------------------------------
    # Helper / Compatibility retrieval method
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Semantically retrieve string chunks for backwards compatibility."""
        all_players = await self.get_all_players()
        matched_player = _find_best_player_match(query, all_players) if all_players else None

        docs = await self.vectordb.asimilarity_search(
            query=query, k=top_k, filter_player=matched_player
        )
        return [doc.page_content for doc in docs]