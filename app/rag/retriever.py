import logging
from app.rag.embedder import EmbeddingService
from app.rag.vectordb import VectorDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, shared by all routes and services.
# This prevents multiple SentenceTransformer model loads, each of which
# costs ~200 MB of RAM.  Import this instance instead of calling Retriever().
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
# Invalidated by calling invalidate_player_cache() after re-indexing.
# Avoids fetching all ChromaDB metadata on every story request.
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
    Normalize a player name to match the metadata stored in ChromaDB.
    The metadata 'player' field is derived from the PDF filename stem
    (e.g. 'MS_Dhoni', 'Virat Kohli', 'Lionel_Messi').
    """
    return name.strip().lower().replace("_", " ")


def _find_best_player_match(query_name: str, all_players: list[str]) -> str | None:
    """
    Find the best matching player name from the stored metadata players list.
    Returns the original (un-normalized) stored player name on match, else None.
    """
    normalized_query = _normalize_player_name(query_name)

    # Exact match (after normalization)
    for player in all_players:
        if _normalize_player_name(player) == normalized_query:
            return player

    # Partial match fallback: check if query words are all in stored name
    query_words = set(normalized_query.split())
    for player in all_players:
        player_words = set(_normalize_player_name(player).split())
        if query_words.issubset(player_words) or player_words.issubset(query_words):
            return player

    return None


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:

    def __init__(self):
        self.embedder = EmbeddingService()
        self.vectordb = VectorDatabase()

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def get_all_players(self) -> list[str]:
        """Return all unique player names stored in the collection metadata.
        Result is cached in memory; call invalidate_player_cache() after re-indexing.
        """
        global _player_cache
        if _player_cache is not None:
            return _player_cache

        count = self.vectordb.collection.count()
        if count == 0:
            return []

        result = self.vectordb.collection.get(include=["metadatas"])
        players = sorted({m["player"] for m in result["metadatas"] if "player" in m})
        _player_cache = players
        return players

    def get_player_stats(self) -> list[dict]:
        """Return each player with their chunk count."""
        count = self.vectordb.collection.count()
        if count == 0:
            return []

        result = self.vectordb.collection.get(include=["metadatas"])
        stats: dict[str, int] = {}
        for m in result["metadatas"]:
            player = m.get("player", "unknown")
            stats[player] = stats.get(player, 0) + 1

        return [
            {"player": p, "chunks_indexed": c}
            for p, c in sorted(stats.items())
        ]

    # ------------------------------------------------------------------
    # Retrieve all chunks for a specific player (debug endpoint)
    # ------------------------------------------------------------------

    def retrieve_all_for_player(self, player_name: str, limit: int = 50) -> dict:
        """
        Return indexed chunks for the matched player.
        Uses `limit` to prevent dumping the entire collection in one response.
        Raises ValueError if no player is matched or the DB is empty.
        """
        all_players = self.get_all_players()
        if not all_players:
            raise ValueError("The vector database is empty. Run the indexer first.")

        matched_player = _find_best_player_match(player_name, all_players)
        if not matched_player:
            raise ValueError(
                f"No player matching '{player_name}' found. "
                f"Available: {all_players}"
            )

        result = self.vectordb.collection.get(
            where={"player": {"$eq": matched_player}},
            include=["documents", "metadatas"],
        )

        docs = result["documents"]
        total = len(docs)
        # Apply limit to avoid oversized responses
        docs = docs[:limit]

        return {
            "player": matched_player,
            "total_chunks": total,
            "returned": len(docs),
            "chunks": docs,
        }

    # ------------------------------------------------------------------
    # Semantic retrieval (used by story generation)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Semantically retrieve the top-k most relevant chunks.
        Automatically filters by player when a match is found.
        Uses the cached player list to avoid a ChromaDB metadata scan per request.
        """
        query_embedding = self.embedder.create_embeddings([query])

        all_players = self.get_all_players()  # reads from cache after first call

        where_filter = None
        if all_players:
            matched_player = _find_best_player_match(query, all_players)
            if matched_player:
                where_filter = {"player": {"$eq": matched_player}}
                logger.debug("Filtering by player: '%s'", matched_player)
            else:
                logger.debug("No exact player match for '%s', doing global search", query)

        query_kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": top_k,
        }

        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.vectordb.collection.query(**query_kwargs)
        return results["documents"][0]