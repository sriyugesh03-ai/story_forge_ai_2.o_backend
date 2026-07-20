from app.rag.embedder import EmbeddingService
from app.rag.vectordb import VectorDatabase


def _normalize_player_name(name: str) -> str:
    """
    Normalize a player name to match the metadata stored in ChromaDB.
    The metadata 'player' field is derived from the PDF filename stem
    (e.g. 'MS_Dhoni', 'Virat Kohli', 'Lionel_Messi').
    This function converts the query topic into a comparable lowercase string.
    """
    return name.strip().lower().replace("_", " ")


def _find_best_player_match(query_name: str, all_players: list[str]) -> str | None:
    """
    Find the best matching player name from the stored metadata players list.
    Compares normalized versions of both the query and stored player names.
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


class Retriever:

    def __init__(self):

        self.embedder = EmbeddingService()

        self.vectordb = VectorDatabase()

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def get_all_players(self) -> list[str]:
        """Return all unique player names stored in the collection metadata."""
        count = self.vectordb.collection.count()
        if count == 0:
            return []

        result = self.vectordb.collection.get(include=["metadatas"])
        players = sorted({m["player"] for m in result["metadatas"] if "player" in m})
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
    # Retrieve all chunks for a specific player
    # ------------------------------------------------------------------

    def retrieve_all_for_player(self, player_name: str) -> dict:
        """
        Return every indexed chunk for the matched player.
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

        return {
            "player": matched_player,
            "total_chunks": len(result["documents"]),
            "chunks": result["documents"],
        }

    # ------------------------------------------------------------------
    # Semantic retrieval (used by story generation)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Semantically retrieve the top-k most relevant chunks.
        Automatically filters by player when a match is found.
        """
        # Embed the query
        query_embedding = self.embedder.create_embeddings([query])

        # Try to find a matching player for filtered retrieval
        all_players = self.get_all_players()

        where_filter = None
        if all_players:
            matched_player = _find_best_player_match(query, all_players)
            if matched_player:
                where_filter = {"player": {"$eq": matched_player}}
                print(f"[Retriever] Filtering by player: '{matched_player}'")
            else:
                print(f"[Retriever] No exact player match for '{query}', doing global search")

        # Build query kwargs
        query_kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": top_k,
        }

        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.vectordb.collection.query(**query_kwargs)

        return results["documents"][0]