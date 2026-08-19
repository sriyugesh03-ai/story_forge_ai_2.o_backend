from dataclasses import dataclass, field

from app.rag.retriever import get_retriever


@dataclass
class RAGRetrievalResult:
    chunks: list[str] = field(default_factory=list)
    docs: list = field(default_factory=list)
    matched_player: str | None = None

    def is_empty(self) -> bool:
        """True when no useful context could be retrieved from the vector store."""
        return not self.chunks


class RAGNode:
    """RAG retrieval node.

    A thin wrapper around the existing Retriever singleton. Keeps the
    retrieve/build-context steps clearly separated from the router so RAG
    logic is NOT duplicated anywhere else.
    """

    def __init__(self, retriever=None, top_k: int = 5):
        self.retriever = retriever or get_retriever()
        self.top_k = top_k

    async def retrieve(self, query: str, player_name: str | None = None) -> RAGRetrievalResult:
        """Run similarity search and return the top chunks for the query.

        The underlying retriever already scopes the vector search to the
        matched player via the metadata.player filter when it can.
        """
        if player_name:
            # Scoped retrieval: surface docs for the exact stored player.
            docs = await self.retriever.retrieve_docs_for_player(player_name, top_k=self.top_k)
        else:
            docs = await self.retriever.ainvoke(query)

        chunks = [doc.page_content for doc in docs if getattr(doc, "page_content", "").strip()]
        return RAGRetrievalResult(chunks=chunks, docs=docs, matched_player=player_name)

    @staticmethod
    def build_context(chunks: list[str]) -> str:
        """Join retrieved chunks into a single context string for the prompt."""
        return "\n\n".join(chunks)