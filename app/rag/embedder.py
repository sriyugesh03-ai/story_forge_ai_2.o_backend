import litellm
import asyncio
import logging
from langchain_core.embeddings import Embeddings
from app.core.config import settings

logger = logging.getLogger("uvicorn")


class EmbeddingModel(Embeddings):
    """
    Turns text into vectors so we can compare how similar two pieces of text are.

    Uses Google's Gemini embedding API via LiteLLM with rate limit retry logic.
    Subclasses LangChain's Embeddings base class for full ecosystem compatibility.
    """

    def __init__(self, model_name: str = "gemini/gemini-embedding-001", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions

    async def _embed_with_retry(self, **kwargs) -> dict:
        """Call litellm.aembedding with retry logic for 429 RateLimitErrors."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return await litellm.aembedding(**kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "rate_limit" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg
                if is_rate_limit and attempt < max_retries - 1:
                    import re
                    match = re.search(r"retry in ([\d\.]+)s", err_msg)
                    if match:
                        delay = float(match.group(1)) + 1.0
                    else:
                        delay = 40.0
                    logger.warning(
                        f"[EmbeddingModel] Rate limit hit (429/ResourceExhausted). "
                        f"Sleeping {delay:.1f}s to reset quota... (Attempt {attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[EmbeddingModel] Embedding call failed permanently: {e}")
                    raise e

    async def aembed_documents(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Embeds many chunks asynchronously — used during ingestion."""
        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = await self._embed_with_retry(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
                api_key=settings.GEMINI_API_KEY,
            )
            embeddings.extend(item["embedding"] for item in response.data)

            # Brief pause to stay under rate limits
            if start + batch_size < len(texts):
                await asyncio.sleep(0.5)

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper for embedding documents."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.aembed_documents(texts))
        except RuntimeError:
            return asyncio.run(self.aembed_documents(texts))

    async def aembed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text asynchronously — used for a user's question."""
        response = await self._embed_with_retry(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
            task_type="RETRIEVAL_QUERY",
            api_key=settings.GEMINI_API_KEY,
        )
        return response.data[0]["embedding"]

    def embed_query(self, text: str) -> list[float]:
        """Synchronous wrapper for embedding query."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.aembed_query(text))
        except RuntimeError:
            return asyncio.run(self.aembed_query(text))

    async def embed_texts(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Compatibility alias for aembed_documents."""
        return await self.aembed_documents(texts, batch_size=batch_size)


# Keep class name compatibility with rest of codebase
EmbeddingService = EmbeddingModel

