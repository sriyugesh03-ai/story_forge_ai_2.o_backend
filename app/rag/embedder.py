from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """Wrapper class for generating text embeddings using SentenceTransformer model."""

    def __init__(self):
        # The model is loaded once here by the retriever singleton
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def create_embeddings(self, chunks):
        """Generate vector embeddings for text chunks without printing a progress bar to console."""
        return self.model.encode(
            chunks,
            show_progress_bar=False
        )