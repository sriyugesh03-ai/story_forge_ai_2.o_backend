import chromadb

class VectorDatabase:
    """Wrapper class for managing ChromaDB persistent client connection and sports_knowledge collection."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="data/chroma"
        )
        self.collection = self.client.get_or_create_collection(
            name="sports_knowledge"
        )