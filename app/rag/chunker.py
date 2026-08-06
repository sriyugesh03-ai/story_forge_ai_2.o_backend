from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    """LangChain-native Text Chunker using RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_text(self, text: str) -> list[str]:
        """Splits a single text string into chunks based on character/paragraph boundaries."""
        return self.splitter.split_text(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits a list of LangChain Document objects into smaller chunk Document objects."""
        return self.splitter.split_documents(documents)