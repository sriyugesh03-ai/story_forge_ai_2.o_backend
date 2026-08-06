from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document


class DocumentLoader:
    """LangChain-native Document Loader for PDF files using PDFPlumberLoader."""

    def load_documents(self, file_path: str) -> list[Document]:
        """Load pages from a PDF file as a list of LangChain Document objects."""
        loader = PDFPlumberLoader(file_path)
        return loader.load()

    def load_pdf(self, file_path: str) -> str:
        """Extract text from all pages of a PDF file, returning a single concatenated string."""
        docs = self.load_documents(file_path)
        return "\n".join(doc.page_content for doc in docs if doc.page_content)