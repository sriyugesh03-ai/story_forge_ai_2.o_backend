from pathlib import Path
from app.rag.loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingService
from app.rag.vectordb import VectorDatabase
from app.rag.retriever import invalidate_player_cache

class RAGIndexer:
    """Class to manage indexing PDF files, parsing text, generating embeddings, and storing them in MongoDB."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = EmbeddingService()
        self.vectordb = VectorDatabase()

    async def reset_index(self):
        """Wipe the entire MongoDB collection."""
        collection = self.vectordb._get_collection()
        await collection.delete_many({})
        print("[Indexer] Collection wiped. Starting fresh build.\n")

    async def build_index(self, reset: bool = False):
        """Processes all PDFs in data/pdfs, chunks their contents, creates vector embeddings, and indexes them in MongoDB."""
        if reset:
            await self.reset_index()

        pdf_folder = Path("data/pdfs")
        pdf_files = sorted(pdf_folder.glob("*.pdf"))
        total = len(pdf_files)

        print(f"\nFound {total} PDFs\n")

        for idx, pdf in enumerate(pdf_files, start=1):
            print("=" * 60)
            print(f"[{idx}/{total}] Processing: {pdf.name}")
            print("=" * 60)

            # Skip if already indexed (only when not doing a full reset)
            if not reset:
                first_id = f"{pdf.stem}_0"
                existing = await self.vectordb.get_by_id(first_id)
                if existing:
                    print(f"  [SKIP] Already indexed: {pdf.name}")
                    continue

            docs = self.loader.load_documents(str(pdf))
            split_docs = self.chunker.split_documents(docs)
            ids = [f"{pdf.stem}_{i}" for i in range(len(split_docs))]
            for i, doc in enumerate(split_docs):
                doc.metadata["player"] = pdf.stem
                doc.metadata["source"] = "Wikipedia"
                doc.metadata["id"] = ids[i]

            await self.vectordb.aadd_documents(split_docs, ids=ids)

            print(f"  [DONE] {len(split_docs)} chunks indexed for '{pdf.stem}'")


        # Invalidate player search cache in case indexer runs in-process
        invalidate_player_cache()

        final_count = await self.vectordb.count()
        print("\n" + "=" * 60)
        print(f"Index build complete. Total chunks in DB: {final_count}")
        print("=" * 60 + "\n")