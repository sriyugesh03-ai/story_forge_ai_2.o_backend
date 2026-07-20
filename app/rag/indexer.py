from pathlib import Path
from app.rag.loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingService
from app.rag.vectordb import VectorDatabase


class RAGIndexer:

    def __init__(self):

        self.loader = DocumentLoader()

        self.chunker = TextChunker()

        self.embedder = EmbeddingService()

        self.vectordb = VectorDatabase()

    def reset_index(self):
        """Wipe the entire ChromaDB collection and recreate it fresh."""
        client = self.vectordb.client
        client.delete_collection("sports_knowledge")
        self.vectordb.collection = client.get_or_create_collection(
            name="sports_knowledge"
        )
        print("[Indexer] Collection wiped. Starting fresh build.\n")

    def build_index(self, reset: bool = False):

        if reset:
            self.reset_index()

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
                existing = self.vectordb.collection.get(ids=[first_id])
                if existing["ids"]:
                    print(f"  [SKIP] Already indexed: {pdf.name}")
                    continue

            text = self.loader.load_pdf(str(pdf))

            chunks = self.chunker.split_text(text)

            embeddings = self.embedder.create_embeddings(chunks)

            ids = [f"{pdf.stem}_{i}" for i in range(len(chunks))]

            self.vectordb.collection.add(

                ids=ids,

                documents=chunks,

                embeddings=embeddings.tolist(),

                metadatas=[
                    {"player": pdf.stem, "source": "Wikipedia"}
                    for _ in chunks
                ]

            )

            print(f"  [DONE] {len(chunks)} chunks indexed for '{pdf.stem}'")

        final_count = self.vectordb.collection.count()
        print("\n" + "=" * 60)
        print(f"Index build complete. Total chunks in DB: {final_count}")
        print("=" * 60 + "\n")