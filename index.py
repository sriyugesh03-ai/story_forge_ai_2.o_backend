from app.rag.indexer import RAGIndexer

indexer = RAGIndexer()

# reset=True wipes the existing collection and re-indexes all PDFs from scratch
indexer.build_index(reset=True)