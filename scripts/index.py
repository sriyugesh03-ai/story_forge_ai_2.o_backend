import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.indexer import RAGIndexer

indexer = RAGIndexer()

# reset=True wipes the existing collection and re-indexes all PDFs from scratch
indexer.build_index(reset=True)
