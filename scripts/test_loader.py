import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import get_retriever

retriever = get_retriever()

results = retriever.retrieve(
    "Tell me about MS Dhoni's captaincy"
)

print("=" * 60)

for index, chunk in enumerate(results, start=1):
    print(f"\nChunk {index}")
    print("-" * 40)
    print(chunk)
