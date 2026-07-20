import sys
import os

# Configure stdout to handle UTF-8 encoding safely on Windows command prompt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    # Safely print character strings that might have non-cp1252 characters
    try:
        print(chunk)
    except UnicodeEncodeError:
        print(chunk.encode("ascii", errors="replace").decode("ascii"))
