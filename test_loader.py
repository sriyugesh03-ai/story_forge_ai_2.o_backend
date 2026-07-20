from app.rag.retriever import Retriever

retriever = Retriever()

results = retriever.retrieve(

    "Tell me about MS Dhoni's captaincy"

)

print("=" * 60)

for index, chunk in enumerate(results, start=1):

    print(f"\nChunk {index}")

    print("-" * 40)

    print(chunk)