import chromadb


class VectorDatabase:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="data/chroma"
        )

        self.collection = self.client.get_or_create_collection(
            name="sports_knowledge"
        )

    def add_documents(self, chunks, embeddings):

        ids = [

            f"chunk_{i}"

            for i in range(len(chunks))

        ]

        self.collection.add(

            ids=ids,

            documents=chunks,

            embeddings=embeddings.tolist()

        )

    def count(self):

        return self.collection.count()