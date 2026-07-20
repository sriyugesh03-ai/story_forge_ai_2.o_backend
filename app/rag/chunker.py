class TextChunker:

    def __init__(

        self,

        chunk_size=1000,

        chunk_overlap=200

    ):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str):

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks