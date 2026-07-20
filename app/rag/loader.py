import pdfplumber

class DocumentLoader:
    """Wrapper class for loading and extracting text content from PDF files."""

    def load_pdf(self, file_path: str) -> str:
        """Extract text from all pages of a PDF file, returning a single concatenated string."""
        pages_content = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_content.append(page_text)

        # O(n) concatenation instead of quadratic string copy allocations
        return "\n".join(pages_content)