from pymupdf import open as open_pdf
from pathlib import Path


def extract_text_chunks(file_path: Path) -> list[str]:
    doc = open_pdf(file_path)
    chunks: list[str] = [
        page.get_textpage().extractText()
        for page in doc
        if page.get_textpage().extractText().strip()
    ]
    doc.close()
    return chunks
