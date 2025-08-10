from pymupdf import open as open_pdf
from pathlib import Path
from typing import Any
import re


def extract_text_with_metadata(file_path: Path) -> list[dict[str, Any]]:
    """Extract text from PDF with proper page metadata and basic preprocessing."""
    doc = open_pdf(file_path)
    pages_data = []
    
    for page_num, page in enumerate(doc):
        text = page.get_textpage().extractText()
        if text.strip():
            # Basic text preprocessing
            cleaned_text = preprocess_text(text)
            if cleaned_text.strip():  # Only add if text remains after cleaning
                pages_data.append({
                    "text": cleaned_text,
                    "page_number": page_num + 1,  # 1-indexed page numbers
                    "source": file_path.name
                })
    
    doc.close()
    return pages_data


def preprocess_text(text: str) -> str:
    """Basic text preprocessing to clean extracted PDF text."""
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Remove page headers/footers patterns (basic heuristic)
    lines = text.split('\n')
    if len(lines) > 3:
        # Remove lines that are very short and at beginning/end (likely headers/footers)
        if len(lines[0].strip()) < 50 and lines[0].strip().isdigit():
            lines = lines[1:]
        if len(lines) > 0 and len(lines[-1].strip()) < 50 and lines[-1].strip().isdigit():
            lines = lines[:-1]
    
    return '\n'.join(lines).strip()


def extract_text_chunks(file_path: Path) -> list[str]:
    """Legacy function for backward compatibility."""
    pages_data = extract_text_with_metadata(file_path)
    return [page["text"] for page in pages_data]
