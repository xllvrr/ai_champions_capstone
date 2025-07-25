from unstructured.partition.pdf import partition_pdf
from typing import List
from pathlib import Path


def extract_text_chunks(file_path: Path) -> List[str]:
    elements = partition_pdf(filename=str(file_path))
    return [el.text for el in elements if el.text and el.text.strip()]
