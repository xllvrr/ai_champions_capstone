from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_core.documents import Document
from typing import Any
import re


def semantic_chunk_texts(
    pages_data: list[dict[str, Any]],
    chunk_size: int = 800,  # Increased for better semantic coherence
    chunk_overlap: int = 100,  # Increased for better context preservation
    use_semantic_splitting: bool = True,
) -> list[Document]:
    """
    Enhanced chunking with semantic awareness and proper metadata.
    """
    all_docs = []
    
    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data["page_number"]
        source = page_data["source"]
        
        if use_semantic_splitting:
            # Try semantic splitting first
            semantic_chunks = _semantic_split_text(text)
            
            # If semantic splitting produces reasonable chunks, use them
            if semantic_chunks and len(semantic_chunks) > 1:
                for i, chunk in enumerate(semantic_chunks):
                    if len(chunk.strip()) > 50:  # Skip very small chunks
                        doc = Document(
                            page_content=chunk.strip(),
                            metadata={
                                "source": source,
                                "page": page_num,
                                "chunk_id": f"{source}_p{page_num}_c{i}",
                                "chunk_type": "semantic"
                            }
                        )
                        all_docs.append(doc)
            else:
                # Fall back to recursive splitting
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "page": page_num,
                        "chunk_id": f"{source}_p{page_num}",
                        "chunk_type": "page"
                    }
                )
                all_docs.append(doc)
        else:
            # Direct page-based document creation
            doc = Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_num,
                    "chunk_id": f"{source}_p{page_num}",
                    "chunk_type": "page"
                }
            )
            all_docs.append(doc)
    
    # Apply recursive splitting to maintain manageable chunk sizes
    if chunk_size and chunk_size > 0:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Prefer natural breaks
        )
        final_docs = splitter.split_documents(all_docs)
        
        # Update metadata for split chunks
        for i, doc in enumerate(final_docs):
            if "chunk_id" in doc.metadata:
                doc.metadata["final_chunk_id"] = f"{doc.metadata['chunk_id']}_split{i}"
        
        return final_docs
    
    return all_docs


def _semantic_split_text(text: str) -> list[str]:
    """
    Split text based on semantic boundaries like paragraphs and sections.
    """
    # Split on paragraph breaks first
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Check if this looks like a section header
        is_header = (
            len(para) < 100 and 
            (para.isupper() or 
             re.match(r'^[A-Z][^.]*[^.]$', para) or
             re.match(r'^\d+\.?\s+[A-Z]', para))
        )
        
        # If adding this paragraph would make chunk too long, start new chunk
        if current_chunk and len(current_chunk + para) > 1200:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                
        # If this is a header and we have content, consider ending chunk
        if is_header and current_chunk.strip() and len(current_chunk) > 200:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if len(chunks) > 1 else []


def chunk_texts(
    texts: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str | None = None,
) -> list[Document]:
    """Legacy function for backward compatibility."""
    # Convert to new format
    pages_data = [
        {"text": text, "page_number": i + 1, "source": source}
        for i, text in enumerate(texts)
    ]
    return semantic_chunk_texts(pages_data, chunk_size, chunk_overlap, use_semantic_splitting=False)
