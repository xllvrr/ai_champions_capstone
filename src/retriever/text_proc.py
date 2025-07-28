from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_texts(
    texts: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = [
        Document(page_content=t, metadata={"source": source, "page": i})
        for i, t in enumerate(texts)
    ]
    return splitter.split_documents(docs)
