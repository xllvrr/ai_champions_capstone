from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from typing import List


def build_vector_store(docs: List[Document], embedder: Embeddings) -> FAISS:
    return FAISS.from_documents(docs, embedder)


def query_vector_store(vstore: FAISS, question: str, k: int = 4) -> str:
    docs = vstore.similarity_search(question, k=k)
    return "\n\n".join([d.page_content for d in docs])
