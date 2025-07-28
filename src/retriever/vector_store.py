from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


def build_vector_store(docs: list[Document], embedder: Embeddings) -> FAISS:
    return FAISS.from_documents(docs, embedder)


def query_vector_store(vstore: FAISS, question: str, k: int = 4) -> str:
    docs = vstore.similarity_search(question, k=k)
    return "\n\n".join([d.page_content for d in docs])
