from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_vector_store(
    docs: list[Document], embedder: OpenAIEmbeddings
) -> InMemoryVectorStore:
    return InMemoryVectorStore.from_documents(docs, embedder)
