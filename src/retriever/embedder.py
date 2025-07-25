from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List


def chunk_texts(
    texts: List[str], chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = [Document(page_content=t) for t in texts]
    return splitter.split_documents(docs)


def get_embedder():
    return OpenAIEmbeddings()
