from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from typing import Any
import asyncio


class AdvancedRetriever:
    """Enhanced retrieval system with multi-query and contextual compression."""
    
    def __init__(self, base_retriever: VectorStoreRetriever, llm_model: str = "gpt-4o-mini"):
        self.base_retriever = base_retriever
        self.llm = init_chat_model(llm_model, temperature=0)
        
        # Multi-query retriever for query expansion
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=self.llm,
            prompt=self._get_multi_query_prompt()
        )
        
        # Contextual compression for relevance filtering
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def _get_multi_query_prompt(self) -> ChatPromptTemplate:
        """Custom prompt for multi-query generation focused on academic content."""
        return ChatPromptTemplate.from_template(
            """You are an academic research assistant. Your task is to generate multiple 
            search queries that would help find comprehensive information about the user's question.
            
            Generate 3-4 alternative queries that:
            1. Use different academic terminology or synonyms
            2. Approach the topic from different angles or perspectives
            3. Include related concepts that might contain relevant information
            4. Are specific enough to retrieve relevant academic content
            
            Original question: {question}
            
            Alternative queries (one per line):"""
        )
    
    def retrieve_with_multi_query(self, query: str, k: int = 8) -> list[Document]:
        """Retrieve documents using multi-query expansion."""
        try:
            self.base_retriever.search_kwargs = {"k": k}
            docs = self.multi_query_retriever.invoke(query)
            return self._deduplicate_docs(docs)
        except Exception as e:
            # Fallback to base retriever if multi-query fails
            print(f"Multi-query retrieval failed: {e}. Using base retriever.")
            return self.base_retriever.invoke(query)
    
    def retrieve_with_compression(self, query: str, k: int = 10) -> list[Document]:
        """Retrieve documents with contextual compression."""
        try:
            self.base_retriever.search_kwargs = {"k": k}
            return self.compression_retriever.invoke(query)
        except Exception as e:
            # Fallback to base retriever
            print(f"Compression retrieval failed: {e}. Using base retriever.")
            return self.base_retriever.invoke(query)
    
    def retrieve_hybrid(self, query: str, k: int = 6) -> list[Document]:
        """Hybrid approach combining multi-query and compression."""
        try:
            # First expand query and retrieve more documents
            self.base_retriever.search_kwargs = {"k": k * 2}
            expanded_docs = self.multi_query_retriever.invoke(query)
            
            # Then apply compression to the expanded set
            if expanded_docs:
                # Create a temporary retriever with the expanded docs
                compressed_docs = self._apply_compression_to_docs(query, expanded_docs)
                return compressed_docs[:k]  # Return top k after compression
            
            return expanded_docs[:k]
            
        except Exception as e:
            print(f"Hybrid retrieval failed: {e}. Using base retriever.")
            return self.base_retriever.invoke(query)
    
    def _apply_compression_to_docs(self, query: str, docs: list[Document]) -> list[Document]:
        """Apply compression scoring to a set of documents."""
        # Simple relevance scoring based on keyword overlap and length
        scored_docs = []
        query_words = set(query.lower().split())
        
        for doc in docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            
            # Score based on overlap and document length (prefer substantial content)
            score = overlap / len(query_words) if query_words else 0
            if len(doc.page_content) > 200:  # Boost longer documents
                score += 0.1
            
            scored_docs.append((score, doc))
        
        # Sort by score and return documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]
    
    def _deduplicate_docs(self, docs: list[Document]) -> list[Document]:
        """Remove duplicate documents based on content similarity."""
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            # Create a simple hash of first 200 characters for deduplication
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def retrieve_with_similarity_threshold(
        self, 
        query: str, 
        similarity_threshold: float = 0.7,
        k: int = 10
    ) -> list[Document]:
        """Retrieve documents with similarity score filtering."""
        try:
            # Get documents with similarity scores
            docs_with_scores = self.base_retriever.vectorstore.similarity_search_with_score(
                query, k=k
            )
            
            # Filter by threshold
            filtered_docs = [
                doc for doc, score in docs_with_scores 
                if score >= similarity_threshold
            ]
            
            return filtered_docs if filtered_docs else [doc for doc, _ in docs_with_scores[:3]]
            
        except Exception as e:
            print(f"Similarity threshold retrieval failed: {e}. Using base retriever.")
            return self.base_retriever.invoke(query)


def create_advanced_retriever(base_retriever: VectorStoreRetriever) -> AdvancedRetriever:
    """Factory function to create an AdvancedRetriever instance."""
    return AdvancedRetriever(base_retriever)