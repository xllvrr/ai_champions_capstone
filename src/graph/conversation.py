from typing import TypedDict, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langchain.chat_models import init_chat_model


class RAGState(TypedDict):
    question: str
    context: str
    docs: list[Document]
    answer: Any
    sources: list[Any]
    query_type: str | None


def _classify_query(question: str) -> str:
    """Basic query classification for different response strategies."""
    question_lower = question.lower()
    
    # Factual questions
    if any(word in question_lower for word in ["what", "who", "when", "where", "which"]):
        return "factual"
    
    # Analytical questions
    elif any(word in question_lower for word in ["why", "how", "analyze", "compare", "evaluate"]):
        return "analytical"
    
    # Summary questions
    elif any(word in question_lower for word in ["summarize", "overview", "main points", "key findings"]):
        return "summary"
    
    # Definition questions
    elif any(word in question_lower for word in ["define", "meaning", "concept", "term"]):
        return "definition"
    
    return "general"


def _get_enhanced_prompt(query_type: str) -> ChatPromptTemplate:
    """Return specialized prompts based on query type."""
    
    base_instructions = """You are an expert academic research assistant. Use ONLY the provided document context to answer the question accurately and comprehensively."""
    
    if query_type == "factual":
        return ChatPromptTemplate.from_template(f"""
        {base_instructions}
        
        Provide a direct, factual answer based on the evidence in the documents. 
        If specific data, numbers, or facts are mentioned, include them.
        Cite the specific sources for each fact.

        Context:
        {"{context}"}

        Question: {"{question}"}
        
        Answer with facts and evidence from the provided sources:""")
    
    elif query_type == "analytical":
        return ChatPromptTemplate.from_template(f"""
        {base_instructions}
        
        Provide a thorough analysis addressing the question. Consider multiple perspectives,
        examine relationships between concepts, and draw insights from the evidence.
        Structure your response with clear reasoning and support each point with source evidence.

        Context:
        {"{context}"}

        Question: {"{question}"}
        
        Analysis based on the provided sources:""")
    
    elif query_type == "summary":
        return ChatPromptTemplate.from_template(f"""
        {base_instructions}
        
        Provide a comprehensive summary that captures the main points and key findings.
        Organize information logically and highlight the most important insights.
        Include specific details and evidence from the sources.

        Context:
        {"{context}"}

        Question: {"{question}"}
        
        Summary based on the provided sources:""")
    
    elif query_type == "definition":
        return ChatPromptTemplate.from_template(f"""
        {base_instructions}
        
        Provide a clear, precise definition along with relevant context and examples from the sources.
        Explain the concept thoroughly and mention any variations or related terms discussed.

        Context:
        {"{context}"}

        Question: {"{question}"}
        
        Definition based on the provided sources:""")
    
    else:  # general
        return ChatPromptTemplate.from_template(f"""
        {base_instructions}
        
        Provide a helpful, comprehensive answer based on the document context.
        Structure your response clearly and support your points with evidence from the sources.

        Context:
        {"{context}"}

        Question: {"{question}"}
        
        Answer based on the provided sources:""")


def create_rag_graph(retriever: VectorStoreRetriever):
    graph = StateGraph(state_schema=RAGState)
    llm = init_chat_model("gpt-4o", temperature=0)

    def classify_step(state: RAGState) -> RAGState:
        """Classify the query type for specialized processing."""
        question = state["question"]
        query_type = _classify_query(question)
        
        return {
            "question": question,
            "context": state["context"],
            "docs": state["docs"],
            "answer": state["answer"],
            "sources": state["sources"],
            "query_type": query_type
        }

    def retrieve_step(state: RAGState) -> RAGState:
        question = state["question"]
        docs = retriever.invoke(question)
        
        # Vectorized context formatting for better performance
        sources = [d.metadata.get('source', 'unknown') for d in docs]
        pages = [str(d.metadata.get('page', '?')) for d in docs]
        chunk_types = [d.metadata.get('chunk_type', 'content') for d in docs]
        contents = [d.page_content for d in docs]
        
        # Use list comprehension with enumerate for efficient formatting
        context_parts = [
            f"[Document {i}: {source} - Page {page} ({chunk_type})]\n{content}"
            for i, (source, page, chunk_type, content) in enumerate(zip(sources, pages, chunk_types, contents), 1)
        ]
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {
            "question": question,
            "context": context,
            "docs": docs,
            "answer": state["answer"],  # propagate existing value
            "sources": state["sources"],  # propagate existing value
            "query_type": state.get("query_type", "general")
        }

    def generate_step(state: RAGState) -> RAGState:
        query_type = state.get("query_type", "general")
        prompt = _get_enhanced_prompt(query_type)
        
        # Prompt LLM with specialized template
        prompt_text = prompt.format(
            context=state["context"], question=state["question"]
        )
        llm_response = llm.invoke(prompt_text)

        # Safely extract just the answer string
        if hasattr(llm_response, "content"):
            answer = llm_response.content
        elif isinstance(llm_response, dict) and "content" in llm_response:
            answer = llm_response["content"]
        else:
            answer = str(llm_response)

        # Vectorized source metadata extraction
        docs = state["docs"]
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "chunk_type": doc.metadata.get("chunk_type", "content"),
                "chunk_id": doc.metadata.get("chunk_id", "")
            }
            for doc in docs
        ]
        
        return {
            "question": state["question"],
            "context": state["context"],
            "docs": state["docs"],
            "answer": answer,
            "sources": sources,
            "query_type": query_type
        }

    _ = graph.add_node("classify", classify_step)
    _ = graph.add_node("retrieve", retrieve_step)
    _ = graph.add_node("generate", generate_step)
    _ = graph.add_edge("classify", "retrieve")
    _ = graph.add_edge("retrieve", "generate")
    _ = graph.set_entry_point("classify")
    _ = graph.set_finish_point("generate")
    return graph.compile()
