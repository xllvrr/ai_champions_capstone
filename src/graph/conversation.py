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


def create_rag_graph(retriever: VectorStoreRetriever):
    graph = StateGraph(state_schema=RAGState)
    llm = init_chat_model("gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful academic assistant.
        Use ONLY the following document context to answer the question.

        {context}

        Question: {question}
        """
    )

    def retrieve_step(state: RAGState) -> RAGState:
        question = state["question"]
        docs = retriever.invoke(question)
        context = "\n\n".join(
            [
                f"[Source: {d.metadata.get('source', 'unknown')} - Page {d.metadata.get('page', '?')}]\n{d.page_content}"
                for d in docs
            ]
        )
        return {
            "question": question,
            "context": context,
            "docs": docs,
            "answer": state["answer"],  # propagate existing value
            "sources": state["sources"],  # propagate existing value
        }

    def generate_step(state: RAGState) -> RAGState:

        # Prompt LLM
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

        # Get sources from the knowledge base
        sources = [d.metadata for d in state["docs"]]
        return {
            "question": state["question"],
            "context": state["context"],
            "docs": state["docs"],
            "answer": answer,
            "sources": sources,
        }

    _ = graph.add_node("retrieve", retrieve_step)
    _ = graph.add_node("generate", generate_step)
    _ = graph.add_edge("retrieve", "generate")
    _ = graph.set_entry_point("retrieve")
    _ = graph.set_finish_point("generate")
    return graph.compile()
