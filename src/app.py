# Streamlit ChromaDB fix
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from ingestion.parser import extract_text_with_metadata, extract_text_chunks
from retriever.text_proc import semantic_chunk_texts, chunk_texts
from retriever.vector_store import build_vector_store
from retriever.advanced_retrieval import create_advanced_retriever
from graph.conversation import create_rag_graph
from agents.crew import create_crew_agent_system
from evaluation.rag_evaluator import RAGEvaluator, EvaluationDataset

load_dotenv()

# === 1. ENVIRONMENT SETUP & PASSWORD PROTECTION ===
load_dotenv()
PASSWORD = os.getenv("APP_PASSWORD")
if PASSWORD is not None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        password_attempt = st.text_input(
            "🔐 Enter password to access app:", type="password"
        )
        if not password_attempt:
            st.stop()
        elif password_attempt == PASSWORD:
            st.session_state.authenticated = True
        else:
            st.warning("Incorrect password")
            st.stop()

st.set_page_config(page_title="Academic PDF RAG", layout="centered")
st.title("📄 Academic PDF Q&A App")

# Configuration sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Processing options
    use_semantic_chunking = st.checkbox("Use Semantic Chunking", value=True, help="Use semantic-aware text chunking for better coherence")
    
    # Retrieval options
    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        ["Standard", "Multi-Query", "Hybrid", "Compression"],
        index=2,
        help="Choose the retrieval enhancement strategy"
    )
    
    chunk_size = st.slider("Chunk Size", 400, 1200, 800, step=100, help="Size of text chunks for processing")
    
    # Evaluation options
    enable_evaluation = st.checkbox("Enable Evaluation", value=False, help="Evaluate system performance (slower)")
    
    if enable_evaluation:
        st.info("Evaluation will analyze retrieval quality and answer accuracy")

# === 2. PDF LOADING & VECTOR INDEX CONSTRUCTION ===
pdf_directory = Path("data/pdfs")
pdf_files = list(pdf_directory.glob("*.pdf"))

if not pdf_files:
    st.warning("No PDFs found in 'data/pdfs'. Please add some files.")
    st.stop()

with st.spinner("Loading and indexing PDFs from directory..."):
    all_docs: list[Document] = []
    
    if use_semantic_chunking:
        # Use enhanced processing with proper metadata
        for pdf_path in pdf_files:
            pages_data = extract_text_with_metadata(pdf_path)
            docs = semantic_chunk_texts(
                pages_data, 
                chunk_size=chunk_size,
                use_semantic_splitting=True
            )
            all_docs.extend(docs)
    else:
        # Use legacy processing for compatibility
        for pdf_path in pdf_files:
            text_chunks = extract_text_chunks(pdf_path)
            docs = chunk_texts(text_chunks, chunk_size=chunk_size, source=pdf_path.name)
            all_docs.extend(docs)

    embedder = OpenAIEmbeddings()
    vstore = build_vector_store(all_docs, embedder)
    base_retriever = vstore.as_retriever()
    
    # Create advanced retriever if needed
    if retrieval_mode != "Standard":
        advanced_retriever = create_advanced_retriever(base_retriever)
        st.session_state['advanced_retriever'] = advanced_retriever
    
    rag_graph = create_rag_graph(base_retriever)
    
    # Initialize evaluation if enabled
    if enable_evaluation:
        evaluator = RAGEvaluator(embedder)
        eval_dataset = EvaluationDataset()
        st.session_state['evaluator'] = evaluator
        st.session_state['eval_dataset'] = eval_dataset

st.success(f"Indexed {len(all_docs)} text chunks from {len(pdf_files)} files.")

# === 3. UI STATE INIT & OPTION FOR CREWAI ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about the documents")
use_crew = st.checkbox("Use CrewAI agents for deeper analysis")

# === 4. MAIN Q&A HANDLER: RAG OR CREWAI ===
if question:
    with st.spinner("Processing your question..."):
        start_time = st.empty()
        
        if use_crew:
            crew = create_crew_agent_system()
            
            # Use advanced retrieval if configured
            if retrieval_mode != "Standard" and 'advanced_retriever' in st.session_state:
                adv_retriever = st.session_state['advanced_retriever']
                if retrieval_mode == "Multi-Query":
                    context_docs = adv_retriever.retrieve_with_multi_query(question)
                elif retrieval_mode == "Compression":
                    context_docs = adv_retriever.retrieve_with_compression(question)
                elif retrieval_mode == "Hybrid":
                    context_docs = adv_retriever.retrieve_hybrid(question)
            else:
                context_docs = base_retriever.invoke(question)
            
            # Vectorized context formatting
            context_parts = [
                f"[Source: {doc.metadata.get('source', 'unknown')} - Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in context_docs
            ]
            context = "\n\n".join(context_parts)
            
            result = crew.kickoff(inputs={"context": context, "question": question})
            answer = result
            sources = {
                f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})"
                for doc in context_docs
            }
            
        else:
            # Enhanced RAG workflow
            if retrieval_mode != "Standard" and 'advanced_retriever' in st.session_state:
                # Override retriever in graph with advanced retrieval
                adv_retriever = st.session_state['advanced_retriever']
                
                if retrieval_mode == "Multi-Query":
                    retrieved_docs = adv_retriever.retrieve_with_multi_query(question)
                elif retrieval_mode == "Compression":
                    retrieved_docs = adv_retriever.retrieve_with_compression(question)
                elif retrieval_mode == "Hybrid":
                    retrieved_docs = adv_retriever.retrieve_hybrid(question)
                
                # Manually construct context for state
                context_parts = [
                    f"[Document {i}: {doc.metadata.get('source', 'unknown')} - Page {doc.metadata.get('page', '?')} ({doc.metadata.get('chunk_type', 'content')})]\n{doc.page_content}"
                    for i, doc in enumerate(retrieved_docs, 1)
                ]
                context = "\n\n---\n\n".join(context_parts)
                
                state = {
                    "question": question,
                    "context": context,
                    "docs": retrieved_docs,
                    "answer": "",
                    "sources": [],
                    "query_type": None
                }
                
            else:
                state = {
                    "question": question,
                    "context": "",
                    "docs": [],
                    "answer": "",
                    "sources": [],
                    "query_type": None
                }
            
            result = rag_graph.invoke(state)
            answer = result["answer"]
            context_docs = result.get("docs", [])
            sources = {
                f"{meta.get('source', 'Unknown')} (page {meta.get('page', '?')})"
                for meta in result.get("sources", [])
            }

        # Evaluation if enabled
        evaluation_result = None
        if enable_evaluation and 'evaluator' in st.session_state:
            evaluator = st.session_state['evaluator']
            eval_dataset = st.session_state['eval_dataset']
            
            evaluation_result = evaluator.comprehensive_evaluation(
                question, answer, result.get("context", ""), context_docs, result.get("sources", [])
            )
            eval_dataset.save_evaluation(evaluation_result)

    st.session_state.chat_history.append((question, answer, sources, evaluation_result))

    # === 5. DISPLAY ANSWER AND SOURCES ===
    st.markdown("### 📌 Answer")
    st.write(answer)

    with st.expander("📁 Sources used"):
        for source in sorted(sources):
            st.markdown(f"- {source}")
    
    # Display evaluation results if available
    if enable_evaluation and evaluation_result:
        with st.expander("📊 Evaluation Results"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{evaluation_result['overall_score']:.2f}/5")
            with col2:
                st.metric("Retrieval Score", f"{evaluation_result['retrieval']['top_k_relevance']:.2f}")
            with col3:
                st.metric("Answer Quality", f"{evaluation_result['answer_quality']['overall']}/5")
            
            if evaluation_result['answer_quality'].get('reasoning'):
                st.text_area("Evaluation Reasoning", evaluation_result['answer_quality']['reasoning'], height=100)

# === 6. DISPLAY CONVERSATION HISTORY ===
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for item in st.session_state.chat_history:
        if len(item) >= 3:  # Handle both old and new format
            q, a, s_list = item[:3]
            eval_result = item[3] if len(item) > 3 else None
            
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                with st.expander("Sources"):
                    for src in sorted(s_list):
                        st.markdown(f"- {src}")
            
            if eval_result and enable_evaluation:
                with col2:
                    st.metric("Score", f"{eval_result['overall_score']:.1f}/5")

# === 7. EVALUATION DASHBOARD ===
if enable_evaluation and 'eval_dataset' in st.session_state:
    eval_dataset = st.session_state['eval_dataset']
    summary = eval_dataset.get_metrics_summary()
    
    if summary.get("total_evaluations", 0) > 0:
        st.markdown("### 📈 Performance Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", summary["total_evaluations"])
        with col2:
            st.metric("Avg Overall Score", f"{summary['average_overall_score']:.2f}/5")
        with col3:
            st.metric("Avg Retrieval", f"{summary['average_retrieval_score']:.2f}")
        with col4:
            st.metric("Latest Score", f"{summary['latest_score']:.2f}/5")
