import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from ingestion.parser import extract_text_chunks
from retriever.text_proc import chunk_texts
from retriever.vector_store import build_vector_store
from graph.conversation import create_rag_graph
from agents.crew import create_crew_agent_system

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

# === 2. PDF LOADING & VECTOR INDEX CONSTRUCTION ===
pdf_directory = Path("data/pdfs")
pdf_files = list(pdf_directory.glob("*.pdf"))

if not pdf_files:
    st.warning("No PDFs found in 'data/pdfs'. Please add some files.")
    st.stop()

with st.spinner("Loading and indexing PDFs from directory..."):
    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        text_chunks = extract_text_chunks(pdf_path)
        docs = chunk_texts(text_chunks, source=pdf_path.name)
        all_docs.extend(docs)

    embedder = OpenAIEmbeddings()
    vstore = build_vector_store(all_docs, embedder)
    retriever = vstore.as_retriever()
    rag_graph = create_rag_graph(retriever)

st.success(f"Indexed {len(all_docs)} text chunks from {len(pdf_files)} files.")

# === 3. UI STATE INIT & OPTION FOR CREWAI ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about the documents")
use_crew = st.checkbox("Use CrewAI agents for deeper analysis")

# === 4. MAIN Q&A HANDLER: RAG OR CREWAI ===
if question:
    if use_crew:
        crew = create_crew_agent_system()
        context_docs = retriever.invoke(question)
        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'unknown')} - Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in context_docs
            ]
        )
        result = crew.kickoff(inputs={"context": context, "question": question})
        answer = result
        sources = {
            f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})"
            for doc in context_docs
        }
    else:
        state = {
            "question": question,
            "context": "",
            "docs": [],
            "answer": "",
            "sources": [],
        }
        result = rag_graph.invoke(state)
        answer = result["answer"]
        sources = {
            f"{meta.get('source', 'Unknown')} (page {meta.get('page', '?')})"
            for meta in result.get("sources", [])
        }

    st.session_state.chat_history.append((question, answer, sources))

    # === 5. DISPLAY ANSWER AND SOURCES ===
    st.markdown("### 📌 Answer")
    st.write(answer)

    with st.expander("📁 Sources used"):
        for source in sorted(sources):
            st.markdown(f"- {source}")

# === 6. DISPLAY CONVERSATION HISTORY ===
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for q, a, s_list in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        with st.expander("Sources"):
            for src in sorted(s_list):
                st.markdown(f"- {src}")
