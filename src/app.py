import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from ingestion.parser import extract_text_chunks
from retriever.embedder import chunk_texts, get_embedder
from retriever.vector_store import build_vector_store
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from agents.crew import create_crew_agent_system

load_dotenv()

# ==== Basic Password Protection ====
PASSWORD = os.getenv("APP_PASSWORD")
if PASSWORD:
    password_attempt = st.text_input(
        "🔐 Enter password to access app:", type="password"
    )
    if password_attempt != PASSWORD:
        st.warning("Incorrect password")
        st.stop()

st.set_page_config(page_title="Academic PDF RAG", layout="centered")
st.title("📄 Knowledge Base Query App")

pdf_directory = Path("data/uploads")
pdf_files = list(pdf_directory.glob("*.pdf"))

if not pdf_files:
    st.warning("No PDFs found in 'data/uploads'. Please add some files.")
    st.stop()

with st.spinner("Loading and indexing PDFs from directory..."):
    all_docs = []
    for pdf_path in pdf_files:
        text_chunks = extract_text_chunks(pdf_path)
        docs = chunk_texts(text_chunks)
        all_docs.extend(docs)

    embedder = get_embedder()
    vstore = build_vector_store(all_docs, embedder)
    retriever = vstore.as_retriever()

st.success(f"Indexed {len(all_docs)} text chunks from {len(pdf_files)} files.")

question = st.text_input("Ask a question about the documents")
use_crew = st.checkbox("Use CrewAI agents for deeper analysis")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if question:
    with st.spinner("Processing answer..."):
        if use_crew:
            crew = create_crew_agent_system()
            result = crew.kickoff()
        else:
            llm = ChatOpenAI(temperature=0)
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory
            )
            result = qa_chain.run(question)

        st.session_state.chat_history.append((question, result))

        st.markdown("### 📌 Answer")
        st.write(result)

if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
