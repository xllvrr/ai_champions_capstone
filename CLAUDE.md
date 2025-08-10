# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
- `streamlit run src/app.py` - Start the Streamlit web application
- The app requires an OpenAI API key in environment variables
- Password protection is optional via `APP_PASSWORD` environment variable

### Development Environment
- This project uses `uv` for Python package management
- Python 3.11+ required
- Dependencies are managed in `pyproject.toml`
- Uses `devenv` for development environment setup

### Project Setup
```bash
# Install dependencies
uv sync

# Set up environment variables (create .env file)
# Required: OPENAI_API_KEY=your_api_key_here
# Optional: APP_PASSWORD=your_password_here
```

## Architecture Overview

This is an Academic PDF RAG (Retrieval-Augmented Generation) application with the following architecture:

### Core Components

1. **Main Application (`src/app.py`)**
   - Streamlit-based web interface
   - Handles PDF loading from `data/pdfs/` directory
   - Provides both standard RAG and CrewAI-enhanced responses
   - Includes SQLite3 compatibility fix for ChromaDB on Streamlit Cloud

2. **Document Processing Pipeline**
   - **PDF Ingestion** (`src/ingestion/parser.py`): Uses PyMuPDF to extract text from PDFs
   - **Text Processing** (`src/retriever/text_proc.py`): Chunks text using LangChain's RecursiveCharacterTextSplitter
   - **Vector Storage** (`src/retriever/vector_store.py`): Creates in-memory vector store with OpenAI embeddings

3. **RAG Implementation**
   - **LangGraph Workflow** (`src/graph/conversation.py`): Implements RAG using LangGraph state machine with retrieve→generate pattern
   - **CrewAI Integration** (`src/agents/crew.py`): Alternative agent-based approach using CrewAI for deeper analysis

### Technology Stack
- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o and GPT-4o-mini
- **Vector Database**: LangChain InMemoryVectorStore with OpenAI embeddings
- **Document Processing**: PyMuPDF, LangChain text splitters
- **Orchestration**: LangGraph for RAG workflow, CrewAI for agent-based processing
- **Environment**: Python 3.11+, uv package manager, devenv

### Data Flow
1. PDFs are loaded from `data/pdfs/` directory
2. Text is extracted and chunked with metadata (source file, page number)
3. Chunks are embedded and stored in vector database
4. User questions trigger retrieval of relevant chunks
5. LLM generates answers using retrieved context
6. Sources are displayed with page references

### Key Features
- Dual processing modes: standard RAG vs CrewAI agents
- Conversation history maintained in session state
- Source attribution with file names and page numbers
- Password protection for deployment
- SQLite3 compatibility fixes for cloud deployment

## New Features (Enhanced RAG System)

### Advanced Processing Options
- **Semantic Chunking**: Intelligent text segmentation based on content structure and meaning
- **Multiple Retrieval Strategies**: Standard, Multi-Query, Hybrid, and Compression modes
- **Query Classification**: Automatic detection and specialized handling of factual, analytical, summary, and definition questions
- **Evaluation Framework**: Built-in performance monitoring and quality assessment

### Configuration Options
- Adjustable chunk sizes (400-1200 characters)
- Configurable retrieval strategies for different use cases
- Optional performance evaluation and analytics
- Enhanced metadata tracking with chunk types and improved source attribution

### Performance Improvements
- Vectorized operations for better performance with large document sets
- Optimized text preprocessing and cleaning
- Enhanced context formatting with better document organization
- Improved error handling and fallback mechanisms

## Important Notes

- The app requires PDFs in the `data/pdfs/` directory to function
- OpenAI API key must be configured in environment variables
- The app uses in-memory vector storage (data is not persisted between sessions)
- CrewAI mode provides more detailed analysis but may be slower
- Backups of original files are stored in `backups/` directory
- Evaluation results are saved to `evaluation_results.json` when enabled