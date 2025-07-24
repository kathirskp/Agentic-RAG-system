# Agentic-RAG-system

This project implements a complete RAG system that allows you to upload documents and ask questions based on their content.

# Features

# 1. Document Processing & Ingestion
- Support for multiple file formats (PDF, DOCX, TXT)
- Text extraction and preprocessing
- Document chunking with overlap for better context

# 2. Vector Database & Embeddings
- FAISS vector database for efficient similarity search
- Sentence-BERT embeddings (all-MiniLM-L6-v2)
- Cosine similarity for document retrieval

# 3. RAG Pipeline
- Document retrieval based on query similarity
- Context-aware response generation
- Source attribution and relevance scoring

# 4. User Interface
- Streamlit-based web interface
- File upload functionality
- Real-time query processing
- Interactive chat-like experience

# 5. Open Source Integration
- Uses open-source embedding models
- Compatible with LangSmith API for tracking
- No dependency on paid APIs for core functionality

 # Quick Start

# Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

# Running the Application

```bash
streamlit run rag_system.py
```

 # How to Use

1. **Start the application** using the commands above
2. **Upload documents** via the sidebar:
   - Option 1: Paste text directly
   - Option 2: Upload PDF, DOCX, or TXT files
3. **Process documents** by clicking the respective button
4. **Ask questions** in the main interface
5. **View results** with relevance scores and source attribution

 # System Architecture

```
Documents → Text Extraction → Chunking → Embeddings → Vector Store
                                                           ↓
User Query → Query Embedding → Similarity Search → Context Retrieval → Response
```

 # Features

- **Multi-format Support**: PDF, DOCX, TXT files
- **Intelligent Chunking**: Overlapping chunks for better context
- **Semantic Search**: Uses sentence transformers for meaning-based retrieval
- **Real-time Processing**: Instant document processing and querying
- **Relevance Scoring**: Shows how relevant each retrieved document is
- **Source Attribution**: Tracks which documents provided the information

 # Technical Details

# Models Used
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Text Processing**: Custom chunking with sentence boundary detection

# Configuration
- **Chunk Size**: 500-1000 characters
- **Chunk Overlap**: 50-200 characters
- **Retrieval Count**: Top 3-5 most relevant chunks
- **Similarity Metric**: Cosine similarity

 # Performance

- **Processing Speed**: ~1-2 seconds per document
- **Query Response**: <1 second for most queries
- **Memory Usage**: Optimized for local deployment
- **Scalability**: Handles documents up to several MB

# Testing the System

# Sample Test Cases

1. **Upload a document** (PDF, DOCX, or TXT)
2. **Try these sample queries**:
   - "What is the main topic of this document?"
   - "Can you summarize the key points?"
   - "What are the important details mentioned?"
   - "Tell me about [specific topic from your document]"

# Expected Results
- Relevant text chunks retrieved from your documents
- Similarity scores showing relevance (0.0 to 1.0)
- Source attribution showing which document provided the information

 # Use Cases

- **Document Q&A**: Ask questions about uploaded documents
- **Content Summarization**: Get key points from large documents
- **Information Retrieval**: Find specific information quickly
- **Research Assistance**: Explore document collections efficiently

# Customization

You can modify the system by:
- Changing chunk sizes in the code
- Adjusting the number of retrieved documents
- Modifying the embedding model
- Customizing the response format

 Please note:

- The system works entirely offline after initial model download
- No API keys required for basic functionality
- LangSmith integration available for advanced tracking
- Optimized for educational and demonstration purposes
