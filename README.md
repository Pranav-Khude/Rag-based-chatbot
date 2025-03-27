# RAG-Based Chatbot

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, and Ollama LLM. The chatbot retrieves relevant documents from a vector store and generates responses based on the retrieved content. It is designed to answer user queries using the most relevant information from stored documents.

## Project Structure
```
├── embeddings.py          # Creates and manages the FAISS vector store
├── chunking.py           # Loads and splits documents for efficient retrieval
├── app.py                # Flask API for chatbot interaction
├── llm_inference.py      # Handles LLM-based inference with retrieval augmentation
├── utils/
│   ├── embeddings.py     # Contains get_retriever() function
│   ├── chunking.py       # Contains document chunking logic
├── data/
│   ├── pdfs/             # PDF documents for knowledge base
│   ├── documents/        # Text documents for knowledge base
├── embeddings/           # Directory to store FAISS index
```

## Installation
### Prerequisites
- Python 3.9+
- Pip

### Create a Virtual Environment
It is recommended to create a virtual environment to manage dependencies independently.
```bash
python -m venv rag_env
source rag_env/bin/activate  # On macOS/Linux
rag_env\Scripts\activate    # On Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Create the Vector Store
Before using the chatbot, you need to create the FAISS vector store by running:
```bash
python embeddings.py
```
This will process documents and store their embeddings in `embeddings/faiss_index`.

### 2. Start the API Server
Run the Flask application:
```bash
python app.py
```
The server will start at `http://127.0.0.1:5000/`.

### 3. Query the Chatbot
Send a POST request to `/query` with a JSON payload:
```json
{
  "query": "Tell me about the functionality of The Urinary System."
}
```
Example using `curl`:
```bash
curl -X POST "http://127.0.0.1:5000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Tell me about the functionality of The Urinary System."}'
```

## Implementation Details
### Embeddings and Retrieval (`embeddings.py`)
- Documents are loaded and split into chunks.
- The FAISS vector store is created using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
- The retriever fetches relevant documents based on query similarity.

### Document Processing (`chunking.py`)
- Loads text and PDF files from `data/`.
- Splits documents into 500-character chunks with a 50-character overlap using `RecursiveCharacterTextSplitter`.
- For different chunking method implementations, refer to my [GitHub repository](https://github.com/Pranav-Khude/Chunking-Techniques-RAG).

### Query Handling (`app.py`)
- The Flask API receives user queries.
- Calls `get_response(query)` from `llm_inference.py`.

### Response Generation (`llm_inference.py`)
- Retrieves relevant documents.
- Uses the `llama3.1` model via `OllamaLLM`.
- Generates responses using a structured prompt.

## Future Enhancements
Follow my GitHub to stay updated on future enhancements.


