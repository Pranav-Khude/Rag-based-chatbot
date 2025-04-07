# RAG-Based Chatbot

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, BM25, and the Ollama LLM. The chatbot leverages a hybrid retrieval system combining lexical (BM25) and semantic (embedding-based) methods to fetch relevant document chunks from a knowledge base, then generates context-aware responses using the `llama3.1` model. It is designed to answer user queries efficiently using only the provided document context.

## Project Structure
```
├── embeddings.py          # Creates and manages the FAISS vector store for embeddings
├── chunking.py           # Loads and splits documents into chunks
├── app.py                # Flask API for chatbot interaction
├── llm_inference.py      # Handles LLM inference with hybrid retrieval augmentation
├── search/
│   ├── hybrid_retriever.py  # Implements HybridRetriever combining BM25 and embeddings
│   ├── bm25_retriever.py    # BM25-based lexical retriever
├── utils/
│   ├── embeddings.py     # Embedding utilities (e.g., get_retriever())
│   ├── chunking.py       # Document chunking logic
├── data/
│   ├── pdfs/             # PDF documents for the knowledge base
│   ├── documents/        # Text documents for the knowledge base
├── embeddings/           # Directory to store FAISS index
```

## Installation
### Prerequisites
- Python 3.9+
- Pip
- Optional: Virtual environment (recommended)

### Create a Virtual Environment
To isolate dependencies, create and activate a virtual environment:
```bash
python -m venv rag_env
source rag_env/bin/activate  # On macOS/Linux
rag_env\Scripts\activate     # On Windows
```

### Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```
*Note*: Ensure `requirements.txt` includes `langchain`, `faiss-cpu`, `sentence-transformers`, `nltk`, `flask`, and `ollama`.

## Usage
### 1. Create the Vector Store
Generate the FAISS vector store for embeddings by running:
```bash
python embeddings.py
```
This processes documents from `data/` and saves embeddings to `embeddings/faiss_index`.

### 2. Start the API Server
Launch the Flask application:
```bash
python app.py
```
The server runs at `http://127.0.0.1:5000/`.

### 3. Query the Chatbot
Send a POST request to the `/query` endpoint with a JSON payload:
```json
{
  "query": "Tell me about the functionality of the urinary system."
}
```
Using `curl`:
```bash
curl -X POST "http://127.0.0.1:5000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Tell me about the functionality of the urinary system."}'
```
**Response**: A JSON object with the chatbot’s answer based on retrieved context.

## Implementation Details
### Embeddings and Retrieval (`embeddings.py`, `search/`)
- **Vector Store**: Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) with FAISS for semantic retrieval.
- **Hybrid Retrieval**: The `HybridRetriever` (in `search/hybrid_retriever.py`) combines:
  - **BM25**: Lexical retrieval for exact keyword matches (`bm25_retriever.py`).
  - **Embeddings**: Semantic similarity via FAISS.
  - Controlled by an `alpha` parameter (default 0.5) to balance lexical and semantic scores.
- **Retriever**: Accessible via `HybridRetriever().as_retriever()` for LangChain compatibility.

### Document Processing (`chunking.py`)
- Loads PDFs and text files from `data/`.
- Splits documents into 500-character chunks with a 50-character overlap using `RecursiveCharacterTextSplitter`.
- For alternative chunking strategies, see [Chunking-Techniques-RAG](https://github.com/Pranav-Khude/Chunking-Techniques-RAG).

### Query Handling (`app.py`)
- Flask API endpoint `/query` accepts POST requests with a `query` field.
- Invokes `get_response(query)` from `llm_inference.py` to process the query.

### Response Generation (`llm_inference.py`)
- **Retrieval**: Uses `HybridRetriever` to fetch the top `k` relevant chunks (default `k=3`).
- **LLM**: Employs `OllamaLLM` with the `llama3.1` model.
- **Prompt**: A structured `ChatPromptTemplate` ensures responses are:
  - Based solely on retrieved context.
  - Generated step-by-step.
  - Fallback: "I do not have enough information to answer that" if context is insufficient.

Example prompt:
```python
ChatPromptTemplate.from_messages([
    ("system", """You are a helpful chatbot that answers using only the provided context.  
    If the context lacks the answer, say: 'I do not have enough information to answer that.'  
    Think step-by-step.  
    Context: {context}"""),
    ("human", "{input}")
])
```

## Future Enhancements
- Add support for multi-modal inputs (e.g., images).
- Optimize `alpha` tuning for hybrid retrieval based on query type.
- Integrate real-time document updates to the knowledge base.
- Follow [my GitHub](https://github.com/Pranav-Khude) for updates.

## Troubleshooting
- **Missing FAISS Index**: Ensure `embeddings.py` runs successfully before starting `app.py`.
- **Dependencies**: Verify all packages are installed (`pip list`).
- **Errors**: Check console output for detailed stack traces.

---

### Updates Made
1. **Structure**: Added `search/` directory with `hybrid_retriever.py` and `bm25_retriever.py` to reflect your hybrid approach.
2. **Clarity**: Improved explanations (e.g., hybrid retrieval, prompt details) and standardized terminology (e.g., "knowledge base").
3. **Details**: Included specific model names (`all-MiniLM-L6-v2`, `llama3.1`) and updated the prompt example.
4. **Future Enhancements**: Expanded with practical next steps based on RAG trends.
5. **Commands**: Added quotes to JSON in `curl` example for correctness.
