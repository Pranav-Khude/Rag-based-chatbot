import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import json
from typing import List, Dict
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class BM25Retriever:
    def __init__(self, chunk_file="document_chunks.json"):
        self.bm25 = None
        self.documents = []
        self.chunk_file = chunk_file

    def load_chunked_documents(self):
        """Load chunked documents from file."""
        try:
            with open(self.chunk_file, 'r') as f:
                chunks = json.load(f)
                self.documents = [chunk["text"] for chunk in chunks]
                print(f"Loaded {len(self.documents)} chunked documents for BM25 retrieval.")
        except FileNotFoundError:
            raise FileNotFoundError("Chunked document file not found. Run DocumentLoader first.")

    def fit(self):
        """Initialize BM25 with chunked documents."""
        if not self.documents:
            self.load_chunked_documents()

        tokenized_docs = [word_tokenize(doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top_k most relevant document chunks using BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 model is not initialized. Call `fit()` first.")

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k document indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        return results


if __name__ == "__main__":
    retriever = BM25Retriever()
    retriever.fit()

    # Test retrieval with a sample query
    query = "explain hypertension"
    results = retriever.retrieve(query, top_k=4)

    print("\nTop retrieved documents:")
    # print(results[0].keys())
    for res in results:
        print(f"Index: {res['index']}, Score: {res['score']:.4f}")
        print(f"Text: {res['text']}\n")
