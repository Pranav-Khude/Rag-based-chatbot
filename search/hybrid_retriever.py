# import sys
# sys.path.append("D:/RAG Pipeline/RAG based chatbot/")

from search.bm25_retriever import BM25Retriever
from utils.embeddings import EmbeddingsHandler
import numpy as np
from typing import List, Dict, Any
from nltk.tokenize import word_tokenize
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

class HybridRetriever:
    def __init__(self, alpha=0.5):
        """ 
        HybridRetriever combines BM25 and Embedding retrievers to get the best of both worlds.
        The alpha parameter controls the balance between the two retrievers.
        A value of 0.5 means equal weightage to both retrievers.
        """
        
        self.lexical_retriever = BM25Retriever()
        self.semantic_handler = EmbeddingsHandler()
        self.semantic_retriever = self.semantic_handler.get_retriever()
        self.alpha = alpha
        self.lexical_retriever.fit()
        
        # Compute global normalization parameters for BM25
        self.bm25_min, self.bm25_max = self.compute_bm25_bounds()
        
    def compute_bm25_bounds(self):
        """Compute min and max BM25 scores across the corpus using sample queries."""
        print("Computing BM25 score bounds for normalization...")
        
        # If the documents are not loaded, load them
        if not self.lexical_retriever.documents:
            self.lexical_retriever.load_chunked_documents()
            
        # Use a sample of documents as queries to get diverse score distributions
        sample_size = min(50, len(self.lexical_retriever.documents))
        sample_indices = np.random.choice(len(self.lexical_retriever.documents), sample_size, replace=False)
        
        all_scores = []
        for idx in sample_indices:
            # Use the first sentence of each document as a query
            query = self.lexical_retriever.documents[idx].split('.')[0]
            if len(query) > 10:  # Ensure query is meaningful
                # Tokenize the query the same way BM25Retriever does
                tokenized_query = word_tokenize(query.lower())
                # Get scores from BM25
                scores = self.lexical_retriever.bm25.get_scores(tokenized_query)
                all_scores.extend(scores)
                
        if not all_scores:
            return 0, 1  # Default values if no scores
            
        return np.min(all_scores), np.max(all_scores)

    def normalize_bm25_score(self, score):
        """Normalize a BM25 score using global min-max values."""
        if self.bm25_max == self.bm25_min:
            return 1.0
        return (score - self.bm25_min) / (self.bm25_max - self.bm25_min + 1e-6)
    
    def normalize_list(self, scores):
        """Normalize a list of scores to range [0,1]."""
        min_val = min(scores) if scores else 0
        max_val = max(scores) if scores else 1
        
        if max_val == min_val:
            return [1.0 for _ in scores]
            
        return [(s - min_val) / (max_val - min_val + 1e-6) for s in scores]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant document chunks using a combination of BM25 and Embedding retrievers.
        """
        # Get results from BM25 retriever
        lexical_results = self.lexical_retriever.retrieve(query, top_k=top_k*2)
        
        # Normalize BM25 scores using global corpus statistics
        for result in lexical_results:
            result['score'] = self.normalize_bm25_score(result['score'])
            # Add source information
            result['source'] = 'lexical'
        
        # Get results from semantic retriever
        semantic_docs = self.semantic_retriever.get_relevant_documents(query, k=top_k*2)
        
        # Convert semantic results to same format and normalize by rank
        # (FAISS doesn't provide actual similarity scores through the LangChain interface)
        semantic_results = []
        for i, doc in enumerate(semantic_docs):
            # Use rank-based scoring (decreasing from 1.0 to 0.0)
            norm_score = 1.0
            if len(semantic_docs) > 1:
                norm_score = 1.0 - (i / (len(semantic_docs) - 1))
                
            semantic_results.append({
                'text': doc.page_content,
                'score': norm_score,
                'metadata': doc.metadata,
                'source': 'semantic' 
            })

        # Combine results using weighted scores
        combined_results = {}

        # Add lexical results with weight (1-alpha)
        for result in lexical_results:
            text = result['text']
            combined_results[text] = {
                'text': text,
                'score': result['score'] * (1 - self.alpha),
                'metadata': getattr(result, 'metadata', {}),
                'source': ['lexical']  # Track sources as a list
            }

        # Add semantic results with weight alpha
        for result in semantic_results:
            text = result['text']
            if text in combined_results:
                combined_results[text]['score'] += result['score'] * self.alpha
                # Track that this document came from both sources
                combined_results[text]['source'].append('semantic')
                # Merge metadata if needed
                if 'metadata' in result and result['metadata']:
                    combined_results[text]['metadata'].update(result['metadata'])
            else:
                combined_results[text] = {
                    'text': text,
                    'score': result['score'] * self.alpha,
                    'metadata': result.get('metadata', {}),
                    'source': ['semantic']  # Track sources as a list
                }

        # Sort by final score (descending)
        sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_results[:top_k]

    def as_retriever(self):
        """Returns a LangChain-compatible retriever by subclassing BaseRetriever."""
        class HybridRetrieverWrapper(BaseRetriever):
            hybrid_retriever: 'HybridRetriever' = Field(...)

            class Config:
                arbitrary_types_allowed = True  # Allow custom types like HybridRetriever

            def __init__(self, hybrid_retriever_instance):
                super().__init__(hybrid_retriever=hybrid_retriever_instance)
                
            def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs):
                top_k = kwargs.get('k', 3)
                results = self.hybrid_retriever.retrieve(query, top_k=top_k)
                
                # Convert to Document objects
                documents = []
                for result in results:
                    metadata = result.get('metadata', {}).copy()
                    metadata.update({"score": result['score'], "source": result['source']})
                    doc = Document(page_content=result['text'], metadata=metadata)
                    documents.append(doc)
                    
                return documents

        return HybridRetrieverWrapper(self)


if __name__ == "__main__":
    hybrid_retriever = HybridRetriever()
    
    # Test retrieval with a sample query
    query = "explain hypertension in detail"
    results = hybrid_retriever.retrieve(query, top_k=5)
    
    print("\nTop hybrid retrieved documents:")
    for i, result in enumerate(results):
        print(f"Result {i+1}, Score: {result['score']:.4f}, Source: {', '.join(result['source'])}")
        print(f"Text: {result['text'][:150]}...\n")
        
    # Count statistics
    sources = {
        'lexical_only': 0,
        'semantic_only': 0,
        'both': 0
    }
    
    for result in results:
        if len(result['source']) == 1:
            if 'lexical' in result['source']:
                sources['lexical_only'] += 1
            else:
                sources['semantic_only'] += 1
        else:
            sources['both'] += 1
            
    print("\nSource Statistics:")
    print(f"Lexical only: {sources['lexical_only']}")
    print(f"Semantic only: {sources['semantic_only']}")
    print(f"Both sources: {sources['both']}")
    
    # # If you want to see the raw results from each retriever separately
    # print("\nTesting individual retrievers...")
    
    # print("\nLexical (BM25) Results:")
    # lexical_only = hybrid_retriever.lexical_retriever.retrieve(query, top_k=3)
    # for i, result in enumerate(lexical_only):
    #     print(f"Result {i+1}, Score: {result['score']:.4f}")
    #     print(f"Text: {result['text'][:150]}...\n")
        
    # print("\nSemantic (Embeddings) Results:")
    # semantic_only = hybrid_retriever.semantic_retriever.get_relevant_documents(query, k=3)
    # for i, doc in enumerate(semantic_only):
    #     print(f"Result {i+1}")
    #     print(f"Text: {doc.page_content[:150]}...\n")