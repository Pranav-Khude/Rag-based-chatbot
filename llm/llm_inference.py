import sys
sys.path.append("D:/RAG Pipeline/RAG based chatbot/")

from search.hybrid_retriever import HybridRetriever

from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.embeddings import EmbeddingsHandler

class RAGPipeline:
    def __init__(self, llm_model="llama3.1"):
        self.llm_model = llm_model
        self.retriever = EmbeddingsHandler().get_retriever()
        self.hybrid_retriever = HybridRetriever().as_retriever()
        self.llm = OllamaLLM(model=self.llm_model)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful chatbot that answers questions solely based on the provided context.  
            Provide detailed, step-by-step explanations, elaborating on each step for clarity.  
            Do not speculate beyond the context; if the answer isn’t present, say: "I do not have enough information to answer that in detail."  
            Think logically and break down your reasoning systematically.  

            Example:  
            Context: The moon orbits Earth and is covered in craters formed by meteor impacts.  
            Question: Why does the moon have craters?  
            Answer: Based on the context, here’s a detailed explanation:  
            1. The context states the moon is covered in craters.  
            2. It explains these craters are formed by meteor impacts, meaning collisions with space debris.  
            3. Since the moon orbits Earth, it’s exposed to such impacts over time.  
            4. Thus, the moon has craters because meteor impacts have shaped its surface.  

            Context: {context}  
            """),
            ("human", "{input}")
        ])
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.hybrid_retriever, self.document_chain)

    def get_response(self, query: str) -> str:
        response = self.retrieval_chain.invoke({"input": query})
        return response["answer"]

if __name__ == "__main__":
    print("Running inference...")
    rag_pipeline = RAGPipeline()
    query = "explain hypertension"
    print(rag_pipeline.get_response(query))


    