import sys
sys.path.append("D:/RAG Pipeline/RAG based chatbot/")

from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.embeddings import EmbeddingsHandler

class RAGPipeline:
    def __init__(self, llm_model="llama3.1"):
        self.llm_model = llm_model
        self.retriever = EmbeddingsHandler().get_retriever()
        self.llm = OllamaLLM(model=self.llm_model)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You're a helpful chatbot who answers questions based on the provided context only.  
            If the answer to the question is not in the context, politely say that you do not have the answer.  
            Think step-by-step before responding.  
            
            Example:  
            Context: The sun is a star at the center of our solar system. It's composed primarily of hydrogen and helium.  
            Question: What is the sun made of?  
            
            Steps:  
            1. Identify the question: It asks about the sun’s composition.  
            2. Find relevant context: The sun is composed primarily of hydrogen and helium.  
            3. Provide a factual answer.  
            
            If context is missing, say: "I do not have enough information to answer that."  
            
            Context:  
            {context}  
            
            Now, answer the question logically:
            """),
            ("human", "{input}")
        ])
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def get_response(self, query: str) -> str:
        response = self.retrieval_chain.invoke({"input": query})
        return response["answer"]

if __name__ == "__main__":
    print("Running inference...")
    rag_pipeline = RAGPipeline()
    query = "What is Hypertension?"
    print(rag_pipeline.get_response(query))