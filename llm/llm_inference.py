# import sys
# sys.path.append("D:/RAG Pipeline/RAG based chatbot/")

from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.embeddings import get_retriever




def create_rag_chain():
    retriever = get_retriever()
    llm = OllamaLLM(model="llama3.1") 
    
    prompt = ChatPromptTemplate.from_messages([
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

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def get_response(query: str) -> str:
    chain = create_rag_chain()
    response = chain.invoke({"input": query})
    return response["answer"]

if __name__ == "__main__":
    print("Running inference...")
    query = "Tell me about the functionality of The Urinary System."
    print(get_response(query))
