import sys
sys.path.append("D:/RAG Pipeline/RAG based chatbot/")
from search.hybrid_retriever import HybridRetriever
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class RAGPipeline:
    def __init__(self, llm_model="llama3.1"):
        self.llm_model = llm_model
        self.llm = OllamaLLM(model=self.llm_model)
        self.base_retriever = HybridRetriever().as_retriever()

        # Prompt for rewriting query with history
        history_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the chat history and the latest question, rewrite the question to be standalone and clear based on the conversation.  
            Chat History: {chat_history}  
            Latest Question: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # History-aware retriever
        self.retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.base_retriever,
            prompt=history_prompt
        )

        # Prompt for answering with context
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful chatbot that answers questions solely based on the provided context.  
            Provide detailed, step-by-step explanations, elaborating on each step for clarity.  
            If the answer isn’t in the context, say: "I do not have enough information to answer that in detail."  
            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.document_chain = create_stuff_documents_chain(self.llm, self.answer_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def get_response(self, query: str, chat_history: list = None) -> str:
        # Default to empty history if none provided
        if chat_history is None:
            chat_history = []
        
        # Convert session history to LangChain message format if needed
        formatted_history = [
            HumanMessage(content=msg["human"]) if "human" in msg else AIMessage(content=msg["ai"])
            for msg in chat_history
        ]

        # Invoke the chain with history and query
        response = self.retrieval_chain.invoke({
            "input": query,
            "chat_history": formatted_history
        })
        
        return response["answer"]

if __name__ == "__main__":
    print("Running inference...")
    rag_pipeline = RAGPipeline()
    chat_history = []

    # Query 1
    query1 = "explain hypertension"
    response1 = rag_pipeline.get_response(query1, chat_history=chat_history)
    chat_history.append({"human": query1, "ai": response1})
    print("Query 1:", query1)
    print("Query 1 Response:", response1)

    # Query 2
    query2 = "what causes it?"
    response2 = rag_pipeline.get_response(query2, chat_history=chat_history)
    chat_history.append({"human": query2, "ai": response2})
    print("Query 2:", query2)
    print("Query 2 Response:", response2)