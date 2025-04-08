from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.chunking import DocumentLoader
import os

class EmbeddingsHandler:
    def __init__(self, model_name="all-MiniLM-L6-v2", vector_store_path="embeddings/faiss_index"):
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def create_vector_store(self):
        docs = DocumentLoader().load_documents()
        vector_store = FAISS.from_documents(docs, self.embedding_model)
        vector_store.save_local(self.vector_store_path)
        print("Vector store created successfully!")

    def get_retriever(self):
        try:
            vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}. Creating a new one...")
            self.create_vector_store()
            vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)

        return vector_store.as_retriever(search_kwargs={"k": 3,"score_threshold":0.4})

