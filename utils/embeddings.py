from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from utils.chunking import load_documents
import os


def create_vector_store():
    docs = load_documents()
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store_path = "embeddings/faiss_index"
    
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(vector_store_path)

    # default index IndexFlatL2
    print(type(vector_store.index)) 
    print("Vector store created successfully!")

def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store_path = "embeddings/faiss_index"
    
    if not os.path.exists(vector_store_path):
        create_vector_store()
        
    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    create_vector_store()