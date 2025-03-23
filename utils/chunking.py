from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

def load_documents():
    documents = []

    # # Load text documents
    # text_loader = TextLoader("data/medical_sample.txt")
    # documents.extend(text_loader.load())

    # Load PDF documents
    pdf_loader = DirectoryLoader("data/pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    # Load documents from a directory
    directory_loader = DirectoryLoader("data/documents", glob="*.txt")
    documents.extend(directory_loader.load())

    # Chunk the documents for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(documents)

    return chunked_docs
