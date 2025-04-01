from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

class DocumentLoader:
    def __init__(self, pdf_path="data/pdfs", text_path="data/documents", chunk_size=500, chunk_overlap=50):
        self.pdf_path = pdf_path
        self.text_path = text_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        documents = []
        pdf_loader = DirectoryLoader(self.pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
        
        text_loader = DirectoryLoader(self.text_path, glob="*.txt")
        documents.extend(text_loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_documents(documents)
