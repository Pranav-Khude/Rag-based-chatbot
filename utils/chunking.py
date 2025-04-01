from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import json
import os

class DocumentLoader:
    def __init__(self, pdf_path="data/pdfs", text_path="data/documents", chunk_size=500, chunk_overlap=50, chunk_file="document_chunks.json"):
        self.pdf_path = pdf_path
        self.text_path = text_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_file = chunk_file

    def load_documents(self):
        # Check if chunks already exist
        if os.path.exists(self.chunk_file):
            with open(self.chunk_file, 'r') as f:
                chunks = json.load(f)
                print("Loaded pre-chunked documents from file.")
                return [self._dict_to_document(doc) for doc in chunks]

        documents = []

        # Loading PDFs
        pdf_loader = DirectoryLoader(self.pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())

        # Loading text files
        text_loader = DirectoryLoader(self.text_path, glob="*.txt")
        documents.extend(text_loader.load())

        # Chunking the documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(documents)

        # Convert documents to a serializable format
        serializable_chunks = [self._document_to_dict(chunk) for chunk in chunks]

        # Save chunks to a file
        with open(self.chunk_file, 'w') as f:
            json.dump(serializable_chunks, f)
            print("Chunks saved to file.")

        return chunks

    def _document_to_dict(self, document):
        """Convert a Document object to a dictionary that can be serialized."""
        return {
            "text": document.page_content,
            "metadata": document.metadata,
        }

    def _dict_to_document(self, doc_dict):
        """Convert a dictionary back to a Document object."""
        from langchain.schema import Document
        return Document(page_content=doc_dict["text"], metadata=doc_dict["metadata"])


if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} chunked documents.")
