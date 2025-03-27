import faiss

# Load FAISS index
index_path = "../embeddings/faiss_index/index.faiss"
index = faiss.read_index(index_path)

# Check index details
print("Index Type:", type(index))  # Shows the index type (e.g., IndexFlatL2)
print("Number of Vectors:", index.ntotal)  # Number of stored embeddings
print("Embedding Dimension:", index.d)
