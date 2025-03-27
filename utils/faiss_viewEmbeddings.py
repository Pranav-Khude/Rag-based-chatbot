import numpy as np
import faiss


# Load FAISS index
index_path = "../embeddings/faiss_index/index.faiss"
index = faiss.read_index(index_path)


# Get first 5 stored embeddings
num_vectors = min(5, index.ntotal)
vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])

print("Sample Vectors:")
print(vectors)
