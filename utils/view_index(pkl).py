import pickle

# Load metadata
pkl_path = "../embeddings/faiss_index/index.pkl"
with open(pkl_path, "rb") as f:
    metadata = pickle.load(f)

# Print metadata structure
print("Keys in Metadata:", metadata.keys())

# View first 5 document mappings
if "docstore" in metadata:
    print("First 5 documents:", list(metadata["docstore"].values())[:5])
