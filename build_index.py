import faiss
import numpy as np
from pathlib import Path

RETRIEVAL_DIR = Path("retrieval")

# Load embeddings built in Step 1
emb = np.load(RETRIEVAL_DIR / "embeddings.npy").astype("float32")
print(f"Loaded {emb.shape[0]} embeddings of dim {emb.shape[1]}")

# Build IndexFlatIP — cosine similarity (vectors already normalized)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

faiss.write_index(index, str(RETRIEVAL_DIR / "faiss.index"))
print(f"FAISS index saved -> retrieval/faiss.index")
print(f"Total vectors indexed: {index.ntotal}")