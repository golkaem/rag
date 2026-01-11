import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


index_path = Path("data/index/faiss.index")
meta_path = Path("data/index/metadata.json")


class EmbeddingRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(index_path))
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str):
        faiss_top_k = 20
        final_top_k = 6

        q_emb = self.embedder.encode([query], normalize_embeddings=True)

        _, indices = self.index.search(np.asarray(q_emb, dtype="float32"), faiss_top_k)
        indices = indices[0].tolist()
        final_indices = indices[:final_top_k]

        return [self.metadata[idx] for idx in final_indices]
