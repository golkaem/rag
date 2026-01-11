import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


def generate_embeddings():
    chunks_dir = Path("data/chunks")
    index_dir = Path("data/index")
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"
    BATCH_SIZE = 512

    index = None
    if faiss_index_path.exists():
        print("[INFO] Loading existing FAISS index")
        index = faiss.read_index(str(faiss_index_path))

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []

    processed_chunks = set(m["chunk_id"] for m in all_metadata)
    print(f"[INFO] Already processed chunks: {len(processed_chunks)}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    texts_batch = []
    metadata_batch = []
    for chunk_file in tqdm(list(chunks_dir.glob("*.json")), desc="Files"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk_file.stem}_{i}"

            if chunk_id in processed_chunks:
                continue

            text = chunk.get("text", "").strip()
            if not text:
                continue

            texts_batch.append(text)
            metadata_batch.append(
                {
                    "chunk_id": chunk_id,
                    "file": chunk_file.name,
                    "page": chunk["page"],
                    "text": text,
                }
            )

            if len(texts_batch) >= BATCH_SIZE:
                embeddings = model.encode(
                    texts_batch, convert_to_numpy=True, normalize_embeddings=True
                )

                if index is None:
                    dim = embeddings.shape[1]
                    index = faiss.IndexFlatIP(dim)

                index.add(embeddings)
                all_metadata.extend(metadata_batch)

                faiss.write_index(index, str(faiss_index_path))
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

                texts_batch.clear()
                metadata_batch.clear()

    if texts_batch:
        embeddings = model.encode(
            texts_batch, convert_to_numpy=True, normalize_embeddings=True
        )

        if index is None:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)

        index.add(embeddings)
        all_metadata.extend(metadata_batch)

        faiss.write_index(index, str(faiss_index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Total indexed chunks: {index.ntotal}")
