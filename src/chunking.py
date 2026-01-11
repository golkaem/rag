import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
)


def chunk_page(text: str, page: int):
    """
    Делит текст одной страницы на чанки.
    """
    if not text or not text.strip():
        return []

    chunks_text = splitter.split_text(text)

    return [
        {"page": page, "text": chunk.strip()} for chunk in chunks_text if chunk.strip()
    ]


def chunk_parsed_pdfs(input_dir="data/parsed", output_dir="data/chunks"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.json"):
        print(f"[INFO] Processing {file.name}")

        with open(file, "r", encoding="utf-8") as f:
            pages = json.load(f)

        all_chunks = []

        for page_obj in pages:
            page_number = page_obj["page"]
            text = page_obj["text"]

            page_chunks = chunk_page(text=text, page=page_number)

            all_chunks.extend(page_chunks)

        out_path = output_dir / file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"    Saved {len(all_chunks)} chunks")


if __name__ == "__main__":
    chunk_parsed_pdfs()
