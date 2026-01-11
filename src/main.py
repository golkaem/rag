from pdf_parser import parse_all_pdfs
from chunking import chunk_parsed_pdfs
from index import generate_embeddings
from answer import run_rag


print("Parsing PDFs")
parse_all_pdfs()

print("Chunking parsed PDFs")
chunk_parsed_pdfs()

print("Indexing")
generate_embeddings()

print("Creating submission file")
run_rag("submission_Shagimardanova_v0.json")
