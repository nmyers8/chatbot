from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

PDF_FOLDER = "pdfs"
INDEX_FILE = "pdf_index.faiss"
CHUNKS_FILE = "pdf_chunks.pkl"
TRACK_FILE = "indexed_files.pkl"

def split_text(text, size = 400):
    words = text.split()
    pieces = []

    for i in range(0, len(words), size):
        pieces.append(" ".join(words[i:i + size]))

    return pieces

def ingest_pdfs():
    if os.path.exists(INDEX_FILE): # faiss file already exists
        index = faiss.read_index(INDEX_FILE) 
    else: # build from scratch
        index = faiss.IndexFlatL2(384) # dimension

    if os.path.exists(CHUNKS_FILE): # chunks file already exists
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
    else: # build from scratch
        chunks = []

    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "rb") as f:
            indexed_files = pickle.load(f)
    else:
        indexed_files = set()

    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        if file in indexed_files: # file has already been vectorized
            print("Skipping:", file)
            continue

        print("Indexing:", file)

        path = os.path.join(PDF_FOLDER, file)
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text:
                continue
            
            page_chunks = split_text(text)

            # locally embed text into vector
            embeddings = model.encode(page_chunks, convert_to_numpy=True).astype("float32")
            index.add(embeddings)

            for chunk in page_chunks:
                chunks.append({"text": chunk, "file": file, "page": page_num + 1})

        indexed_files.add(file)

    # overwrite files and mark any new files as added
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    with open(TRACK_FILE, "wb") as f:
        pickle.dump(indexed_files, f)

    return index, chunks