# app/services/rag_service.py

import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------
# SETUP EMBEDDING MODEL & VECTORSTORE
# ---------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "vectorstore", "chroma"))
print("Chroma DB path:", CHROMA_PATH)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
COLLECTION_NAME = "gov_docs"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---------------------
# FILE READERS — FIXED
# ---------------------

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            if not text.strip():
                print(f"[RAG] TXT empty: {path}")
            else:
                print(f"[RAG] TXT extracted: {path}")
            return text
    except Exception as e:
        print(f"[RAG] TXT read error: {e}")
        return ""

def read_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber (most reliable)."""
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        if not text.strip():
            print(f"[RAG] PDF has NO extractable text: {path}")
            return ""

        print(f"[RAG] PDF extracted: {path}")
        return text

    except Exception as e:
        print(f"[RAG] PDF read error: {e}")
        return ""

def read_file(path: str) -> str:
    """Auto-select TXT or PDF reader."""
    if path.lower().endswith(".txt"):
        return read_txt(path)
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    return ""

# ---------------------
# TEXT CHUNKING
# ---------------------

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ---------------------
# INDEX DOCUMENTS
# ---------------------

def index_documents_from_folder(folder_path: str):
    """Index all PDFs and TXT files in folder into Chroma collection."""

    if not os.path.exists(folder_path):
        print(f"[RAG] Folder does not exist: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".txt", ".pdf"))]
    if not files:
        print(f"[RAG] No supported files in: {folder_path}")
        return

    docs, ids, metas = [], [], []

    for fn in files:
        full_path = os.path.join(folder_path, fn)
        text = read_file(full_path)

        if not text.strip():
            print(f"[RAG] No text extracted from: {fn}")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{fn}_chunk_{i}")
            docs.append(chunk)
            metas.append({"source": fn, "chunk": i})

    if docs:
        embeddings = embedder.encode(docs).tolist()
        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=ids,
            metadatas=metas
        )
        print(f"[RAG] Indexed {len(docs)} chunks from {folder_path}")
    else:
        print(f"[RAG] No chunks produced in: {folder_path}")

# ---------------------
# INDEX ALL DOMAINS
# ---------------------

def index_all_domains(base_path="app/data/domains"):
    if not os.path.exists(base_path):
        print("[RAG] No domain folder found.")
        return

    for domain in os.listdir(base_path):
        folder = os.path.join(base_path, domain)
        if os.path.isdir(folder):
            index_documents_from_folder(folder)

    print("[RAG] Full indexing completed.")

# ---------------------
# RETRIEVE
# ---------------------
def retrieve_relevant_docs(query: str, top_k: int = 3) -> list:
    """Return list of dicts with text and metadata."""
    if collection.count() == 0:
        return []

    query_emb = embedder.encode([query]).tolist()[0]

    res = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i in range(len(docs)):
        out.append({
            "id": f"{i}",
            "text": docs[i],
            "metadata": metas[i] if metas else {},
            "score": float(dists[i]) if dists else 0.0
        })
    return out
