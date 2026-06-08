"""Embed chunks with all-MiniLM-L6-v2 and store them in ChromaDB.

This is the "Embedding + Vector Store" stage of the pipeline. It builds the
chunks from ingest.py, embeds them locally with sentence-transformers, and
upserts them into a persistent ChromaDB collection with source metadata so
retrieval can attribute answers later.

Run directly to (re)build the index:
    python embed_store.py
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import Chunk, build_chunks

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "hunter_professor_reviews"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def build_index(reset: bool = True) -> int:
    """Embed all chunks and (re)load them into ChromaDB. Returns chunk count."""
    chunks: list[Chunk] = build_chunks()
    if not chunks:
        raise RuntimeError("No chunks produced — check documents/ and ingest.py")

    if reset:
        # Start from a clean collection so re-runs don't duplicate ids.
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    client = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


if __name__ == "__main__":
    print(f"Embedding model: {EMBED_MODEL_NAME}")
    count = build_index(reset=True)
    client = get_client()
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Stored {count} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Collection now reports {collection.count()} items at {CHROMA_PATH}")
