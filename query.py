"""Retrieval + grounded generation for The Unofficial Guide.

- retrieve(query, k): semantic search over the ChromaDB collection.
- ask(query, k): retrieve context, then generate a grounded answer with Groq
  that only uses the retrieved chunks, with source attribution appended
  programmatically (not left to the model).

Run directly for a retrieval smoke test (no API key needed for retrieval):
    python query.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from embed_store import COLLECTION_NAME, embed_texts, get_client

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOP_K = 4
# Cosine distance above this is treated as too weak to be real context.
# Chunks beyond it are dropped before generation so the model can't lean on
# loosely-related noise. If nothing survives, we refuse.
MAX_DISTANCE = 0.9

REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are The Unofficial Guide, a question-answering assistant for student "
    "reviews of CUNY Hunter College professors. You must answer using ONLY the "
    "information in the CONTEXT block provided by the user. Follow these rules "
    "strictly:\n"
    "1. Base every claim on the CONTEXT. Do not use any outside or prior knowledge.\n"
    "2. If the CONTEXT does not contain enough information to answer the "
    f"question, reply with exactly: \"{REFUSAL}\" and nothing else.\n"
    "3. Do not invent professors, courses, ratings, or policies that are not in "
    "the CONTEXT.\n"
    "4. Summarize what students actually said. It is fine to note disagreement "
    "between reviewers.\n"
    "5. Keep the answer concise (2-5 sentences)."
)


@dataclass
class Retrieved:
    text: str
    source: str
    professor: str
    course: str
    distance: float


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list[Retrieved]:
    """Return the top-k most semantically similar chunks for a query."""
    client = get_client()
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = embed_texts([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    retrieved: list[Retrieved] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        retrieved.append(
            Retrieved(
                text=text,
                source=meta.get("source", "unknown"),
                professor=meta.get("professor", "unknown"),
                course=meta.get("course", "unknown"),
                distance=float(dist),
            )
        )
    return retrieved


def _format_context(chunks: list[Retrieved]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Document {i} | source: {c.source}]\n{c.text}")
    return "\n\n".join(blocks)


def ask(query: str, k: int = DEFAULT_TOP_K) -> dict:
    """End-to-end: retrieve context and generate a grounded, cited answer.

    Returns a dict with keys: answer, sources, chunks, distances.
    """
    retrieved = retrieve(query, k=k)
    # Keep only chunks that are actually close enough to be relevant context.
    relevant = [c for c in retrieved if c.distance <= MAX_DISTANCE]

    if not relevant:
        return {
            "answer": REFUSAL,
            "sources": [],
            "chunks": [c.text for c in retrieved],
            "distances": [c.distance for c in retrieved],
        }

    context = _format_context(relevant)
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above."
    )

    # Import here so retrieval-only use (python query.py) needs no API key.
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    # Source attribution is added from metadata, not trusted to the LLM.
    if answer == REFUSAL:
        sources: list[str] = []
    else:
        seen: list[str] = []
        for c in relevant:
            if c.source not in seen:
                seen.append(c.source)
        sources = seen

    return {
        "answer": answer,
        "sources": sources,
        "chunks": [c.text for c in relevant],
        "distances": [c.distance for c in relevant],
    }


if __name__ == "__main__":
    smoke_queries = [
        "What do students say about Professor Rivera's exams in CSCI 135?",
        "Is Professor Chen's MATH 155 a heavy workload?",
        "Do students recommend Professor Alvarez for ENGL 120?",
    ]
    for q in smoke_queries:
        print("=" * 80)
        print("QUERY:", q)
        for r in retrieve(q, k=DEFAULT_TOP_K):
            print(f"  [{r.distance:.3f}] {r.source}: {r.text[:90]}...")
        print()
