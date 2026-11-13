"""RAG: retrieve the most relevant chunks from the kb/ folder.

Uses Gemini embeddings + cosine similarity (numpy only, so it stays light on
Render's free 512 MB instance). If embeddings fail (quota / network), it falls
back to simple keyword matching so the demo never breaks.
"""
import os
import re
import threading
from pathlib import Path

import numpy as np

KB_DIR = Path(__file__).parent / "kb"
EMBED_MODEL = "gemini-embedding-001"

_lock = threading.Lock()
_chunks = None      # list of {"source", "title", "text"}
_vectors = None     # np.ndarray (n, d) normalised, or None


def _load_chunks():
    """Split every .md file in kb/ into chunks, one per '## ' section."""
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for part in re.split(r"(?m)^## ", text)[1:]:
            title, _, body = part.partition("\n")
            chunks.append({
                "source": path.name,
                "title": title.strip(),
                "text": f"{title.strip()}\n{body.strip()}",
            })
    return chunks


def _embed(texts, task_type):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    vecs = np.array([e.values for e in result.embeddings], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _ensure_index():
    global _chunks, _vectors
    with _lock:
        if _chunks is None:
            _chunks = _load_chunks()
        if _vectors is None:
            try:
                _vectors = _embed([c["text"] for c in _chunks], "RETRIEVAL_DOCUMENT")
            except Exception:
                _vectors = None  # keyword fallback; retried on the next query


def _keyword_scores(query):
    words = set(re.findall(r"\w+", query.lower()))
    scores = []
    for c in _chunks:
        cw = re.findall(r"\w+", c["text"].lower())
        hits = sum(1 for w in cw if w in words)
        scores.append(hits / (len(cw) ** 0.5 + 1))
    return np.array(scores, dtype=np.float32)


def search(query: str, top_k: int = 3):
    """Return the top_k chunks for a query as a list of dicts."""
    _ensure_index()
    method = "keyword"
    scores = None
    if _vectors is not None:
        try:
            q = _embed([query], "RETRIEVAL_QUERY")[0]
            scores = _vectors @ q
            method = "embeddings"
        except Exception:
            scores = None
    if scores is None:
        scores = _keyword_scores(query)
    order = np.argsort(-scores)[:top_k]
    return [
        {
            "source": _chunks[i]["source"],
            "title": _chunks[i]["title"],
            "text": _chunks[i]["text"],
            "score": round(float(scores[i]), 3),
            "method": method,
        }
        for i in order
        if scores[i] > 0
    ]