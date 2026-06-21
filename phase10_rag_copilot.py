"""
Phase 10 — RAG Design Copilot

A retrieval-augmented chat layer grounded in (a) a knowledge base of component
datasheets + EEE design rules and (b) the user's current design.  It answers
design questions and suggests refinements with CITED answers instead of one-shot
generation.

Advisory only: it reads the CircuitGraph READ-ONLY and never mutates it or any
pipeline state.  Any change is surfaced as a suggestion / revised English prompt
the user may choose to re-run through Phase 0.

Stack
-----
- Embeddings: sentence-transformers (local, no API key) — EMBEDDING_MODEL.
- Vector store: chromadb persistent collection at VECTOR_DB_PATH.
- Generation: the existing Groq client (GROQ_API_KEY, GROQ_MODEL).

Sections
--------
1. Config + shared embedder / Chroma collection
2. Chunking (used by ingestion)
3. Retriever
4. Design summary (read-only CircuitGraph -> text)
5. Grounded generation (Groq) — run_phase10 + stream_phase10
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent

COLLECTION_NAME = "eda_knowledge"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Config + shared embedder / Chroma collection
# ═══════════════════════════════════════════════════════════════════════════════

def embedding_model_name() -> str:
    """Embedding model id from .env (default all-MiniLM-L6-v2)."""
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def vector_db_path() -> Path:
    """Persistent Chroma directory from .env (default vectorstore/)."""
    p = os.getenv("VECTOR_DB_PATH", "vectorstore")
    path = Path(p)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def top_k() -> int:
    """Number of chunks to retrieve from .env (default 4)."""
    return int(os.getenv("RAG_TOP_K", "4"))


@lru_cache(maxsize=4)
def _embedding_function(model_name: str):
    """Cached Chroma SentenceTransformer embedding function (local model)."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_collection(db_path: Path | None = None, reset: bool = False):
    """Return the persistent Chroma collection (optionally reset first).

    Args:
        db_path: Override the vector-store directory (defaults to .env path).
        reset:   If True, delete any existing collection first (clean re-ingest).

    Returns:
        A chromadb Collection bound to the local embedding function.
    """
    import chromadb

    path = db_path or vector_db_path()
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    ef = _embedding_function(embedding_model_name())

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Split text into ~`size`-char chunks with `overlap`, on word boundaries.

    Args:
        text:    Document text.
        size:    Target chunk length in characters.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Prefer to break on whitespace near the end of the window.
        if end < n:
            ws = text.rfind(" ", start + size - overlap, end)
            if ws > start:
                end = ws
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Retriever
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, k: int | None = None, db_path: Path | None = None) -> list[dict]:
    """Similarity-search the knowledge collection for the top-k chunks.

    Args:
        query:   User question.
        k:       Number of chunks (defaults to RAG_TOP_K).
        db_path: Override vector-store directory (for tests).

    Returns:
        List of {text, source, chunk_index, distance}, best first (may be empty
        if the collection has not been ingested yet).
    """
    k = k or top_k()
    col = get_collection(db_path=db_path)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    docs   = res.get("documents", [[]])[0]
    metas  = res.get("metadatas", [[]])[0]
    dists  = res.get("distances", [[]])[0]
    out: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "text":        doc,
            "source":      (meta or {}).get("source", "unknown"),
            "chunk_index": (meta or {}).get("chunk_index", -1),
            "distance":    dist,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Design summary (read-only)
# ═══════════════════════════════════════════════════════════════════════════════

def build_design_summary(graph) -> str:
    """Build a compact, read-only text summary of a CircuitGraph.

    Includes components (id/type/name + key properties), the net list with
    membership, and basic counts.  Never mutates the graph.

    Args:
        graph: A CircuitGraph (or None).

    Returns:
        A human-readable summary string ("" if graph is None).
    """
    if graph is None:
        return ""

    lines: list[str] = []
    lines.append(f"Design name: {graph.metadata.name}")
    lines.append(f"Grid: {graph.metadata.width} x {graph.metadata.height} "
                 f"{graph.metadata.unit}")
    lines.append(f"Components ({len(graph.nodes)}):")
    for cid in sorted(graph.nodes.keys()):
        comp = graph.nodes[cid]
        props = comp.properties or {}
        prop_str = ", ".join(f"{k}={v}" for k, v in props.items()) or "—"
        lines.append(f"  - {cid} [{comp.comp_type}] \"{comp.name}\" "
                     f"at ({comp.x},{comp.y}); {prop_str}")

    # Net membership from edges.
    net_members: dict[str, set[str]] = {}
    net_types: dict[str, str] = {}
    for e in graph.edges:
        net_members.setdefault(e.net_id, set()).update([e.source[0], e.target[0]])
        net_types[e.net_id] = e.net_type
    lines.append(f"Nets ({len(net_members)}):")
    for nid in sorted(net_members):
        members = ", ".join(sorted(net_members[nid]))
        lines.append(f"  - {nid} [{net_types.get(nid, '?')}]: {members}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Grounded generation (Groq)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are the EDA Engine Design Copilot, an electronics design assistant.

GROUNDING RULES — follow without exception:
1. Answer ONLY using the KNOWLEDGE BASE CONTEXT and the CURRENT DESIGN SUMMARY
   provided below. Do not use outside knowledge.
2. Cite the sources you used by their filename, e.g. "(source: esp32_power.md)".
3. If the provided context does NOT contain the answer, reply that the
   information is "not in the knowledge base" and do not guess. NEVER fabricate
   electrical specifications, current figures, or part numbers.
4. You are advisory only. If you suggest a design change, present it as a
   suggestion or a revised prompt the user can re-run — never claim to have
   changed the design.
"""


def _make_client():
    """Create a Groq client from GROQ_API_KEY (patched out in tests)."""
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set (see .env.example).")
    return Groq(api_key=api_key)


def _model_name() -> str:
    """Groq model id from .env."""
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a source-tagged context block."""
    if not chunks:
        return "(no knowledge-base context retrieved)"
    parts = []
    for c in chunks:
        parts.append(f"[source: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)


def build_messages(query: str, chunks: list[dict], design_summary: str,
                   history: list | None) -> list[dict]:
    """Assemble the chat messages for grounded generation.

    Args:
        query:          User question.
        chunks:         Retrieved KB chunks.
        design_summary: Current-design summary ("" if none).
        history:        Prior turns [{role, content}, ...] or None.

    Returns:
        List of chat-message dicts for the Groq chat-completions API.
    """
    context = _format_context(chunks)
    user_block = (
        "KNOWLEDGE BASE CONTEXT:\n"
        f"{context}\n\n"
        "CURRENT DESIGN SUMMARY:\n"
        f"{design_summary or '(no design loaded)'}\n\n"
        f"USER QUESTION:\n{query}"
    )
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_block})
    return messages


def _extract_citations(answer: str, chunks: list[dict]) -> list[str]:
    """Citations = retrieved sources whose filename appears in the answer.

    Falls back to all unique retrieved sources when the answer cites none but is
    not a "not in the knowledge base" refusal.
    """
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    used = [s for s in sources if s in answer]
    if used:
        return used
    if "not in the knowledge base" in answer.lower():
        return []
    return sources


def run_phase10(query: str, circuit_graph=None, history: list | None = None) -> dict:
    """Phase 10 entry-point: grounded, cited answer to a design question.

    Args:
        query:         User question (non-empty).
        circuit_graph: Optional CircuitGraph for design-aware answers (read-only).
        history:       Optional prior turns for multi-turn refinement.

    Returns:
        {"answer": str, "citations": list[str], "retrieved_chunks": list[dict]}.

    Raises:
        ValueError: If query is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    chunks  = retrieve(query)
    summary = build_design_summary(circuit_graph)
    messages = build_messages(query, chunks, summary, history)

    client = _make_client()
    resp = client.chat.completions.create(
        model=_model_name(), messages=messages,
        temperature=0.2, max_tokens=1024,
    )
    answer = resp.choices[0].message.content or ""
    citations = _extract_citations(answer, chunks)

    return {
        "answer":           answer,
        "citations":        citations,
        "retrieved_chunks": chunks,
    }


def stream_phase10(query: str, circuit_graph=None, history: list | None = None):
    """Yield answer tokens as they stream from Groq, then a final summary.

    Yields:
        ("token", str) for each content delta, then
        ("done", {"answer", "citations", "retrieved_chunks"}).
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    chunks  = retrieve(query)
    summary = build_design_summary(circuit_graph)
    messages = build_messages(query, chunks, summary, history)

    client = _make_client()
    stream = client.chat.completions.create(
        model=_model_name(), messages=messages,
        temperature=0.2, max_tokens=1024, stream=True,
    )
    answer_parts: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            answer_parts.append(delta)
            yield ("token", delta)
    answer = "".join(answer_parts)
    yield ("done", {
        "answer":           answer,
        "citations":        _extract_citations(answer, chunks),
        "retrieved_chunks": chunks,
    })


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Is my decoupling correct for an ESP32?"
    result = run_phase10(q)
    print("\n=== ANSWER ===\n", result["answer"])
    print("\n=== CITATIONS ===\n", result["citations"])
    print("\n=== RETRIEVED ===")
    for c in result["retrieved_chunks"]:
        print(f"  - {c['source']} (chunk {c['chunk_index']}, dist {c['distance']:.3f})")
