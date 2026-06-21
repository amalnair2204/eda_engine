"""
Phase 10 — Knowledge-base ingestion.

Reads documents from knowledge/ (.md, .txt, .pdf), plus the EEE Domain
Constraints section of CLAUDE.md, chunks them, embeds with sentence-transformers
and stores them in a persistent Chroma collection at VECTOR_DB_PATH.

Re-running performs a clean refresh (the collection is reset first), so chunks
are never duplicated.

Run:  python -m ingest_knowledge
"""

from __future__ import annotations

import sys
from pathlib import Path

from phase10_rag_copilot import (
    COLLECTION_NAME,
    chunk_text,
    get_collection,
    vector_db_path,
)

_PROJECT_ROOT = Path(__file__).parent
_KNOWLEDGE_DIR = _PROJECT_ROOT / "knowledge"
_CLAUDE_MD = _PROJECT_ROOT / "CLAUDE.md"


# ---------------------------------------------------------------------------
# Document readers
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    """Extract text from a PDF via pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_document(path: Path) -> str:
    """Read a .md / .txt / .pdf document into plain text."""
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def _extract_eee_constraints(claude_md: Path) -> str:
    """Pull the 'EEE Domain Constraints' section out of CLAUDE.md.

    Returns:
        The section text (heading + body), or "" if not found.
    """
    if not claude_md.exists():
        return ""
    text = claude_md.read_text(encoding="utf-8")
    marker = "## EEE Domain Constraints"
    start = text.find(marker)
    if start == -1:
        return ""
    rest = text[start + len(marker):]
    end = rest.find("\n## ")          # next top-level section
    body = rest if end == -1 else rest[:end]
    return f"EEE Domain Constraints (Engineering Rules)\n{body.strip()}"


# ---------------------------------------------------------------------------
# Collect documents
# ---------------------------------------------------------------------------

def collect_documents(knowledge_dir: Path = _KNOWLEDGE_DIR) -> list[tuple[str, str]]:
    """Gather (source_name, text) pairs from the knowledge dir + CLAUDE.md.

    Args:
        knowledge_dir: Directory of .md/.txt/.pdf documents.

    Returns:
        List of (source filename, document text).
    """
    docs: list[tuple[str, str]] = []
    if knowledge_dir.exists():
        for path in sorted(knowledge_dir.iterdir()):
            if path.suffix.lower() in (".md", ".txt", ".pdf") and path.is_file():
                try:
                    text = _read_document(path)
                except Exception as exc:               # skip unreadable docs
                    print(f"  [skip] {path.name}: {exc}")
                    continue
                if text.strip():
                    docs.append((path.name, text))

    eee = _extract_eee_constraints(_CLAUDE_MD)
    if eee:
        docs.append(("CLAUDE_EEE_Domain_Constraints.md", eee))
    return docs


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(knowledge_dir: Path = _KNOWLEDGE_DIR, db_path: Path | None = None) -> int:
    """Chunk + embed + store all knowledge documents in a fresh collection.

    Args:
        knowledge_dir: Directory of source documents.
        db_path:       Override the vector-store directory (for tests).

    Returns:
        Total number of chunks ingested.
    """
    docs = collect_documents(knowledge_dir)
    if not docs:
        print(f"[Ingest] No documents found in {knowledge_dir}")
        return 0

    col = get_collection(db_path=db_path, reset=True)   # clean refresh

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    for source, text in docs:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{source}::{i}")
            texts.append(chunk)
            metas.append({"source": source, "chunk_index": i})
        print(f"  [doc] {source}: {len(chunks)} chunk(s)")

    if ids:
        col.add(ids=ids, documents=texts, metadatas=metas)

    total = col.count()
    print(f"[Ingest] Stored {total} chunk(s) from {len(docs)} document(s) "
          f"into '{COLLECTION_NAME}' at {db_path or vector_db_path()}")
    return total


if __name__ == "__main__":
    n = ingest()
    sys.exit(0 if n > 0 else 1)
