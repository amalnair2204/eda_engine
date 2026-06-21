"""Tests for Phase 10 — RAG Design Copilot.

The Groq client is mocked everywhere, so these tests are deterministic and need
no live API key.  Embeddings (sentence-transformers) and the Chroma store run
for real against an isolated temp vector store per test.
One dedicated test per behaviour (no bundling).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

import phase10_rag_copilot as rag
from ingest_knowledge import ingest
from phase1_eda_engine import NetlistParser, InitialPlacer, CircuitGraph

client = TestClient(app, raise_server_exceptions=False)

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_groq(content: str, capture: dict | None = None) -> MagicMock:
    """A mock Groq client whose chat.completions.create returns `content`."""
    cl = MagicMock()

    def _create(model, messages, **kw):
        if capture is not None:
            capture["messages"] = messages
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    cl.chat.completions.create.side_effect = _create
    return cl


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """Isolated knowledge dir + temp vector store, ingested and ready."""
    kn = tmp_path / "knowledge"
    kn.mkdir()
    (kn / "zorb.md").write_text(
        "# Zorb Module\nThe ZorbConverter regulator outputs 7.77 volts and "
        "draws 42 mA in active mode. Deep-sleep current is 3 nA.\n",
        encoding="utf-8",
    )
    (kn / "widget.txt").write_text(
        "The Flubber widget is rated for 99 degrees and uses a wibble bus.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "vs"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    rag.get_collection.cache_clear() if hasattr(rag.get_collection, "cache_clear") else None
    total = ingest(knowledge_dir=kn)
    assert total > 0
    return kn


def _sample_graph() -> CircuitGraph:
    raw = json.loads(_SAMPLE_JSON.read_text(encoding="utf-8"))
    nl = NetlistParser().parse(raw)
    InitialPlacer(nl.metadata).place(nl)
    return CircuitGraph.from_netlist(nl)


# ---------------------------------------------------------------------------
# test_ingest_builds_persistent_collection
# ---------------------------------------------------------------------------

def test_ingest_builds_persistent_collection(kb) -> None:
    """Ingestion builds a populated, persistent collection."""
    col = rag.get_collection()
    assert col.count() > 0
    # Re-open via a fresh collection handle — persistence holds.
    assert rag.get_collection().count() == col.count()


# ---------------------------------------------------------------------------
# test_ingest_no_duplicate_on_rerun
# ---------------------------------------------------------------------------

def test_ingest_no_duplicate_on_rerun(kb) -> None:
    """Re-running ingestion refreshes without duplicating chunks."""
    first = rag.get_collection().count()
    again = ingest(knowledge_dir=kb)
    assert again == first


# ---------------------------------------------------------------------------
# test_retrieval_returns_seeded_source
# ---------------------------------------------------------------------------

def test_retrieval_returns_seeded_source(kb) -> None:
    """A query about a seeded doc returns its source in the top-k."""
    chunks = rag.retrieve("What voltage does the ZorbConverter output?")
    assert chunks, "expected retrieved chunks"
    assert "zorb.md" in {c["source"] for c in chunks}


# ---------------------------------------------------------------------------
# test_run_phase10_shape_and_citations
# ---------------------------------------------------------------------------

def test_run_phase10_shape_and_citations(kb) -> None:
    """run_phase10 returns the right shape and cites a retrieved source."""
    answer = "The ZorbConverter outputs 7.77 V (source: zorb.md)."
    with patch.object(rag, "_make_client", return_value=_mock_groq(answer)):
        out = rag.run_phase10("What voltage does the ZorbConverter output?")
    assert set(out.keys()) == {"answer", "citations", "retrieved_chunks"}
    assert "zorb.md" in out["citations"]
    assert "zorb.md" in {c["source"] for c in out["retrieved_chunks"]}


# ---------------------------------------------------------------------------
# test_grounding_refusal_out_of_corpus
# ---------------------------------------------------------------------------

def test_grounding_refusal_out_of_corpus(kb) -> None:
    """An out-of-corpus question yields a grounded refusal, no citations."""
    refusal = "That information is not in the knowledge base."
    with patch.object(rag, "_make_client", return_value=_mock_groq(refusal)):
        out = rag.run_phase10("What is the airspeed velocity of a swallow?")
    assert "not in the knowledge base" in out["answer"].lower()
    assert out["citations"] == []


# ---------------------------------------------------------------------------
# test_design_summary_in_prompt_context
# ---------------------------------------------------------------------------

def test_design_summary_in_prompt_context(kb) -> None:
    """A supplied CircuitGraph's component refs appear in the prompt context."""
    graph = _sample_graph()
    # Direct summary check.
    summary = rag.build_design_summary(graph)
    for cid in graph.nodes:
        assert cid in summary

    # End-to-end: refs reach the messages sent to Groq.
    capture: dict = {}
    with patch.object(rag, "_make_client",
                      return_value=_mock_groq("ok (source: esp32_power.md)", capture)):
        rag.run_phase10("What draws the most power here?", circuit_graph=graph)
    joined = "\n".join(m["content"] for m in capture["messages"])
    assert "U1" in joined and "ESP32" in joined


# ---------------------------------------------------------------------------
# test_app_copilot_returns_answer_and_citations
# ---------------------------------------------------------------------------

def test_app_copilot_returns_answer_and_citations(kb) -> None:
    """POST /copilot returns answer + citations (Groq mocked)."""
    answer = "Decoupling caps go within 2 cells (source: decoupling_guidelines.md)."
    with patch.object(rag, "_make_client", return_value=_mock_groq(answer)):
        resp = client.post("/copilot", json={"query": "decoupling distance?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == answer
    assert "citations" in data and "sources" in data


# ---------------------------------------------------------------------------
# test_app_copilot_rejects_empty_query
# ---------------------------------------------------------------------------

def test_app_copilot_rejects_empty_query() -> None:
    """POST /copilot with an empty query returns HTTP 400."""
    resp = client.post("/copilot", json={"query": "   "})
    assert resp.status_code == 400
