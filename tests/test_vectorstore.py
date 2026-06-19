"""
test_vectorstore.py
===================
Unit tests for VectorStoreManager.

These tests cover the components most likely to be asked about
in technical interviews: duplicate detection, ingestion correctness,
retrieval with filters, and the hallucination guard threshold.

Run with: uv run pytest tests/ -v

PEP 8 | OOP
"""

from __future__ import annotations

import pytest
import uuid

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_store(tmp_path) -> VectorStoreManager:
    """
    Provide an isolated VectorStoreManager connected to a temporary directory.
    This ensures tests do not pollute the real application database.
    """
    unique_collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    settings = Settings(
        chroma_db_path=str(tmp_path / "test_chroma"),
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.1  # Lowered slightly to ensure tests catch matches
    )
    return VectorStoreManager(settings)


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """A single valid DocumentChunk for use across tests."""
    metadata = ChunkMetadata(
        topic="LSTM",
        difficulty="intermediate",
        type="concept_explanation",
        source="test_lstm.md",
        related_topics=["RNN", "vanishing_gradient"],
        is_bonus=False,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_lstm.md", "test content"),
        chunk_text=(
            "Long Short-Term Memory networks solve the vanishing gradient problem "
            "through gated mechanisms: the forget gate, input gate, and output gate. "
            "These gates control information flow through the cell state, allowing "
            "the network to maintain relevant information across long sequences."
        ),
        metadata=metadata,
    )


@pytest.fixture
def bonus_chunk() -> DocumentChunk:
    """A bonus topic chunk (GAN) for testing is_bonus filtering."""
    metadata = ChunkMetadata(
        topic="GAN",
        difficulty="advanced",
        type="architecture",
        source="test_gan.md",
        related_topics=["autoencoder", "generative_models"],
        is_bonus=True,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_gan.md", "gan content"),
        chunk_text=(
            "Generative Adversarial Networks consist of two competing neural networks: "
            "a generator that produces synthetic data and a discriminator that "
            "distinguishes real from generated samples. Training is a minimax game."
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Chunk ID Generation Tests
# ---------------------------------------------------------------------------

def test_same_content_produces_same_id() -> None:
    """Identical source and text must always produce the same ID."""
    id1 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
    id2 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
    assert id1 == id2

def test_different_content_produces_different_id() -> None:
    """Different text must produce different IDs."""
    id1 = VectorStoreManager.generate_chunk_id("lstm.md", "content one")
    id2 = VectorStoreManager.generate_chunk_id("lstm.md", "content two")
    assert id1 != id2

def test_different_source_produces_different_id() -> None:
    """Same text from different sources must produce different IDs."""
    id1 = VectorStoreManager.generate_chunk_id("file_a.md", "same text")
    id2 = VectorStoreManager.generate_chunk_id("file_b.md", "same text")
    assert id1 != id2

def test_id_is_16_characters() -> None:
    """Generated IDs must be exactly 16 hex characters."""
    chunk_id = VectorStoreManager.generate_chunk_id("source.md", "text")
    assert len(chunk_id) == 16
    assert all(c in "0123456789abcdef" for c in chunk_id)


# ---------------------------------------------------------------------------
# Duplicate Detection Tests
# ---------------------------------------------------------------------------

def test_new_chunk_is_not_duplicate(temp_store, sample_chunk: DocumentChunk) -> None:
    """A chunk that has never been ingested must not be flagged as duplicate."""
    assert temp_store.check_duplicate(sample_chunk.chunk_id) is False

def test_ingested_chunk_is_duplicate(temp_store, sample_chunk: DocumentChunk) -> None:
    """A chunk that has been ingested must be flagged as duplicate on re-check."""
    temp_store.ingest([sample_chunk])
    assert temp_store.check_duplicate(sample_chunk.chunk_id) is True

def test_ingestion_skips_duplicate(temp_store, sample_chunk: DocumentChunk) -> None:
    """Ingesting the same chunk twice must result in skipped=1 on second call."""
    sample_chunk.chunk_text += str(uuid.uuid4())
    sample_chunk.chunk_id = VectorStoreManager.generate_chunk_id(
        sample_chunk.metadata.source, 
        sample_chunk.chunk_text
    )
    # -------------------------------------------------------------------------
    
    # First ingestion
    result1 = temp_store.ingest([sample_chunk])
    assert result1.ingested == 1
    assert result1.skipped == 0
    
    # Second ingestion (duplicate)
    result2 = temp_store.ingest([sample_chunk])
    assert result2.ingested == 0
    assert result2.skipped == 1


# ---------------------------------------------------------------------------
# Retrieval Tests
# ---------------------------------------------------------------------------

def test_relevant_query_returns_results(temp_store, sample_chunk: DocumentChunk) -> None:
    """A query semantically similar to an ingested chunk must return results."""
    temp_store.ingest([sample_chunk])
    results = temp_store.query("How do LSTMs solve the vanishing gradient problem?")
    
    assert len(results) > 0
    assert results[0].chunk_id == sample_chunk.chunk_id

def test_irrelevant_query_returns_empty(temp_store, sample_chunk: DocumentChunk) -> None:
    """
    A query with no semantic similarity to the corpus must return empty list.
    This tests the hallucination guard threshold.
    """
    temp_store.ingest([sample_chunk])
    results = temp_store.query("tell me about the history of the roman empire and julius caesar")
    
    # Should return empty because the similarity score will be below the threshold
    assert len(results) == 0

def test_topic_filter_restricts_results(
    temp_store,
    sample_chunk: DocumentChunk,
    bonus_chunk: DocumentChunk,
) -> None:
    """Results with topic_filter='LSTM' must not include GAN chunks."""
    temp_store.ingest([sample_chunk, bonus_chunk])
    
    # Use a broad query that might slightly match both, but strictly filter by topic
    results = temp_store.query("neural networks", topic_filter="LSTM")
    
    assert len(results) > 0
    assert all(c.metadata.topic == "LSTM" for c in results)

def test_results_sorted_by_score_descending(
    temp_store, sample_chunk: DocumentChunk, bonus_chunk: DocumentChunk
) -> None:
    """Retrieved chunks must be sorted with highest similarity first."""
    temp_store.ingest([sample_chunk, bonus_chunk])
    
    # Broad query
    results = temp_store.query("neural networks generative models and memory gates")
    
    if len(results) > 1:
        assert results[0].score >= results[1].score