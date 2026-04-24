"""
Unit tests for embedding and retrieval components.

Tests focus on:
  - Embedding output shapes and normalization
  - FAISS index building and searching
  - Retrieval correctness
  - Context formatting
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingModel, FAISSIndex
from src.ingest import Document, chunk_text
from src.retriever import Retriever, format_context


# ============================================================================
# Embedding Model Tests
# ============================================================================

class TestEmbeddingModel:
    """Test suite for EmbeddingModel class."""

    @pytest.fixture
    def embedder(self):
        """Create embedding model instance for tests."""
        return EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_embedding_model_output_shape(self, embedder):
        """Test that embedding multiple texts produces correct shape."""
        texts = ["Diabetes symptoms", "Heart disease", "Pneumonia treatment"]
        embeddings = embedder.embed(texts, show_progress=False)

        # Shape should be (3, 384) for all-MiniLM-L6-v2
        assert embeddings.shape == (3, 384), \
            f"Expected shape (3, 384), got {embeddings.shape}"
        assert embeddings.dtype == np.float32, \
            f"Expected dtype float32, got {embeddings.dtype}"

    def test_embed_query_output_shape(self, embedder):
        """Test that embedding single query produces correct shape."""
        query = "What is diabetes?"
        embedding = embedder.embed_query(query)

        # Shape should be (384,) for all-MiniLM-L6-v2
        assert embedding.shape == (384,), \
            f"Expected shape (384,), got {embedding.shape}"
        assert embedding.dtype == np.float32

    def test_embeddings_are_normalized(self, embedder):
        """Test that embeddings are L2-normalized."""
        texts = ["Test 1", "Test 2", "Test 3"]
        embeddings = embedder.embed(texts, show_progress=False)

        # L2 norm of each embedding should be ~1.0 (L2 normalized)
        norms = np.linalg.norm(embeddings, axis=1)

        for i, norm in enumerate(norms):
            assert np.isclose(norm, 1.0, atol=1e-5), \
                f"Embedding {i} has norm {norm}, expected ~1.0"

    def test_query_embedding_normalized(self, embedder):
        """Test that query embedding is normalized."""
        query = "What are the symptoms?"
        embedding = embedder.embed_query(query)

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5), \
            f"Query embedding norm is {norm}, expected ~1.0"


# ============================================================================
# FAISS Index Tests
# ============================================================================

class TestFAISSIndex:
    """Test suite for FAISSIndex class."""

    @pytest.fixture
    def dummy_documents(self):
        """Create dummy documents for testing."""
        docs = [
            Document(
                id=f"doc_{i}",
                content=f"Document content {i}: medical topic about disease {i}",
                metadata={"source": "test", "topic": f"Disease {i}"}
            )
            for i in range(5)
        ]
        return docs

    @pytest.fixture
    def dummy_embeddings(self):
        """Create dummy normalized embeddings."""
        np.random.seed(42)
        # Create random embeddings
        embeddings = np.random.randn(5, 384).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings

    def test_faiss_index_init(self):
        """Test FAISS index initialization."""
        index = FAISSIndex(embedding_dim=384)

        assert index.embedding_dim == 384
        assert index.index is not None
        assert len(index.documents) == 0

    def test_faiss_index_build_and_search(self, dummy_embeddings, dummy_documents):
        """Test building index and searching."""
        index = FAISSIndex(embedding_dim=384)
        index.build(dummy_embeddings, dummy_documents)

        # Index should have 5 documents
        assert len(index.documents) == 5

        # Search for top-3 similar to first embedding
        query = dummy_embeddings[0]
        results = index.search(query, k=3)

        # Should return exactly 3 results
        assert len(results) == 3

        # Each result should be (Document, score) tuple
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0 <= score <= 1, f"Score {score} outside [0, 1]"

    def test_faiss_index_search_returns_ordered_results(self, dummy_embeddings, dummy_documents):
        """Test that search returns results ordered by similarity."""
        index = FAISSIndex(embedding_dim=384)
        index.build(dummy_embeddings, dummy_documents)

        query = dummy_embeddings[0]
        results = index.search(query, k=5)

        # Results should be ordered by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), \
            f"Results not sorted descending: {scores}"

    def test_faiss_index_k_limit(self, dummy_embeddings, dummy_documents):
        """Test that search respects k parameter."""
        index = FAISSIndex(embedding_dim=384)
        index.build(dummy_embeddings, dummy_documents)

        query = dummy_embeddings[0]

        # Test k=2
        results = index.search(query, k=2)
        assert len(results) == 2

        # Test k=5
        results = index.search(query, k=5)
        assert len(results) == 5

        # Test k=10 (more than available, should return all 5)
        results = index.search(query, k=10)
        assert len(results) == 5


# ============================================================================
# Retriever Tests
# ============================================================================

class TestRetriever:
    """Test suite for Retriever class."""

    @pytest.fixture
    def setup_retriever(self):
        """Set up retriever with dummy index."""
        # Create dummy documents and embeddings
        np.random.seed(42)
        docs = [
            Document(
                id=f"doc_{i}",
                content=f"Content about medical topic {i}",
                metadata={"source": "test", "topic": f"Topic {i}"}
            )
            for i in range(5)
        ]

        embeddings = np.random.randn(5, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Build index
        index = FAISSIndex(embedding_dim=384)
        index.build(embeddings, docs)

        # Create mock embedding model (just for retriever initialization)
        class MockEmbedder:
            embedding_dim = 384
            def embed_query(self, query):
                # Return a normalized random vector
                v = np.random.randn(384).astype(np.float32)
                return v / np.linalg.norm(v)
            def embed(self, texts, show_progress=False):
                vecs = np.random.randn(len(texts), 384).astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / norms

        embedder = MockEmbedder()

        # Create retriever
        retriever = Retriever(index, embedder, top_k=3)
        return retriever

    def test_retriever_returns_top_k(self, setup_retriever):
        """Test that retriever returns exactly top_k results."""
        retriever = setup_retriever

        # Should return 3 results (top_k=3)
        results = retriever.retrieve("diabetes symptoms")
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    def test_retriever_results_have_correct_structure(self, setup_retriever):
        """Test that retrieved chunks have correct attributes."""
        retriever = setup_retriever

        results = retriever.retrieve("medical question")

        for chunk in results:
            assert hasattr(chunk, 'document')
            assert hasattr(chunk, 'score')
            assert hasattr(chunk, 'rank')
            assert isinstance(chunk.score, float)
            assert isinstance(chunk.rank, int)
            assert 0 <= chunk.score <= 1

    def test_retriever_ranks_are_sequential(self, setup_retriever):
        """Test that ranks are 0, 1, 2, ..."""
        retriever = setup_retriever

        results = retriever.retrieve("medical question")
        ranks = [chunk.rank for chunk in results]

        assert ranks == list(range(len(results))), \
            f"Ranks not sequential: {ranks}"


# ============================================================================
# Context Formatting Tests
# ============================================================================

class TestFormatContext:
    """Test suite for context formatting."""

    def test_format_context_includes_sources(self):
        """Test that formatted context includes source information."""
        from src.retriever import RetrievedChunk

        doc = Document(
            id="test_doc",
            content="Type 2 diabetes is characterized by insulin resistance.",
            metadata={"source": "MedQuAD", "topic": "Diabetes"}
        )

        chunk = RetrievedChunk(document=doc, score=0.85, rank=0)
        context = format_context([chunk])

        # Context should include source and score
        assert "source:" in context.lower()
        assert "similarity:" in context.lower() or "0.85" in context
        assert "Type 2 diabetes" in context

    def test_format_context_empty_list(self):
        """Test formatting with empty chunk list."""
        context = format_context([])
        assert "No relevant documents" in context

    def test_format_context_multiple_chunks(self):
        """Test formatting multiple chunks."""
        from src.retriever import RetrievedChunk

        docs = [
            Document(
                id=f"doc_{i}",
                content=f"Medical content {i}",
                metadata={"source": f"Source{i}", "topic": f"Topic{i}"}
            )
            for i in range(3)
        ]

        chunks = [
            RetrievedChunk(document=doc, score=0.9 - i*0.05, rank=i)
            for i, doc in enumerate(docs)
        ]

        context = format_context(chunks)

        # Should contain all documents
        for i in range(3):
            assert f"Medical content {i}" in context


# ============================================================================
# Chunk Text Tests
# ============================================================================

class TestChunkText:
    """Test suite for text chunking."""

    def test_chunk_text_respects_size(self):
        """Test that chunks don't exceed specified size."""
        text = "A" * 2000  # 2000 characters
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        for chunk in chunks:
            assert len(chunk) <= 200, \
                f"Chunk size {len(chunk)} exceeds limit 200"

    def test_chunk_text_has_overlap(self):
        """Test that consecutive chunks have overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100  # 2600 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        # Check overlap between consecutive chunks
        assert len(chunks) > 1, "Expected multiple chunks"

        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Last 50 chars of current should match first 50 of next
            overlap = current[-50:]
            expected_start = next_chunk[:50]

            assert overlap == expected_start, \
                f"Chunk {i} and {i+1} don't have expected overlap"

    def test_chunk_text_no_data_loss(self):
        """Test that all text is preserved in chunks."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        # Reconstruct text from chunks (accounting for overlap)
        reconstructed = chunks[0]
        for i in range(1, len(chunks)):
            # Skip overlap, add rest
            reconstructed += chunks[i][50:]

        assert reconstructed == text, \
            "Text not perfectly preserved in chunks"

    def test_chunk_text_small_text(self):
        """Test chunking of text smaller than chunk_size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == text
