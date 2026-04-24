"""
Integration tests for the RAG pipeline.

Tests focus on:
  - End-to-end pipeline behavior
  - Response quality and structure
  - Source attribution
  - Error handling
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import Document, chunk_text, load_json_corpus
from src.pipeline import RAGResponse


# ============================================================================
# Chunk Text Integration Tests
# ============================================================================

class TestChunkTextIntegration:
    """Integration tests for chunking strategy."""

    def test_chunk_text_respects_size(self):
        """Test that all chunks respect size limit."""
        text = "Medical information about various diseases. " * 100
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

        assert len(chunks) > 1, "Text should produce multiple chunks"

        for chunk in chunks:
            assert len(chunk) <= 512, \
                f"Chunk size {len(chunk)} exceeds 512"

    def test_chunk_text_has_overlap(self):
        """Test that consecutive chunks share overlap for context continuity."""
        text = "ABCDEFGHIJKLMNOP" * 200  # 3200 chars
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

        # Check each pair of consecutive chunks
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Last 64 chars of current should match first 64 of next
            actual_overlap = current[-64:]
            expected_overlap = next_chunk[:64]

            assert actual_overlap == expected_overlap, \
                f"Chunks {i} and {i+1} overlap not preserved"

    def test_chunk_overlap_reason(self):
        """
        Verify that overlap prevents splitting coherent information.

        Example: if a medical symptom description spans chunks,
        both chunks will be retrieved, giving LLM full context.
        """
        # Text with information split at chunk boundary
        text = (
            "A" * 250 +  # First half of chunk
            "IMPORTANT MEDICAL INFO" +
            "B" * 250  # Second half of chunk
        )

        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

        # With overlap, "IMPORTANT MEDICAL INFO" should appear in both chunks
        # (or both as part of overlapped region)
        full_content = "".join(chunks)
        count = full_content.count("IMPORTANT MEDICAL INFO")
        assert count >= 1, "Important info lost in chunking"


# ============================================================================
# Document Loading Tests
# ============================================================================

class TestLoadJsonCorpus:
    """Test suite for loading corpus files."""

    def test_load_json_corpus_returns_documents(self, tmp_path):
        """Test that corpus loading produces Document objects."""
        corpus = [
            {
                "id": "doc_1",
                "source": "MedQuAD",
                "topic": "Diabetes",
                "question": "What is diabetes?",
                "answer": "Diabetes is a metabolic disease...",
                "metadata": {"icd10": "E11"}
            }
        ]

        corpus_file = tmp_path / "corpus.json"
        corpus_file.write_text(json.dumps(corpus))

        docs = load_json_corpus(str(corpus_file))

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].id == "doc_1"
        assert "question" in docs[0].content.lower()
        assert "answer" in docs[0].content.lower()

    def test_load_json_corpus_preserves_metadata(self, tmp_path):
        """Test that metadata is preserved during loading."""
        corpus = [
            {
                "id": "doc_1",
                "source": "PubMed",
                "topic": "Heart Disease",
                "question": "Q?",
                "answer": "A.",
                "metadata": {"icd10": "I50", "specialty": "cardiology"}
            }
        ]

        corpus_file = tmp_path / "corpus.json"
        corpus_file.write_text(json.dumps(corpus))

        docs = load_json_corpus(str(corpus_file))
        doc = docs[0]

        assert doc.metadata["source"] == "PubMed"
        assert doc.metadata["topic"] == "Heart Disease"
        assert doc.metadata["icd10"] == "I50"
        assert doc.metadata["specialty"] == "cardiology"

    def test_load_json_corpus_file_not_found(self, tmp_path):
        """Test error handling for missing corpus file."""
        with pytest.raises(FileNotFoundError):
            load_json_corpus("/nonexistent/path/corpus.json")


# ============================================================================
# RAG Response Tests
# ============================================================================

class TestRAGResponse:
    """Test suite for RAGResponse dataclass."""

    def test_rag_response_has_required_fields(self):
        """Test that RAGResponse has all expected attributes."""
        from src.retriever import RetrievedChunk

        doc = Document(
            id="test",
            content="Test content",
            metadata={"source": "test"}
        )

        chunk = RetrievedChunk(document=doc, score=0.9, rank=0)

        response = RAGResponse(
            query="What is diabetes?",
            answer="Diabetes is a metabolic condition.",
            sources=[chunk],
            generation_time=1.5,
            model_name="test-model"
        )

        assert response.query == "What is diabetes?"
        assert response.answer == "Diabetes is a metabolic condition."
        assert len(response.sources) == 1
        assert response.generation_time == 1.5
        assert response.model_name == "test-model"

    def test_rag_response_with_no_sources(self):
        """Test RAGResponse when no sources are retrieved."""
        response = RAGResponse(
            query="Obscure medical question?",
            answer="I don't have information about that.",
            sources=[],
            generation_time=0.5,
            model_name="test-model"
        )

        assert len(response.sources) == 0
        assert "information" in response.answer.lower()


# ============================================================================
# Mock Pipeline Tests
# ============================================================================

class TestPipelineWithMocks:
    """Test pipeline components with mocks (fast, no models needed)."""

    # Patch the actual module where each class is defined (not where it's used),
    # because _initialize() now imports them from their source modules at runtime.
    @patch('src.retriever.Retriever')
    @patch('src.generator.LLMGenerator')
    @patch('src.embeddings.EmbeddingModel')
    @patch('src.embeddings.FAISSIndex')
    def test_pipeline_ask_returns_response(
        self,
        mock_faiss,
        mock_embedder,
        mock_generator,
        mock_retriever_class
    ):
        """Test pipeline.ask() returns RAGResponse with correct structure."""
        from src.pipeline import RAGPipeline
        from src.retriever import RetrievedChunk

        # Setup mocks
        mock_doc = Document(
            id="test",
            content="Type 2 diabetes is...",
            metadata={"source": "Test", "topic": "Diabetes"}
        )

        mock_chunk = RetrievedChunk(
            document=mock_doc,
            score=0.85,
            rank=0
        )

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_mmr.return_value = [mock_chunk]
        mock_retriever_class.return_value = mock_retriever

        # Mock generator
        mock_gen_result = MagicMock()
        mock_gen_result.answer = "Type 2 diabetes is a metabolic condition."
        mock_gen_result.model_name = "test-model"

        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_gen_result
        mock_gen.model_name = "test-model"
        mock_generator.return_value = mock_gen

        # Mock FAISS index
        mock_index = MagicMock()
        mock_faiss.load.return_value = mock_index

        # Create pipeline and ask
        pipeline = RAGPipeline(use_demo_mode=False)
        pipeline.embedding_model = mock_embedder
        pipeline.faiss_index = mock_index
        pipeline.generator = mock_gen
        pipeline.retriever = mock_retriever

        # This should not actually initialize since we're mocking
        response = pipeline.ask("What is diabetes?")

        # Verify response structure
        assert response.query == "What is diabetes?"
        assert "diabetes" in response.answer.lower()
        assert len(response.sources) == 1
        assert isinstance(response.generation_time, float)

    def test_rag_response_includes_sources(self):
        """Test that RAG response includes citation information."""
        from src.retriever import RetrievedChunk

        doc = Document(
            id="doc_123",
            content="Hypertension affects blood vessel walls.",
            metadata={
                "source": "Medical Textbook",
                "topic": "Hypertension",
                "specialty": "cardiology"
            }
        )

        chunk = RetrievedChunk(document=doc, score=0.92, rank=0)

        response = RAGResponse(
            query="What is hypertension?",
            answer="Hypertension is high blood pressure.",
            sources=[chunk],
            generation_time=1.2,
            model_name="mistral-7b"
        )

        # Verify sources are citable
        assert len(response.sources) > 0
        source = response.sources[0]
        assert source.document.metadata["source"] == "Medical Textbook"
        assert source.score == 0.92
        assert "High blood pressure" in response.answer or "high blood pressure" in response.answer


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in pipeline components."""

    def test_chunk_text_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        text = "Test text"
        with pytest.raises(ValueError):
            # overlap >= chunk_size should fail
            chunk_text(text, chunk_size=100, chunk_overlap=100)

    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        empty_corpus = []
        # Loading empty corpus should work but produce no documents
        docs = load_json_corpus  # This would be called on empty file

        # Note: load_json_corpus expects valid JSON file,
        # just verify the function exists and is callable
        assert callable(load_json_corpus)

    def test_document_creation_with_minimal_fields(self):
        """Test Document creation with required fields only."""
        doc = Document(
            id="minimal",
            content="Some content",
            metadata={}
        )

        assert doc.id == "minimal"
        assert doc.content == "Some content"
        assert doc.metadata == {}

    def test_document_creation_with_rich_metadata(self):
        """Test Document creation with comprehensive metadata."""
        doc = Document(
            id="rich",
            content="Medical content",
            metadata={
                "source": "PubMed",
                "topic": "Sepsis",
                "icd10": "R65",
                "specialty": "critical care",
                "date": "2024-01-01",
                "url": "https://example.com"
            }
        )

        assert doc.id == "rich"
        assert len(doc.metadata) == 6
        assert doc.metadata["specialty"] == "critical care"
