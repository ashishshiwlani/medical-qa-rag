"""
End-to-end RAG pipeline orchestration.

Coordinates:
  - Loading/building FAISS index
  - Embedding models
  - Retrieval
  - Generation

Key design choices:
  - Lazy initialization: models loaded on first use (faster startup)
  - Demo mode: fallback to lightweight models on CPU
  - Single unified interface: ask(query) -> answer with sources
"""
# PEP 563: postpone annotation evaluation so TYPE_CHECKING-only symbols
# (RetrievedChunk, etc.) work as type hints without being imported at runtime.
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Iterator, TYPE_CHECKING
import json

# Lightweight imports that are always safe at module level
from src.ingest import load_json_corpus, IngestPipeline
# format_context is a pure string-formatting function with no heavy deps
from src.retriever import format_context

# TYPE_CHECKING-only imports so mypy/IDEs know the types without triggering
# the heavy sentence-transformers / torch / faiss downloads at import time.
# The actual runtime imports live inside _initialize() — called on first use.
if TYPE_CHECKING:
    from src.embeddings import EmbeddingModel, FAISSIndex
    from src.retriever import Retriever, RetrievedChunk
    from src.generator import LLMGenerator, DemoGenerator, GenerationResult


@dataclass
class RAGResponse:
    """
    Complete RAG response with sources.

    Attributes:
        query: User's original question
        answer: Generated answer from LLM
        sources: List of retrieved context documents
        generation_time: Time taken to generate answer (seconds)
        model_name: Name of LLM used
    """
    query: str
    answer: str
    sources: List[RetrievedChunk]
    generation_time: float
    model_name: str


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Orchestrates: embedding -> retrieval -> generation

    Attributes:
        index_dir: Directory containing FAISS index
        embedding_model_name: HuggingFace model for embeddings
        llm_model_name: HuggingFace model for generation
        top_k: Default number of documents to retrieve
        use_demo_mode: Whether to use lightweight models
        embedding_model: Loaded EmbeddingModel (lazy)
        faiss_index: Loaded FAISSIndex (lazy)
        generator: Loaded LLM generator (lazy)
        retriever: Retriever instance (lazy)
    """

    def __init__(
        self,
        index_dir: str = "faiss_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        top_k: int = 5,
        use_demo_mode: bool = False
    ) -> None:
        """
        Initialize RAG pipeline.

        Does NOT load models yet (lazy initialization for fast startup).
        Models loaded on first call to ask().

        Args:
            index_dir: Path to FAISS index directory
            embedding_model_name: Embedding model
            llm_model_name: LLM model
            top_k: Number of documents to retrieve
            use_demo_mode: Use lightweight models (CPU-friendly)

        Example:
            >>> pipeline = RAGPipeline(use_demo_mode=True)
            >>> response = pipeline.ask("What is diabetes?")
        """
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = top_k
        self.use_demo_mode = use_demo_mode

        # Lazy initialization
        self.embedding_model: Optional[EmbeddingModel] = None
        self.faiss_index: Optional[FAISSIndex] = None
        self.generator: Optional[GenerationResult] = None
        self.retriever: Optional[Retriever] = None

    def _initialize(self) -> None:
        """
        Lazy initialization: load models and index on first use.

        Steps:
          1. Check if FAISS index exists
          2. If not, build from sample corpus
          3. Load embedding model
          4. Load FAISS index
          5. Load LLM generator
          6. Initialize retriever

        This is called automatically by ask() before first query.
        """
        if self.embedding_model is not None:
            # Already initialized
            return

        # Deferred heavy imports — only executed at first ask() call so that
        # importing RAGResponse or instantiating the class doesn't require
        # torch / sentence-transformers / faiss to be installed.
        from src.embeddings import EmbeddingModel, FAISSIndex, build_and_save_index  # noqa: F811
        from src.retriever import Retriever, RetrievedChunk  # noqa: F811
        from src.generator import LLMGenerator, DemoGenerator, GenerationResult  # noqa: F811

        print("Initializing RAG Pipeline...")

        # Check if index exists, if not build from sample corpus
        index_path = os.path.join(self.index_dir, "index.faiss")
        docs_path = os.path.join(self.index_dir, "documents.pkl")

        if not os.path.exists(index_path):
            print(f"FAISS index not found at {self.index_dir}")
            print("Building index from sample corpus...")

            corpus_path = os.path.join(
                Path(__file__).parent.parent,
                "data",
                "sample_corpus.json"
            )

            if not os.path.exists(corpus_path):
                raise FileNotFoundError(
                    f"Sample corpus not found at {corpus_path}. "
                    f"Please create data/sample_corpus.json or provide index_dir."
                )

            build_and_save_index(
                corpus_path=corpus_path,
                index_dir=self.index_dir,
                model_name=self.embedding_model_name
            )

        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel(
            model_name=self.embedding_model_name
        )

        # Load FAISS index
        print("Loading FAISS index...")
        self.faiss_index = FAISSIndex.load(index_path, docs_path)

        # Load LLM generator
        print("Loading LLM generator...")
        if self.use_demo_mode:
            self.generator = DemoGenerator()
        else:
            self.generator = LLMGenerator(
                model_name=self.llm_model_name,
                max_new_tokens=512,
                use_4bit=True  # 4-bit quantization for Mistral
            )

        # Initialize retriever
        self.retriever = Retriever(
            faiss_index=self.faiss_index,
            embedding_model=self.embedding_model,
            top_k=self.top_k
        )

        print("✓ RAG Pipeline initialized")

    def ask(self, query: str) -> RAGResponse:
        """
        Answer a medical question using RAG.

        Process:
          1. Embed query
          2. Retrieve top-k relevant documents (MMR)
          3. Format context with citations
          4. Generate answer from context
          5. Return answer + sources

        Args:
            query: Medical question

        Returns:
            RAGResponse with answer and source documents

        Example:
            >>> response = pipeline.ask("What are diabetes symptoms?")
            >>> print(response.answer)
            >>> for chunk in response.sources:
            ...     print(f"Source: {chunk.document.metadata['topic']}")
        """
        # Lazy initialization on first call
        self._initialize()

        print(f"\nQuery: {query}")

        start_time = time.time()

        # Step 1: Retrieve relevant documents with MMR diversification
        print("Retrieving relevant documents...")
        retrieved = self.retriever.retrieve_with_mmr(query, lambda_mult=0.5)

        if not retrieved:
            # No relevant documents found
            return RAGResponse(
                query=query,
                answer="I don't have information about that in the available medical literature.",
                sources=[],
                generation_time=time.time() - start_time,
                model_name=self.generator.model_name
            )

        # Step 2: Format context with citations
        context = format_context(retrieved)

        # Step 3: Generate answer
        print("Generating answer...")
        generation_result = self.generator.generate(query, context)

        generation_time = time.time() - start_time

        # Step 4: Return complete response
        response = RAGResponse(
            query=query,
            answer=generation_result.answer,
            sources=retrieved,
            generation_time=generation_time,
            model_name=generation_result.model_name
        )

        print(f"✓ Generated in {generation_time:.2f}s")

        return response

    def ask_stream(self, query: str) -> Iterator[str]:
        """
        Generate answer as token stream (for streaming UI).

        Not fully implemented in this version (requires streaming LLM).
        Returns complete answer as single token for compatibility.

        Args:
            query: Medical question

        Yields:
            Answer tokens (currently: single yield of complete answer)
        """
        response = self.ask(query)
        yield response.answer


def build_index_from_corpus(corpus_path: str, index_dir: str) -> None:
    """
    Standalone function to build and save FAISS index.

    Useful for preprocessing/setup before running pipeline.

    Args:
        corpus_path: Path to corpus.json file
        index_dir: Directory to save index

    Example:
        >>> build_index_from_corpus(
        ...     "data/sample_corpus.json",
        ...     "faiss_index"
        ... )
    """
    print(f"Building index from {corpus_path}...")
    build_and_save_index(corpus_path, index_dir)
    print(f"✓ Index saved to {index_dir}")
