"""
Embedding models and FAISS vector index management.

Handles:
  - Loading pre-trained sentence transformers for embedding
  - Building and searching FAISS indices for fast similarity retrieval
  - Persistence of indices to disk

Key concepts:
  - Normalized embeddings: L2 normalization enables cosine similarity via dot product
  - IndexFlatIP: Inner Product index works perfectly with normalized embeddings
    (dot product of normalized vectors = cosine similarity)
  - Why not IndexIVFFlat: We don't have millions of vectors, so exact search is fine
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore  # lazy — raise at use time, not import time

# sentence_transformers is heavy; import lazily inside _load_model() so that
# modules that only import Document or utility classes don't need it installed.
# This also lets test collection succeed on machines without a GPU / full deps.
from src.ingest import Document


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for efficient text embedding.

    Attributes:
        model_name: HuggingFace model identifier
        model: Loaded SentenceTransformer instance
        embedding_dim: Output embedding dimension
        device: GPU/CPU device string
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ) -> None:
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model ID
                Default is all-MiniLM-L6-v2: 384-dim, 82MB, fast
            device: Device to load on ('cuda', 'cpu', etc.)
                If None, auto-detects

        Example:
            >>> embedder = EmbeddingModel()
            >>> embeddings = embedder.embed(["Hello", "World"])
            >>> embeddings.shape
            (2, 384)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None

        self._load_model()

    def _load_model(self) -> None:
        """
        Load SentenceTransformer model from HuggingFace.

        Sets embedding_dim based on model's output size.
        Prints diagnostic info about model and device.
        """
        print(f"Loading embedding model: {self.model_name}")

        # Lazy import — only required when we actually load a model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install with: pip install sentence-transformers"
            ) from exc

        # Load model (auto-downloads on first use)
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Get embedding dimension by doing a dummy forward pass
        dummy_embedding = self.model.encode("test", convert_to_numpy=True)
        self.embedding_dim = len(dummy_embedding)

        device_info = self.model.device if hasattr(self.model, 'device') else "CPU"
        print(
            f"✓ Loaded {self.model_name} "
            f"(dim={self.embedding_dim}, device={device_info})"
        )

    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts to normalized vectors.

        Normalization: Each vector is L2-normalized so ||v|| = 1.
        This enables cosine similarity via dot product: cos(a, b) = a·b

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding (controls memory usage)
            show_progress: Show tqdm progress bar

        Returns:
            Array of shape (len(texts), embedding_dim), float32, L2-normalized
        """
        # Convert texts to embeddings (handles batching internally)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization applied here
        )

        # Ensure float32 for FAISS compatibility
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Same normalization as embed() to ensure compatibility with index.

        Args:
            query: Query string

        Returns:
            Array of shape (embedding_dim,), float32, L2-normalized
        """
        # Embed as single-item list, then extract
        embedding = self.embed([query], show_progress=False)
        return embedding[0]  # Return (dim,) array


class FAISSIndex:
    """
    FAISS vector index for fast similarity search.

    Uses IndexFlatIP (inner product) with normalized embeddings,
    which is equivalent to cosine similarity.

    Attributes:
        embedding_dim: Dimension of embeddings (must match model)
        index: FAISS index object
        documents: List of Document objects, parallel to index rows
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        """
        Initialize empty FAISS index.

        Args:
            embedding_dim: Expected embedding dimension
                Default 384 matches all-MiniLM-L6-v2
        """
        self.embedding_dim = embedding_dim
        if not _FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        # IndexFlatIP: exhaustive search with inner product metric
        # Why not IndexIVFFlat? Unnecessary for ~1000s of vectors
        # Why not Flat L2? Because we normalize vectors, so IP = cosine
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[Document] = []

    def build(
        self,
        embeddings: np.ndarray,
        documents: List[Document]
    ) -> None:
        """
        Build index from pre-computed embeddings.

        Args:
            embeddings: Array of shape (N, embedding_dim), float32, normalized
            documents: List of Document objects, len(documents) == embeddings.shape[0]

        Raises:
            ValueError: If dimensions don't match
        """
        if embeddings.shape[0] != len(documents):
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) "
                f"!= documents count ({len(documents)})"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) "
                f"!= index dimension ({self.embedding_dim})"
            )

        # Add all embeddings to index in one batch (faster than incremental adds)
        self.index.add(embeddings)
        self.documents = documents

        print(f"✓ Built FAISS index with {len(documents)} documents")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search index for top-k most similar documents.

        Args:
            query_embedding: Query vector of shape (embedding_dim,), normalized
            k: Number of results to return

        Returns:
            List of (Document, similarity_score) tuples, sorted by score (descending)
            Scores are in [0, 1] after L2 norm (dot product of unit vectors)
        """
        # Reshape query to (1, dim) for batch search
        query_batch = query_embedding.reshape(1, -1)

        # Search: returns (distances, indices)
        # distance here = inner product (cosine for normalized vectors)
        distances, indices = self.index.search(query_batch, k)

        # Flatten since we only have one query
        distances = distances[0]  # shape (k,)
        indices = indices[0]      # shape (k,)

        results = []
        for distance, idx in zip(distances, indices):
            # idx is -1 if fewer than k results available
            if idx >= 0:
                doc = self.documents[idx]
                score = float(distance)  # Inner product = cosine similarity [0, 1]
                results.append((doc, score))

        return results

    def save(self, index_path: str, docs_path: str) -> None:
        """
        Persist index and documents to disk.

        Args:
            index_path: Path to save FAISS index (.faiss file)
            docs_path: Path to save documents pickle (.pkl file)
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(docs_path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save documents list (parallel data structure)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"✓ Saved index to {index_path}")
        print(f"✓ Saved documents to {docs_path}")

    @classmethod
    def load(cls, index_path: str, docs_path: str) -> "FAISSIndex":
        """
        Load index and documents from disk.

        Args:
            index_path: Path to FAISS index file
            docs_path: Path to documents pickle file

        Returns:
            Loaded FAISSIndex instance

        Raises:
            FileNotFoundError: If either file doesn't exist
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents file not found: {docs_path}")

        # Load FAISS index
        index = faiss.read_index(index_path)

        # Load documents
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)

        # Reconstruct FAISSIndex object
        faiss_index = cls(embedding_dim=index.d)
        faiss_index.index = index
        faiss_index.documents = documents

        print(f"✓ Loaded index with {len(documents)} documents")
        return faiss_index


def build_and_save_index(
    corpus_path: str,
    index_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FAISSIndex:
    """
    End-to-end: load corpus, embed, build index, save to disk.

    Convenience function for one-shot index creation.

    Args:
        corpus_path: Path to JSON corpus file
        index_dir: Directory to save index and documents
        model_name: Embedding model to use

    Returns:
        Built and saved FAISSIndex instance

    Example:
        >>> index = build_and_save_index(
        ...     corpus_path="data/sample_corpus.json",
        ...     index_dir="faiss_index",
        ... )
    """
    from src.ingest import load_json_corpus

    print(f"Building index from {corpus_path}...")

    # Load documents
    documents = load_json_corpus(corpus_path)
    print(f"Loaded {len(documents)} documents")

    # Initialize embedding model
    embedder = EmbeddingModel(model_name=model_name)

    # Extract text content to embed
    texts = [doc.content for doc in documents]

    # Embed all documents
    print("Embedding documents...")
    embeddings = embedder.embed(texts, show_progress=True)

    # Build FAISS index
    faiss_index = FAISSIndex(embedding_dim=embedder.embedding_dim)
    faiss_index.build(embeddings, documents)

    # Save to disk
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index_path = os.path.join(index_dir, "index.faiss")
    docs_path = os.path.join(index_dir, "documents.pkl")

    faiss_index.save(index_path, docs_path)

    return faiss_index
