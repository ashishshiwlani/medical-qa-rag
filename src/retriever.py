"""
Retrieval logic with Maximal Marginal Relevance (MMR) diversification.

Key concepts:
  - Basic retrieval: top-k most similar chunks
  - MMR retrieval: trade-off between relevance and diversity
    to avoid redundant/near-duplicate chunks
  - Why MMR for medical Q&A: same condition can have many similar
    retrieval results; we want diverse perspectives/sources
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from src.ingest import Document
from src.embeddings import EmbeddingModel, FAISSIndex


@dataclass
class RetrievedChunk:
    """
    A document chunk retrieved from the index.

    Attributes:
        document: The Document object
        score: Similarity score to query (0-1, higher is better)
        rank: Rank in retrieval results (0-indexed)
    """
    document: Document
    score: float
    rank: int


class Retriever:
    """
    Retrieval component that finds relevant documents from the index.

    Supports both basic top-k retrieval and MMR diversified retrieval.

    Attributes:
        faiss_index: FAISS index for similarity search
        embedding_model: Model for embedding queries
        top_k: Default number of results to retrieve
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        embedding_model: EmbeddingModel,
        top_k: int = 5
    ) -> None:
        """
        Initialize retriever.

        Args:
            faiss_index: Built FAISS index
            embedding_model: Embedding model (for query embedding)
            top_k: Default number of results to retrieve
        """
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Basic retrieval: return top-k most similar documents.

        Simple approach: embed query, search FAISS, return results.
        Good for most cases but can suffer from redundancy.

        Args:
            query: User query string

        Returns:
            List of RetrievedChunk objects, ranked by similarity (descending)
        """
        # Embed query using same normalization as index
        query_embedding = self.embedding_model.embed_query(query)

        # Search FAISS index
        results = self.faiss_index.search(query_embedding, k=self.top_k)

        # Convert to RetrievedChunk objects with rank
        retrieved = []
        for rank, (doc, score) in enumerate(results):
            chunk = RetrievedChunk(
                document=doc,
                score=score,
                rank=rank
            )
            retrieved.append(chunk)

        return retrieved

    def retrieve_with_mmr(
        self,
        query: str,
        lambda_mult: float = 0.5
    ) -> List[RetrievedChunk]:
        """
        Maximal Marginal Relevance (MMR) diversified retrieval.

        MMR balances relevance vs. diversity:
            MMR = lambda * relevance(doc, query) - (1 - lambda) * max(redundancy(doc, selected))

        Why useful for medical Q&A:
          - Same disease often appears in multiple similar documents
          - Want both relevance AND different perspectives/sources
          - Avoids returning 5 near-identical chunks

        Algorithm:
          1. Embed query, get all candidates from index
          2. Greedily select: highest (relevance - redundancy) each round
          3. Mark selected, repeat until k results

        Args:
            query: User query string
            lambda_mult: Balance parameter in [0, 1]
                - 1.0 = pure relevance (same as basic retrieval)
                - 0.0 = pure diversity (random selection)
                - 0.5 = balanced (recommended for medical Q&A)

        Returns:
            List of RetrievedChunk objects, diversified
        """
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)

        # Get candidate pool: return more than top_k for diversification
        # (search for 2*top_k to have room for diversity filtering)
        candidate_pool_size = min(len(self.faiss_index.documents), self.top_k * 3)
        all_results = self.faiss_index.search(query_embedding, k=candidate_pool_size)

        if not all_results:
            return []

        # Extract embeddings and scores for candidates
        # We need to recompute embeddings to compare diversity
        candidate_docs = [doc for doc, _ in all_results]
        candidate_scores = np.array([score for _, score in all_results])
        candidate_texts = [doc.content for doc in candidate_docs]

        # Embed candidates for diversity comparison
        candidate_embeddings = self.embedding_model.embed(
            candidate_texts,
            show_progress=False
        )

        # Normalize for cosine similarity calculation
        candidate_embeddings = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        )
        query_embedding_norm = query_embedding / (
            np.linalg.norm(query_embedding) + 1e-10
        )

        # Greedy selection: build result list incrementally
        selected_indices = []
        remaining_indices = set(range(len(all_results)))

        while len(selected_indices) < self.top_k and remaining_indices:
            # Calculate MMR for each remaining candidate
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance: dot product with query
                relevance = candidate_scores[idx]

                # Redundancy: max similarity to already-selected docs
                if selected_indices:
                    selected_embeddings = candidate_embeddings[selected_indices]
                    candidate_embedding = candidate_embeddings[idx]

                    # Cosine similarities to all selected documents
                    redundancies = np.dot(
                        selected_embeddings,
                        candidate_embedding
                    )
                    max_redundancy = np.max(redundancies)
                else:
                    # No selected docs yet, so no redundancy
                    max_redundancy = 0.0

                # MMR formula: balance relevance and diversity
                mmr = (lambda_mult * relevance -
                       (1 - lambda_mult) * max_redundancy)
                mmr_scores.append(mmr)

            # Select candidate with highest MMR
            best_idx = remaining_indices.pop(
                list(remaining_indices)[np.argmax(mmr_scores)]
            )
            selected_indices.append(best_idx)

        # Convert selected indices to RetrievedChunk objects
        retrieved = []
        for rank, idx in enumerate(selected_indices):
            doc = candidate_docs[idx]
            score = float(candidate_scores[idx])
            chunk = RetrievedChunk(
                document=doc,
                score=score,
                rank=rank
            )
            retrieved.append(chunk)

        return retrieved


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Includes source citations and confidence scores.

    Args:
        chunks: List of RetrievedChunk objects

    Returns:
        Formatted context string with citations

    Example:
        >>> context = format_context(chunks)
        >>> # Context:
        >>> # [1] (similarity: 0.89, source: MedQuAD, topic: Diabetes)
        >>> # Type 2 diabetes symptoms include...
        >>> # [2] (similarity: 0.85, source: PubMed, title: ...)
        >>> # ...
    """
    if not chunks:
        return "[No relevant documents found]"

    context_parts = ["## Retrieved Medical Context\n"]

    for chunk in chunks:
        doc = chunk.document
        rank = chunk.rank + 1  # 1-indexed for readability

        # Citation header with metadata
        meta_str = f"source: {doc.metadata.get('source', 'unknown')}"
        if "topic" in doc.metadata:
            meta_str += f", topic: {doc.metadata['topic']}"
        if "specialty" in doc.metadata:
            meta_str += f", specialty: {doc.metadata['specialty']}"

        header = (
            f"\n[{rank}] (similarity: {chunk.score:.2f}, {meta_str})\n"
        )
        context_parts.append(header)

        # Content
        context_parts.append(doc.content)

    return "".join(context_parts)
