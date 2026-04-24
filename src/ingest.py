"""
Document ingestion and chunking pipeline.

Handles loading documents from various formats (JSON, TXT, JSONL) and
chunking them into overlapping segments for embedding and retrieval.

Key concepts:
  - Chunking with overlap: preserves context at chunk boundaries (e.g., a sentence
    split across chunks remains coherent if chunks overlap)
  - Metadata preservation: maintains source and domain info for citation
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm


@dataclass
class Document:
    """
    Represents a text document with metadata.

    Attributes:
        id: Unique identifier for the document
        content: Full text content to be embedded
        metadata: Additional contextual information (source, topic, specialty, etc.)
    """
    id: str
    content: str
    metadata: Dict[str, Any]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> List[str]:
    """
    Split text into overlapping chunks.

    Overlap is critical for RAG because:
    - Prevents relevant information from being split across chunks
    - Ensures semantic coherence at chunk boundaries
    - Increases chances of retrieving complete information units

    For example, if a sentence spans chunks 1-2, both will be retrieved
    if the query matches either, giving the LLM the full context.

    Args:
        text: Raw text to chunk
        chunk_size: Target chunk length in characters (not tokens)
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks with overlap

    Example:
        >>> text = "ABC" * 300  # 900 chars
        >>> chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        >>> len(chunks)  # Multiple chunks due to overlap
        >>> len(chunks[0])  # ~200
        >>> chunks[0][-50:] == chunks[1][:50]  # Overlap verified
    """
    # Prevent invalid parameters (chunk_overlap must be < chunk_size)
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    chunks = []
    start = 0

    # Sliding window approach: stride = chunk_size - overlap
    while start < len(text):
        # End of chunk, but don't exceed text length
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        # If we've reached the end, stop
        if end == len(text):
            break

        # Move start position by (chunk_size - overlap) for next iteration
        start = end - chunk_overlap

    return chunks


def load_json_corpus(filepath: str) -> List[Document]:
    """
    Load medical Q&A corpus from JSON file.

    Expected format: list of objects with 'id', 'source', 'topic',
    'question', 'answer', and 'metadata' fields.

    Args:
        filepath: Path to JSON corpus file

    Returns:
        List of Document objects (question + answer combined as content)

    Raises:
        FileNotFoundError: If filepath doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for entry in data:
        # Combine question and answer for richer context
        # Why: LLM can leverage both to understand the topic better
        # Use full words ("Question:"/"Answer:") so keyword searches and test
        # assertions for the word "question" match reliably
        content = f"Question: {entry['question']}\nAnswer: {entry['answer']}"

        doc = Document(
            id=entry["id"],
            content=content,
            metadata={
                "source": entry.get("source", "unknown"),
                "topic": entry.get("topic", "general"),
                "question": entry.get("question", ""),
                **entry.get("metadata", {})
            }
        )
        documents.append(doc)

    return documents


def load_text_file(
    filepath: str,
    source_name: str = "text_document"
) -> List[Document]:
    """
    Load and chunk a plain text file.

    Args:
        filepath: Path to text file
        source_name: Source identifier for metadata

    Returns:
        List of Document objects (chunked)

    Raises:
        FileNotFoundError: If filepath doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Text file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

    documents = []
    basename = os.path.basename(filepath)

    for i, chunk in enumerate(chunks):
        doc = Document(
            id=f"{basename}_chunk_{i}",
            content=chunk,
            metadata={
                "source": source_name,
                "file": basename,
                "chunk_index": i
            }
        )
        documents.append(doc)

    return documents


def load_pubmed_abstracts(filepath: str) -> List[Document]:
    """
    Load PubMed abstracts from JSONL file.

    Expected format: one JSON object per line with 'id', 'title',
    'abstract', and optional 'mesh_terms' fields.

    Args:
        filepath: Path to JSONL file with PubMed abstracts

    Returns:
        List of Document objects

    Raises:
        FileNotFoundError: If filepath doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PubMed file not found: {filepath}")

    documents = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                # Combine title and abstract for full context
                content = f"{entry.get('title', '')}\n\n{entry.get('abstract', '')}"

                doc = Document(
                    id=entry.get("id", f"pubmed_{line_num}"),
                    content=content,
                    metadata={
                        "source": "PubMed",
                        "title": entry.get("title", ""),
                        "mesh_terms": entry.get("mesh_terms", [])
                    }
                )
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue

    return documents


class IngestPipeline:
    """
    Orchestrates multi-format document ingestion and chunking.

    Handles heterogeneous data sources (JSON corpora, TXT files, JSONL datasets)
    and produces a unified list of Document objects ready for embedding.

    Attributes:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize ingestion pipeline.

        Args:
            chunk_size: Target chunk length
            chunk_overlap: Overlap length between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, filepath: str) -> List[Document]:
        """
        Ingest a single file, auto-detecting format.

        Args:
            filepath: Path to file (.json, .txt, or .jsonl)

        Returns:
            List of Document objects

        Raises:
            ValueError: If file format not recognized
        """
        filepath = str(filepath)
        suffix = Path(filepath).suffix.lower()

        if suffix == ".json":
            # Check if it's a corpus file (array) or single object
            return load_json_corpus(filepath)
        elif suffix == ".txt":
            return load_text_file(filepath, source_name=Path(filepath).stem)
        elif suffix == ".jsonl":
            return load_pubmed_abstracts(filepath)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: .json, .txt, .jsonl"
            )

    def ingest_directory(self, data_dir: str) -> List[Document]:
        """
        Recursively ingest all supported documents from directory.

        Walks directory tree and loads all .json, .txt, .jsonl files.
        Useful for building large corpora from mixed sources.

        Args:
            data_dir: Root directory containing documents

        Returns:
            Aggregated list of Document objects from all files

        Raises:
            FileNotFoundError: If data_dir doesn't exist
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        all_documents = []
        supported_extensions = {".json", ".txt", ".jsonl"}

        # Walk all subdirectories
        for root, dirs, files in os.walk(data_dir):
            for file in sorted(files):
                file_path = Path(root) / file

                # Only process supported formats
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        docs = self.ingest_file(str(file_path))
                        all_documents.extend(docs)
                        print(f"✓ Ingested {len(docs)} documents from {file}")
                    except Exception as e:
                        print(f"✗ Failed to ingest {file}: {e}")
                        continue

        return all_documents
