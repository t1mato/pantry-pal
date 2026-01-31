"""
Smart Pantry - Embedding and Vector Store Initialization

This module handles:
- HuggingFace embedding model initialization
- ChromaDB vector store connection
- BM25 + Semantic retriever setup

Centralizing this ensures ingest.py and app.py use identical settings.
"""

import os
from typing import Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from core.config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    NUM_RESULTS
)


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Create the HuggingFace embedding model instance.

    This function ensures consistent embedding configuration across
    ingestion and retrieval. The model downloads to ~/.cache/huggingface
    on first run (~120MB).

    Configuration:
    - device: "cpu" (change to "cuda" for GPU acceleration)
    - normalize_embeddings: True (required for cosine similarity)

    Returns:
        Configured HuggingFaceEmbeddings instance

    Note:
        Both ingest.py and app.py MUST use this function to ensure
        embedding consistency. Mismatched embeddings = broken search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def initialize_vectorstore() -> Chroma:
    """
    Load the existing ChromaDB vector store.

    This connects to the database created by ingest.py. It does NOT
    create a new database — use ingest.py for that.

    Returns:
        Chroma vector store instance

    Raises:
        FileNotFoundError: If database doesn't exist (run ingest.py first)
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector database not found at {CHROMA_PATH}. "
            f"Please run 'python ingest.py' first to create it."
        )

    embeddings = create_embeddings()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def initialize_hybrid_retriever(vectorstore: Chroma) -> Tuple:
    """
    Set up BM25 and semantic retrievers for hybrid search.

    This creates two parallel search paths:
    1. BM25: Keyword matching (exact terms like "2 cups flour")
    2. Semantic: Meaning matching (concepts like "chicken" → "poultry")

    Both retrieve extra results (2x NUM_RESULTS) to feed into RRF fusion,
    which combines them into a single ranked list.

    Args:
        vectorstore: Initialized Chroma vector store

    Returns:
        Tuple of (bm25_retriever, semantic_retriever)
    """
    # Extract all documents from vectorstore for BM25 indexing
    # BM25 needs the full corpus to calculate term frequencies (TF-IDF)
    all_docs = vectorstore.get()['documents']
    all_metadatas = vectorstore.get()['metadatas']

    # Reconstruct Document objects
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs, all_metadatas)
    ]

    # BM25 retriever (keyword-based)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = NUM_RESULTS * 2  # Extra results for fusion

    # Semantic retriever (embedding-based)
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": NUM_RESULTS * 2}
    )

    return bm25_retriever, semantic_retriever
