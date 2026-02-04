"""
Smart Pantry - Core Module

Shared utilities for the Smart Pantry RAG application.
Import from here for clean, centralized access to core functionality.

Usage:
    from core import (
        # Config
        EMBEDDING_MODEL,
        CHROMA_PATH,
        NUM_RESULTS,

        # Embeddings
        create_embeddings,
        initialize_vectorstore,
        initialize_hybrid_retriever,

        # Retrieval
        reciprocal_rank_fusion,
        rerank_with_cross_encoder,
        search_hybrid,
        format_context,
    )
"""

# Configuration constants
from core.config import (
    # Paths
    CHROMA_PATH,
    DEFAULT_PDF_PATH,

    # Embedding settings
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,

    # Chunking settings
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    BATCH_SIZE,

    # Retrieval settings
    NUM_RESULTS,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    RRF_K,

    # Reranking settings
    CROSS_ENCODER_MODEL,
    RERANK_TOP_K,

    # LLM settings
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GROQ_MODEL,
)

# Embedding and vectorstore functions
from core.embeddings import (
    create_embeddings,
    initialize_vectorstore,
    initialize_hybrid_retriever,
)

# Retrieval functions
from core.retrieval import (
    reciprocal_rank_fusion,
    rerank_with_cross_encoder,
    search_hybrid,
    format_context,
)

# LLM functions (shared by app.py and evaluation.py)
from core.llm import (
    initialize_llm,
    generate_recipe,
    RECIPE_PROMPT_TEMPLATE,
)
