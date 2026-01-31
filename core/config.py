"""
Smart Pantry - Centralized Configuration

This module contains all configuration constants used across the application.
Centralizing config prevents version skew between ingest.py, app.py, and evaluation.py.

Usage:
    from core.config import EMBEDDING_MODEL, CHROMA_PATH
"""

# ============================================================================
# PATHS
# ============================================================================

# Vector database storage location (shared between ingest and app)
CHROMA_PATH = "./chroma_db"

# Default PDF for ingestion
DEFAULT_PDF_PATH = "data/good-and-cheap-by-leanne-brown.pdf"


# ============================================================================
# EMBEDDING MODEL
# ============================================================================

# CRITICAL: This model MUST be identical in ingest.py and app.py
# Mismatched embeddings result in nonsensical similarity scores
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding dimension (for reference/validation)
EMBEDDING_DIMENSION = 384


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Maximum characters per chunk
# Larger = more context per chunk, but less precise matching
CHUNK_SIZE = 2000

# Characters shared between adjacent chunks
# Prevents splitting mid-recipe or mid-sentence
CHUNK_OVERLAP = 200

# Split preferentially on these patterns (recipe-optimized)
CHUNK_SEPARATORS = ["\n\n", "Title:", "Ingredients:"]

# Batch size for processing chunks during ingestion
BATCH_SIZE = 50


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Number of final results to return to user
NUM_RESULTS = 5

# BM25/Semantic balance for hybrid search (must sum to 1.0)
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

# RRF (Reciprocal Rank Fusion) constant
# Higher k = less aggressive rank differences, more stable fusion
RRF_K = 60


# ============================================================================
# CROSS-ENCODER RERANKING
# ============================================================================

# Model trained on Microsoft MARCO passage ranking dataset
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of candidates to retrieve before reranking
# Higher = better reranking quality, but slower
RERANK_TOP_K = 10


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Google Gemini model for recipe generation
# "gemini-flash-latest" auto-selects the newest flash model
GEMINI_MODEL = "gemini-flash-latest"

# Temperature for generation (0 = deterministic, 1 = creative)
GEMINI_TEMPERATURE = 0.3


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Groq model for RAGAS evaluation (fast, free tier)
GROQ_MODEL = "llama-3.1-8b-instant"
