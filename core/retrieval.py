"""
Smart Pantry - Retrieval Functions

Core retrieval algorithms used by app.py and evaluation.py:
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Hybrid search orchestration
"""

from typing import List, Tuple
from langchain_core.documents import Document

from core.config import (
    RRF_K,
    NUM_RESULTS,
    RERANK_TOP_K,
    CROSS_ENCODER_MODEL
)


def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = RRF_K
) -> List[Document]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is a simple but effective rank aggregation method that:
    - Doesn't require score normalization (unlike weighted average)
    - Handles missing documents gracefully
    - Works well when sources have different scoring scales

    Formula: score(d) = Σ 1/(k + rank_r(d)) for each ranker r

    Args:
        results_list: List of result lists from different retrievers
        k: RRF constant (default 60). Higher = smoother ranking differences.

    Returns:
        Fused list of unique documents, sorted by combined RRF score

    Example:
        >>> bm25_results = [doc_a, doc_b, doc_c]  # BM25 ranking
        >>> semantic_results = [doc_b, doc_a, doc_d]  # Semantic ranking
        >>> fused = reciprocal_rank_fusion([bm25_results, semantic_results])
        >>> # doc_a and doc_b score higher (appear in both), doc_c and doc_d lower
    """
    # Track scores per document: {doc_id: (cumulative_score, Document)}
    doc_scores: dict[Tuple[str, str], Tuple[float, Document]] = {}

    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            # Create unique ID from content + metadata
            doc_id = (doc.page_content, str(doc.metadata))

            # RRF score for this rank position
            score = 1.0 / (k + rank)

            # Accumulate scores across retrievers
            if doc_id in doc_scores:
                current_score, _ = doc_scores[doc_id]
                doc_scores[doc_id] = (current_score + score, doc)
            else:
                doc_scores[doc_id] = (score, doc)

    # Sort by score descending and return documents
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


def rerank_with_cross_encoder(
    query: str,
    documents: List[Document],
    top_k: int = NUM_RESULTS,
    model_name: str = CROSS_ENCODER_MODEL
) -> List[Document]:
    """
    Rerank documents using a cross-encoder model for improved precision.

    Cross-encoders vs Bi-encoders (embeddings):
    - Bi-encoder: Encodes query and doc separately, compares vectors (fast)
    - Cross-encoder: Encodes query+doc together, captures interactions (accurate)

    This function uses cross-encoder as a second-stage ranker:
    1. First stage: Fast retrieval with embeddings (get 10+ candidates)
    2. Second stage: Accurate reranking with cross-encoder (return top 5)

    Args:
        query: User's search query
        documents: Candidate documents to rerank
        top_k: Number of top documents to return after reranking
        model_name: HuggingFace cross-encoder model ID

    Returns:
        Top-k documents sorted by cross-encoder relevance score

    Note:
        Model is loaded lazily (on first call) and cached for subsequent calls.
        First call may take 1-2 seconds for model loading.
    """
    if not documents:
        return []

    # Lazy import to avoid loading model when not needed
    from sentence_transformers import CrossEncoder

    # Load cross-encoder (cached after first load)
    cross_encoder = CrossEncoder(model_name)

    # Create query-document pairs for scoring
    pairs = [[query, doc.page_content] for doc in documents]

    # Get relevance scores (higher = more relevant)
    scores = cross_encoder.predict(pairs)

    # Sort by score and return top-k
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in doc_score_pairs[:top_k]]


def search_hybrid(
    bm25_retriever,
    semantic_retriever,
    query: str,
    num_results: int = NUM_RESULTS,
    use_reranking: bool = False
) -> List[Document]:
    """
    Perform hybrid search combining BM25 and semantic retrieval.

    Pipeline:
    1. Get results from BM25 (keyword matching)
    2. Get results from semantic (embedding similarity)
    3. Fuse using Reciprocal Rank Fusion
    4. Optionally rerank with cross-encoder

    Args:
        bm25_retriever: Initialized BM25Retriever
        semantic_retriever: Initialized semantic retriever (from vectorstore)
        query: Search query
        num_results: Number of results to return
        use_reranking: Whether to apply cross-encoder reranking

    Returns:
        List of Document objects, ranked by relevance
    """
    # Get results from both retrievers
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)

    # Combine using RRF
    fused_results = reciprocal_rank_fusion([bm25_results, semantic_results])

    # Apply reranking if enabled
    if use_reranking:
        candidates = fused_results[:RERANK_TOP_K]
        return rerank_with_cross_encoder(query, candidates, top_k=num_results)

    return fused_results[:num_results]


def format_context(search_results: List[Document]) -> str:
    """
    Format search results into a context string for LLM prompts.

    Creates a structured format that preserves:
    - Source attribution (cookbook name, page number)
    - Content hierarchy (numbered chunks)

    Args:
        search_results: List of Document objects from search

    Returns:
        Formatted string with all context, ready for LLM prompt
    """
    context_parts = []

    for i, doc in enumerate(search_results, 1):
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')

        context_parts.append(
            f"--- Recipe Chunk {i} ---\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n".join(context_parts)
