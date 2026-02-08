"""
Tests for core/retrieval.py search pipeline

Tests the hybrid search orchestration and cross-encoder reranking
using mocked retrievers and models. No real ML models are loaded.

Why mocks?
- search_hybrid depends on BM25/semantic retrievers (need a real database)
- rerank_with_cross_encoder loads a ~100MB neural network
- Mocking replaces these with controlled fakes so we test OUR logic,
  not ChromaDB or HuggingFace infrastructure
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from core.retrieval import search_hybrid, rerank_with_cross_encoder


class TestSearchHybrid:
    """Tests for the hybrid search orchestration function."""

    def test_combines_bm25_and_semantic_results(self, bm25_results, semantic_results):
        """search_hybrid should call both retrievers and fuse results."""
        # Create fake retrievers that return our fixture data
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = bm25_results

        mock_semantic = MagicMock()
        mock_semantic.invoke.return_value = semantic_results

        results = search_hybrid(mock_bm25, mock_semantic, "chicken recipes")

        # Verify both retrievers were called with the query
        mock_bm25.invoke.assert_called_once_with("chicken recipes")
        mock_semantic.invoke.assert_called_once_with("chicken recipes")
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)

    def test_respects_num_results(self, bm25_results, semantic_results):
        """Should return at most num_results documents."""
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = bm25_results

        mock_semantic = MagicMock()
        mock_semantic.invoke.return_value = semantic_results

        results = search_hybrid(mock_bm25, mock_semantic, "test", num_results=2)

        assert len(results) <= 2

    def test_empty_results_from_both_retrievers(self):
        """Should handle both retrievers returning empty lists."""
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = []

        mock_semantic = MagicMock()
        mock_semantic.invoke.return_value = []

        results = search_hybrid(mock_bm25, mock_semantic, "nonexistent recipe")

        assert results == []

    @patch("core.retrieval.rerank_with_cross_encoder")
    def test_reranking_path(self, mock_rerank, bm25_results, semantic_results):
        """When use_reranking=True, should call rerank_with_cross_encoder."""
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = bm25_results

        mock_semantic = MagicMock()
        mock_semantic.invoke.return_value = semantic_results

        # Control what the reranker returns
        mock_rerank.return_value = bm25_results[:2]

        results = search_hybrid(
            mock_bm25, mock_semantic, "chicken", num_results=2, use_reranking=True
        )

        # Verify reranker was invoked
        mock_rerank.assert_called_once()
        assert len(results) == 2


class TestRerankWithCrossEncoder:
    """Tests for cross-encoder reranking."""

    @pytest.fixture(autouse=True)
    def clear_model_cache(self):
        """Clear the cross-encoder cache before each test.

        Why? We implemented model caching in core/retrieval.py for performance.
        But that means the mock from one test persists into the next.
        This fixture clears the cache so each test gets a fresh mock.
        """
        from core.retrieval import _cross_encoder_cache
        _cross_encoder_cache.clear()
        yield
        _cross_encoder_cache.clear()

    def test_empty_documents_returns_empty(self):
        """Reranking with no documents should return empty list."""
        result = rerank_with_cross_encoder("test query", [])

        assert result == []

    @patch("sentence_transformers.CrossEncoder")
    def test_respects_top_k(self, mock_ce_class, sample_documents):
        """Should return exactly top_k documents."""
        # Create a fake model that returns controlled scores
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
        mock_ce_class.return_value = mock_model

        result = rerank_with_cross_encoder("test", sample_documents, top_k=3)

        assert len(result) == 3

    @patch("sentence_transformers.CrossEncoder")
    def test_returns_highest_scored_first(self, mock_ce_class, sample_documents):
        """Documents should be sorted by cross-encoder score descending."""
        mock_model = MagicMock()
        # Give the last document (Pasta Primavera) the highest score
        mock_model.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.9]
        mock_ce_class.return_value = mock_model

        result = rerank_with_cross_encoder("test", sample_documents, top_k=2)

        # Pasta Primavera (score 0.9) should be first
        assert result[0].page_content == sample_documents[4].page_content
