"""
Tests for core/retrieval.py

Tests the retrieval algorithms that power Smart Pantry's search:
- reciprocal_rank_fusion: Combines BM25 + semantic results
- format_context: Prepares search results for LLM prompt
"""

import pytest
from langchain_core.documents import Document

from core.retrieval import reciprocal_rank_fusion, format_context


class TestReciprocalRankFusion:
    """Tests for the RRF algorithm."""

    def test_combines_two_result_lists(self, bm25_results, semantic_results):
        """RRF should combine results from multiple retrievers."""
        fused = reciprocal_rank_fusion([bm25_results, semantic_results])

        # Should return documents (not empty)
        assert len(fused) > 0

        # All results should be Document objects
        assert all(isinstance(doc, Document) for doc in fused)

    def test_documents_appearing_in_both_lists_rank_higher(self, bm25_results, semantic_results):
        """Documents in both result lists should get higher RRF scores."""
        fused = reciprocal_rank_fusion([bm25_results, semantic_results])

        # Chicken Stir Fry and Chicken Soup appear in both lists
        # They should be in the top results
        top_contents = [doc.page_content for doc in fused[:2]]

        # At least one "chicken" recipe should be in top 2
        assert any("Chicken" in content for content in top_contents)

    def test_removes_duplicates(self, sample_documents):
        """RRF should not return the same document twice."""
        # Create two lists with overlapping documents
        list1 = sample_documents[:3]  # First 3 docs
        list2 = sample_documents[1:4]  # Docs 2, 3, 4 (overlap on 2 and 3)

        fused = reciprocal_rank_fusion([list1, list2])

        # Check for unique documents by content
        contents = [doc.page_content for doc in fused]
        assert len(contents) == len(set(contents)), "Duplicates found in fused results"

    def test_empty_input_returns_empty(self):
        """RRF with no results should return empty list."""
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_single_list_returns_same_order(self, bm25_results):
        """RRF with one list should preserve original order."""
        fused = reciprocal_rank_fusion([bm25_results])

        # Order should be preserved (same ranks)
        assert len(fused) == len(bm25_results)
        for i, doc in enumerate(fused):
            assert doc.page_content == bm25_results[i].page_content

    def test_k_parameter_affects_scoring(self, bm25_results, semantic_results):
        """
        TODO(human): Implement this test!

        The 'k' parameter in RRF affects how much rank differences matter.
        - Higher k = smoother differences between ranks
        - Lower k = more aggressive rank differentiation

        Your task: Test that changing k changes the relative scores.

        Hint: With k=1, rank 1 gets score 0.5, rank 2 gets 0.33
              With k=60, rank 1 gets 0.016, rank 2 gets 0.016 (almost same)

        Approach:
        1. Call reciprocal_rank_fusion with k=1 and k=100
        2. The order might be the same, but you could verify the function
           accepts the k parameter without error
        3. Or create a scenario where different k values produce different orders
        """

        fused = reciprocal_rank_fusion([bm25_results, semantic_results])

        fused_low_k = reciprocal_rank_fusion([bm25_results, semantic_results], k=1)
        fused_high_k = reciprocal_rank_fusion([bm25_results, semantic_results], k=100)

        assert len(fused_low_k) > 0
        assert len(fused_high_k) > 0


class TestFormatContext:
    """Tests for the context formatting function."""

    def test_includes_all_documents(self, sample_documents):
        """Formatted context should include all input documents."""
        context = format_context(sample_documents[:3])

        # Should mention all 3 documents
        assert "Recipe Chunk 1" in context
        assert "Recipe Chunk 2" in context
        assert "Recipe Chunk 3" in context

    def test_includes_metadata(self, sample_documents):
        """Formatted context should include source and page info."""
        context = format_context([sample_documents[0]])

        assert "asian-cookbook.pdf" in context
        assert "42" in context  # Page number

    def test_includes_content(self, sample_documents):
        """Formatted context should include document content."""
        context = format_context([sample_documents[0]])

        assert "Chicken Stir Fry" in context
        assert "Heat oil in wok" in context

    def test_empty_input_returns_empty_string(self):
        """Empty document list should return empty string."""
        context = format_context([])
        assert context == ""

    def test_handles_missing_metadata(self):
        """Should handle documents with missing metadata gracefully."""
        doc_no_metadata = Document(
            page_content="Some recipe content",
            metadata={}  # No source or page
        )

        context = format_context([doc_no_metadata])

        # Should use default values, not crash
        assert "Unknown source" in context
        assert "Unknown page" in context
