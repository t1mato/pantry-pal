"""
Pytest Configuration and Shared Fixtures

Fixtures are reusable test components that pytest injects into test functions.
Define mock data and setup logic here to avoid repetition across tests.
"""

import pytest
from langchain_core.documents import Document


# ============================================================================
# MOCK DOCUMENTS
# ============================================================================

@pytest.fixture
def sample_documents():
    """
    Create sample Document objects for testing retrieval functions.

    These mimic real cookbook chunks with page_content and metadata.
    """
    return [
        Document(
            page_content="Chicken Stir Fry: Heat oil in wok, add chicken pieces...",
            metadata={"source": "asian-cookbook.pdf", "page": 42}
        ),
        Document(
            page_content="Vegetable Fried Rice: Use day-old rice for best results...",
            metadata={"source": "asian-cookbook.pdf", "page": 45}
        ),
        Document(
            page_content="Beef Tacos: Season ground beef with cumin and chili...",
            metadata={"source": "mexican-cookbook.pdf", "page": 12}
        ),
        Document(
            page_content="Chicken Soup: Simmer chicken bones for rich broth...",
            metadata={"source": "comfort-food.pdf", "page": 8}
        ),
        Document(
            page_content="Pasta Primavera: Toss pasta with seasonal vegetables...",
            metadata={"source": "italian-cookbook.pdf", "page": 67}
        ),
    ]


@pytest.fixture
def bm25_results(sample_documents):
    """
    Simulate BM25 retriever results (keyword-based ranking).

    BM25 ranks by term frequency, so "chicken" queries return
    chicken recipes first.
    """
    # BM25 would rank these by keyword matches
    return [
        sample_documents[0],  # Chicken Stir Fry (has "chicken")
        sample_documents[3],  # Chicken Soup (has "chicken")
        sample_documents[1],  # Fried Rice (has "rice", related)
    ]


@pytest.fixture
def semantic_results(sample_documents):
    """
    Simulate semantic retriever results (meaning-based ranking).

    Semantic search might find conceptually related recipes
    even without exact keyword matches.
    """
    # Semantic search finds conceptually similar recipes
    return [
        sample_documents[3],  # Chicken Soup (semantically similar to chicken query)
        sample_documents[0],  # Chicken Stir Fry
        sample_documents[4],  # Pasta (might be semantically related to "dinner")
    ]


@pytest.fixture
def empty_documents():
    """Empty document list for edge case testing."""
    return []


@pytest.fixture
def single_document(sample_documents):
    """Single document for minimal case testing."""
    return [sample_documents[0]]
