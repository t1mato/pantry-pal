"""
Smart Pantry - FastAPI REST API

Provides programmatic access to recipe search and generation.
Uses the same core/ module as the Streamlit app.

Run with:
    uvicorn api:app --reload

Docs at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from core import (
    initialize_vectorstore,
    initialize_hybrid_retriever,
    search_hybrid,
    format_context,
    initialize_llm,
    generate_recipe,
)

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Smart Pantry API",
    description="Recipe search and generation using hybrid retrieval (BM25 + Semantic) with optional cross-encoder reranking.",
    version="1.0.0",
)

# Initialize components once at startup (cached for all requests)
# This is similar to @st.cache_resource in Streamlit
vectorstore = None
bm25_retriever = None
semantic_retriever = None
llm = None


@app.on_event("startup")
async def startup():
    """Initialize search components when the server starts."""
    global vectorstore, bm25_retriever, semantic_retriever, llm

    try:
        vectorstore = initialize_vectorstore()
        bm25_retriever, semantic_retriever = initialize_hybrid_retriever(vectorstore)
        llm = initialize_llm()
    except FileNotFoundError as e:
        # Database not found — server starts but endpoints will fail gracefully
        print(f"Warning: {e}")
    except ValueError as e:
        # API key not found — search works but generation fails
        print(f"Warning: {e}")


# -----------------------------------------------------------------------------
# Request/Response schemas
# -----------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Request body for /search endpoint."""

    query: str = Field(..., description="Natural language query about recipes", example="I have chicken and rice, what can I make?")
    restrictions: Optional[str] = Field(None, description="Dietary restrictions", example="gluten-free")
    num_results: int = Field(5, ge=1, le=20, description="Number of results to return")
    use_reranking: bool = Field(True, description="Enable cross-encoder reranking for better precision")


class SearchResult(BaseModel):
    """A single search result."""

    content: str = Field(..., description="Recipe chunk content")
    source: str = Field(..., description="Source cookbook filename")
    page: int = Field(..., description="Page number in the source PDF")


class SearchResponse(BaseModel):
    """Response body for /search endpoint."""

    query: str = Field(..., description="The constructed search query")
    results: list[SearchResult] = Field(..., description="Retrieved recipe chunks")


class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""

    query: str = Field(..., description="Natural language query about recipes", example="I have chicken and rice, what can I make?")
    restrictions: Optional[str] = Field(None, description="Dietary restrictions", example="gluten-free")
    num_results: int = Field(5, ge=1, le=20, description="Number of context chunks to use")
    use_reranking: bool = Field(True, description="Enable cross-encoder reranking")


class GenerateResponse(BaseModel):
    """Response body for /generate endpoint."""

    recipe: str = Field(..., description="Generated recipe recommendation")
    sources: list[SearchResult] = Field(..., description="Source chunks used for generation")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns the status of all components:
    - vectorstore: Is the recipe database loaded?
    - retrievers: Are BM25 and semantic search ready?
    - llm: Is the Gemini API configured?
    """
    return {
        "status": "healthy",
        "components": {
            "vectorstore": vectorstore is not None,
            "retrievers": bm25_retriever is not None and semantic_retriever is not None,
            "llm": llm is not None,
        },
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for recipes using natural language queries.

    Uses hybrid retrieval (BM25 + Semantic) with optional cross-encoder reranking.
    Returns raw recipe chunks without LLM generation.
    """
    if bm25_retriever is None or semantic_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Search not available. Run 'python ingest.py' to create the recipe database.",
        )

    # Use natural language query directly, append restrictions if provided
    query = request.query
    if request.restrictions:
        query += f" (dietary restrictions: {request.restrictions})"

    # Perform hybrid search
    results = search_hybrid(
        bm25_retriever,
        semantic_retriever,
        query,
        num_results=request.num_results,
        use_reranking=request.use_reranking,
    )

    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", 0),
            )
            for doc in results
        ],
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Search for recipes and generate a personalized recommendation.

    Performs hybrid search, then uses Gemini LLM to synthesize a recipe
    recommendation based on the retrieved context.
    """
    if bm25_retriever is None or semantic_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Search not available. Run 'python ingest.py' to create the recipe database.",
        )

    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Generation not available. Add GOOGLE_API_KEY to .env file.",
        )

    # Use natural language query directly, append restrictions if provided
    query = request.query
    if request.restrictions:
        query += f" (dietary restrictions: {request.restrictions})"

    # Perform hybrid search
    results = search_hybrid(
        bm25_retriever,
        semantic_retriever,
        query,
        num_results=request.num_results,
        use_reranking=request.use_reranking,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No recipes found matching your query.",
        )

    # Format context and generate
    context = format_context(results)
    recipe = generate_recipe(
        llm,
        context,
        request.query,
        request.restrictions or "",
    )

    return GenerateResponse(
        recipe=recipe,
        sources=[
            SearchResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", 0),
            )
            for doc in results
        ],
    )
