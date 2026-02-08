# PantryPal

[![Tests](https://github.com/t1mato/smart-pantry/actions/workflows/test.yml/badge.svg)](https://github.com/t1mato/smart-pantry/actions/workflows/test.yml)

**Make the most of your pantry with a click of a button.**

A chatbot-style recipe finder powered by hybrid search (BM25 + Semantic) with cross-encoder reranking. Ask naturally about what you want to cook!

<!-- TODO: Replace with actual screenshot of the Streamlit UI -->
<!-- Run the app, search for a recipe, and capture a screenshot: -->
<!-- Save it as assets/screenshot.png, then uncomment the line below -->
<!-- ![PantryPal UI](assets/screenshot.png) -->

## Overview

Three-phase RAG system with rigorous RAGAS evaluation:

1. **Ingestion:** PDFs → text extraction → chunking → local embeddings → ChromaDB
2. **Query Expansion:** User query → LLM adds synonyms/keywords → enriched search query
3. **Hybrid Retrieval:** BM25 (keyword) + Semantic (embeddings) → RRF fusion → cross-encoder reranking
4. **Generation:** Context → Gemini LLM → formatted recipe with dietary adaptation

### Architecture

- **Query Expansion:** LLM-based keyword extraction and synonym generation
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (local, 384-dim)
- **Hybrid Search:** BM25 + Semantic with Reciprocal Rank Fusion (RRF)
- **Reranking:** Cross-encoder `ms-marco-MiniLM-L-6-v2` (+6.1% answer quality)
- **Vector Store:** ChromaDB (local SQLite persistence)
- **LLM:** Google Gemini 2.5 Flash
- **Evaluation:** RAGAS framework (Groq llama-3.1-8b for metrics)
- **Frontend:** Streamlit with Space Grotesk font and Soft Sage theme

### Key Features

- **Query Expansion:** LLM expands user queries with synonyms and related terms for better matching
- **Hybrid Search:** Combines keyword matching (BM25) with semantic understanding
- **Cross-Encoder Reranking:** Improves answer relevancy from 53.4% → 59.5%
- **Batch Ingestion:** Process entire folders of PDFs with duplicate detection
- **Modular Architecture:** Shared `core/` module with centralized config
- **RAGAS Evaluation:** 4 metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy)
- **Unit Tests:** pytest suite for core retrieval algorithms
- **Perfect Recall:** 100% - users won't miss relevant recipes
- **Debug Mode:** View raw retrieval results without LLM generation

## Requirements

- Python 3.10+ (3.9 has compatibility issues with `sentence-transformers`)
- Google AI API key (https://ai.google.dev/) - for recipe generation
- Groq API key (https://console.groq.com/) - optional, for RAGAS evaluation only

## Quick Start

A sample cookbook is included: [*Good and Cheap*](https://www.leannebrown.com/good-and-cheap/) by Leanne Brown (Creative Commons BY-NC-SA). No additional data needed.

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GOOGLE_API_KEY=<your-key>" > .env
echo "GROQ_API_KEY=<your-key>" >> .env  # Optional: for evaluation only

# Ingest the included cookbook (or add your own PDFs to data/)
python ingest.py                         # Default: Good and Cheap cookbook
python ingest.py data/cookbook.pdf        # Single file
python ingest.py data/                   # All PDFs in folder (batch mode)

# Run Streamlit app
streamlit run app.py

# Or run REST API
uvicorn api:app --reload
```

## Usage

### Streamlit App (PantryPal)

1. Navigate to http://localhost:8501
2. Describe what you want to cook naturally (e.g., "I have chicken, rice, and some vegetables")
3. Optionally specify dietary restrictions (e.g., "gluten-free, vegetarian")
4. Click "Find recipes"
5. Receive a formatted recipe with:
   - Ingredients list with measurements
   - Step-by-step instructions
   - Chef's notes with substitutions and "See also" alternatives
   - Source citations from the cookbook

**Debug mode:** Add `?debug=true` to URL to view raw retrieval results without LLM generation.

### REST API

Start the API server:
```bash
uvicorn api:app --reload
```

**Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — shows component status |
| POST | `/search` | Search recipes, returns raw chunks |
| POST | `/generate` | Search + LLM generation, returns formatted recipe |

**Interactive docs:** http://localhost:8000/docs (Swagger UI)

**Example request:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "I have chicken and rice, what can I make?", "use_reranking": true}'
```

### Running Evaluation
```bash
# Requires Groq API key in .env
python evaluation.py              # Run with default output
python evaluation.py --verbose    # Show debug output
python evaluation.py --quiet      # Only show errors

# Generates:
# - evaluation_results.csv (raw scores)
# - evaluation_results_report.txt (summary)
```

Evaluates 15 test cases across 5 categories (ingredients, dietary, budget, cuisine, meal type) against 4 retrieval methods.

## Project Structure

```
smart-pantry/
├── core/                       # Shared utilities module
│   ├── __init__.py             # Package exports
│   ├── config.py               # Centralized configuration (single source of truth)
│   ├── embeddings.py           # Embedding model + vectorstore initialization
│   ├── retrieval.py            # RRF fusion, cross-encoder reranking (cached), search
│   └── llm.py                  # LLM initialization, recipe generation, prompt template
├── tests/                      # pytest test suite (26 tests)
│   ├── conftest.py             # Shared fixtures (mock recipe data)
│   ├── test_retrieval.py       # Tests for RRF and format_context
│   ├── test_ingest.py          # Tests for PDF discovery and document splitting
│   └── test_search.py          # Tests for hybrid search pipeline (mocked)
├── .streamlit/config.toml      # Soft Sage theme configuration
├── app.py                      # PantryPal Streamlit UI (Space Grotesk font, centered layout)
├── ingest.py                   # PDF processing + batch ingestion + deduplication
├── evaluation.py               # RAGAS evaluation framework (imports from core only)
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
├── .env                        # API credentials (gitignored)
├── data/                       # Source PDFs (gitignored)
├── chroma_db/                  # Vector database (gitignored)
├── CLAUDE.md                   # Development documentation
├── EVALUATION_RESULTS.md       # RAGAS evaluation findings
└── README.md                   # This file
```

## Configuration

All configuration is centralized in `core/config.py` (single source of truth):

### Embeddings & Chunking
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (shared across all modules)
- `CHUNK_SIZE`: 2000 / `CHUNK_OVERLAP`: 200
- `CHUNK_SEPARATORS`: `["\n\n", "Title:", "Ingredients:"]`

### Retrieval
- `NUM_RESULTS`: 5 (final results returned)
- `RERANK_TOP_K`: 10 (candidates for reranking)
- `CROSS_ENCODER_MODEL`: "cross-encoder/ms-marco-MiniLM-L-6-v2"

### LLM & Evaluation
- `GEMINI_MODEL`: "gemini-flash-latest"
- `GROQ_MODEL`: "llama-3.1-8b-instant"
- Evaluates 4 methods: Semantic Only, BM25 Only, Hybrid (RRF), Hybrid + Reranking

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_retrieval.py::TestReciprocalRankFusion -v
```

26 unit tests covering RRF fusion, context formatting, PDF ingestion, and hybrid search.

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Vector database not found` | Run `python ingest.py` |
| `GOOGLE_API_KEY not found` | Create `.env` file |
| `404 models/gemini-* not found` | Use `gemini-flash-latest` |
| `Cannot copy out of meta tensor` | `pip install sentence-transformers==3.0.1` |
| Python 3.9 import errors | Upgrade to Python 3.10+ |

## Performance

### Speed
- **Ingestion**: ~2s for 90-page PDF
- **Query latency**: 2-5s (search + generation), +200ms with reranking
- **Storage**: ~3MB per cookbook

### Quality (RAGAS Evaluation)
- **Answer Relevancy**: 59.5% (with reranking, +6.1% improvement)
- **Context Recall**: 100% (perfect - won't miss recipes)
- **Faithfulness**: 88.4% (high answer grounding)
- **Context Precision**: 66.8% (acceptable noise/signal ratio)

See `EVALUATION_RESULTS.md` for detailed analysis.

### Cost
- **Embeddings**: $0 (local)
- **Vector DB**: $0 (local)
- **Generation**: ~$0.0001/query (Gemini free tier: 1500 req/day)
- **Total**: **Free** for typical usage

## Design Decisions

### Why local embeddings instead of Google's Embedding API?
Google's Embedding API hit rate limits (429 errors) immediately on the free tier. Since embeddings are a bulk operation during ingestion, local HuggingFace models (`all-MiniLM-L6-v2`, 384-dim) provide unlimited free embedding at the cost of slightly lower dimensionality vs Google's 768-dim — a worthwhile tradeoff for recipe matching.

### Why hybrid search instead of pure semantic?
Semantic search understands meaning ("poultry" matches "chicken") but can miss exact ingredient names. BM25 keyword search catches exact terms ("2 cups flour") but misses synonyms. Combining both with Reciprocal Rank Fusion (RRF) achieves **100% recall** — users never miss relevant recipes — while cross-encoder reranking on top brings answer relevancy to its highest at 59.5%.

### Why a separate evaluation framework?
Without measurement, "better" is just a feeling. The RAGAS evaluation framework compares 4 retrieval methods across 4 metrics, providing data-backed evidence that hybrid + reranking is the best approach. This caught a non-obvious finding: pure semantic search has *higher precision* (79.6% vs 66.8%) but *lower final answer quality* (58.7% vs 59.5%) — the LLM filters noisy context effectively.

## License

MIT
