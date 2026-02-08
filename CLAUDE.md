# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PantryPal** is a chatbot-style recipe finder powered by RAG (Retrieval-Augmented Generation). Users describe what they want to cook naturally, and the app searches PDF cookbooks using hybrid search (BM25 + Semantic) with cross-encoder reranking to find and adapt recipes.

### UI Design
- **Branding:** "PantryPal" with tagline "Make the most of your pantry with a click of a button"
- **Typography:** Space Grotesk font (Google Fonts)
- **Color scheme:** Soft Sage theme (`#5a7d5a` primary, `#f8faf8` background)
- **Layout:** Centered, minimal, professional
- **Theme config:** `.streamlit/config.toml`

## Architecture

### Four-Phase Pipeline
1. **Ingestion** (`ingest.py`): PDF → text chunks → local embeddings → ChromaDB (batch support with deduplication)
2. **Query Expansion** (`core/llm.py`): User query → LLM extracts keywords + adds synonyms → enriched query
3. **Hybrid Retrieval** (`core/retrieval.py`): BM25 + Semantic → RRF fusion → cross-encoder reranking
4. **Generation** (`core/llm.py`): Context → Gemini → formatted recipe (served via `app.py` Streamlit UI)

### Evaluation (offline)
- **`evaluation.py`**: RAGAS metrics via Groq API — runs independently from the user-facing pipeline

### Module Structure
- **`core/`** — Shared utilities module (config, retrieval algorithms, embeddings, LLM)
  - `config.py`: All configuration constants (single source of truth)
  - `retrieval.py`: RRF fusion, cross-encoder reranking (cached), hybrid search, format_context
  - `embeddings.py`: Embedding model creation, vectorstore initialization, hybrid retriever setup
  - `llm.py`: LLM initialization, query expansion, recipe generation, prompt templates (shared by app + evaluation)
  - `__init__.py`: Clean public API for imports
- **`app.py`** — PantryPal Streamlit UI (Space Grotesk font, Soft Sage theme, centered layout)
- **`.streamlit/config.toml`** — Theme configuration (colors, font)
- **`ingest.py`** — PDF processing, batch ingestion, deduplication
- **`evaluation.py`** — RAGAS evaluation (imports everything from `core`, no dependency on `app`)
- **`tests/`** — pytest suite with fixtures and unit tests

### Key Components

**Vector Database (ChromaDB)**
- Local SQLite persistence at `./chroma_db`
- Stores 384-dim embeddings from HuggingFace `all-MiniLM-L6-v2`
- Chunks: 2000 chars with 200-char overlap

**Embeddings Strategy**
- **Critical:** Uses local HuggingFace embeddings, NOT Google Embedding API
- Reason: Google's free tier has strict rate limits (hit 429 errors immediately)
- Model: `all-MiniLM-L6-v2` - fast, lightweight (~120MB), sufficient for recipe matching
- Embedding creation centralized in `core/embeddings.py::create_embeddings()` — ensures consistency

**LLM Integration**
- Google Gemini 2.5 Flash via `langchain-google-genai`
- Model: `gemini-flash-latest` (automatically uses newest available)
- Used for query expansion AND recipe generation, NOT embeddings
- API key: `GOOGLE_API_KEY` in `.env`
- **Query expansion:** Extracts ingredients, adds synonyms (chicken→poultry), adds cooking methods (stir-fry, curry)
- **Response format:** Clean markdown with ## headers, ingredients list, numbered instructions, and Chef's Notes with "See also" alternatives
- **Important:** Handles both old (string) and new (list of content blocks) Gemini response formats

**Document Processing**
- PDFs loaded via `PyPDFLoader` from `data/` directory
- `RecursiveCharacterTextSplitter` with recipe-specific separators: `["\n\n", "Title:", "Ingredients:"]`
- Preserves recipe structure (title, ingredients, instructions)

**Hybrid Search (core/retrieval.py)**
- **BM25 Retriever**: Keyword-based search using `rank-bm25` library
- **Semantic Retriever**: ChromaDB vector similarity search
- **RRF Fusion**: Reciprocal Rank Fusion combines both result sets
- Weighting: 40% BM25, 60% semantic (configurable in `core/config.py`)

**Cross-Encoder Reranking (core/retrieval.py)**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Retrieves top 10 candidates from RRF fusion
- Reranks with cross-encoder scores
- Returns top 5 for LLM generation
- **Improvement**: +6.1% answer relevancy (RAGAS evaluated)

**RAGAS Evaluation (evaluation.py)**
- Framework: RAGAS v0.3 (legacy API for LangChain compatibility)
- LLM: Groq `llama-3.1-8b-instant` (fast, free tier)
- Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
- Evaluates 4 methods: Semantic Only, BM25 Only, Hybrid (RRF), Hybrid + Reranking
- **Critical Fix**: `AnswerRelevancy(strictness=1)` for Groq n=1 compatibility

## Development Commands

See `README.md` for setup, installation, and usage commands. Key quick-reference:

```bash
python ingest.py              # Ingest default PDF
python ingest.py data/        # Batch ingest all PDFs in folder
python ingest.py --verbose    # Show debug output
python ingest.py --quiet      # Only show errors
streamlit run app.py          # Run the app (http://localhost:8501)
uvicorn api:app --reload      # Run REST API (http://localhost:8000)
python evaluation.py          # Run RAGAS evaluation
python evaluation.py --verbose  # Evaluation with debug output
rm -rf chroma_db/ && python ingest.py  # Reset database
```

## Configuration

**All configuration is centralized in `core/config.py`** — single source of truth. See `README.md` for the full settings reference.

**Evaluation-specific settings** (in `evaluation.py`):
- `TEST_CASES = 15` - diverse queries across 5 categories (ingredients, dietary, budget, cuisine, meal type)
- `AnswerRelevancy(strictness=1)` - CRITICAL: Groq only supports n=1
- Use `ragas.metrics` (legacy), NOT `ragas.metrics.collections`
- Supports `--verbose` / `--quiet` flags for log level control

## Important Implementation Details

### Embedding Model Consistency
**Critical:** All modules must use identical embedding configuration. This is now enforced by centralizing embedding creation in `core/embeddings.py::create_embeddings()`.

```python
# All modules use:
from core import create_embeddings
embeddings = create_embeddings()  # Guarantees consistent config
```

### Chunk Configuration
- `chunk_size=2000`: Balances keeping recipes together vs staying within model limits
- `chunk_overlap=200`: Prevents splitting mid-recipe
- Separators preserve recipe structure for better retrieval

### Error Handling
- `ingest.py`: Validates PDF existence, progress bars via `tqdm`, graceful failures
- `app.py`: Checks database existence, validates API key, handles search failures

### Model Name Format
- LangChain Google GenAI 4.x uses model names without `models/` prefix
- Use `gemini-flash-latest` not `models/gemini-1.5-flash`
- Available models queryable via Google GenAI API

## Common Issues & Solutions

### Python Version
**Problem:** Python 3.9 has `importlib.metadata` and package compatibility issues
**Solution:** Use Python 3.10 or higher

### Sentence-Transformers Error
**Problem:** `Cannot copy out of meta tensor` with v5.2.0
**Solution:** Pinned to v3.0.1 in requirements.txt

### Gemini 404 Errors
**Problem:** `models/gemini-*` not found
**Solution:** Use `gemini-flash-latest` (newer API versions)

### Import Errors
**Problem:** `ModuleNotFoundError: langchain.prompts`
**Solution:** Use `langchain_core.prompts` (LangChain 0.1.0+)

### Rate Limits
**Problem:** Google Embedding API 429 errors
**Solution:** Using local HuggingFace embeddings (already implemented)

### RAGAS AnswerRelevancy with Groq
**Problem:** `BadRequestError: 'n' : number must be at most 1`
**Solution:** Use `AnswerRelevancy(strictness=1)` - Groq only supports n=1, but AnswerRelevancy generates N questions by default

### RAGAS Collections Incompatibility
**Problem:** `ValueError: Collections metrics only support modern InstructorLLM. Found: ChatGroq`
**Solution:** Use legacy `ragas.metrics` API (not `ragas.metrics.collections`) - collections doesn't support LangChain wrappers

### RAGAS Timeout Errors
**Problem:** `TimeoutError()` during evaluation
**Solution:** Increase `RunConfig(timeout=300)` and reduce `max_workers=2` to avoid rate limits

## Design Rationale

### Why Local Embeddings?
- Google Embedding API has strict free tier limits (0 quota in our testing)
- Embeddings are bulk operations (expensive at scale with APIs)
- One-time compute cost, then unlimited local searches
- Trade-off: 384-dim vs Google's 768-dim, but sufficient for recipe matching

### Why Gemini for Generation?
- LLM queries are low-volume (per user search)
- Free tier covers ~1500 requests/day
- Excellent instruction following for recipe adaptation
- Latest model (2.5 Flash) is fast and capable

### Code Organization
- Configuration centralized in `core/config.py` (single source of truth)
- Shared retrieval algorithms in `core/retrieval.py` (cross-encoder model cached at module level)
- Shared embedding setup in `core/embeddings.py`
- Shared LLM functions in `core/llm.py` (initialize_llm, generate_recipe, prompt template)
- `app.py` contains only Streamlit UI logic (uses `@st.cache_resource` for performance)
- `ingest.py` contains only PDF processing and batch ingestion logic
- `evaluation.py` imports entirely from `core` (fully decoupled from `app.py`)
- Helper functions with comprehensive docstrings
- Main execution in `main()` function

## Dependencies

See `requirements.txt` for the full list. Critical version pins:
- `sentence-transformers==3.0.1` — **must** be pinned; v5.2.0 causes `Cannot copy out of meta tensor`
- `ragas` — use `ragas.metrics` (legacy API), NOT `ragas.metrics.collections` (requires InstructorLLM)
- `numpy<2.0` — required for compatibility with sentence-transformers 3.x

## File Purposes

- `core/config.py`: Centralized configuration constants (single source of truth)
- `core/retrieval.py`: RRF fusion, cross-encoder reranking (model cached), hybrid search, format_context
- `core/embeddings.py`: Embedding model creation, vectorstore initialization, hybrid retriever setup
- `core/llm.py`: LLM initialization, query expansion, recipe generation, prompt templates
- `core/__init__.py`: Package exports for clean imports
- `.streamlit/config.toml`: Soft Sage theme configuration (colors, font)
- `app.py`: PantryPal Streamlit UI with Space Grotesk font, centered layout, Soft Sage theme
- `ingest.py`: PDF processing, batch ingestion, deduplication (imports from `core`)
- `evaluation.py`: RAGAS evaluation framework, tests 4 retrieval methods (imports from `core` only)
- `tests/conftest.py`: Shared pytest fixtures (mock recipe data)
- `tests/test_retrieval.py`: Unit tests for RRF and format_context (11 tests)
- `pytest.ini`: Test configuration
- `requirements.txt`: Pinned dependencies
- `EVALUATION_RESULTS.md`: RAGAS evaluation findings and recommendations
- `.env`: API credentials (GOOGLE_API_KEY, GROQ_API_KEY - create manually, gitignored)
- `data/`: PDF cookbook storage (gitignored)
- `chroma_db/`: Vector database (gitignored)

## Testing

```bash
python -m pytest tests/ -v          # Run all 11 tests
```

**Test structure:**
- `tests/conftest.py`: Shared fixtures (mock Document objects, simulated retriever results)
- `tests/test_retrieval.py`: Unit tests for `reciprocal_rank_fusion()` and `format_context()`

**Quick verification after changes:**
```bash
python -c "from core import create_embeddings, initialize_llm, generate_recipe; print('OK')"
```

## Performance Characteristics

See `README.md` for full performance tables and cost breakdown. See `EVALUATION_RESULTS.md` for detailed RAGAS analysis.

**Key facts for development:**
- **Recommendation:** Use Hybrid + Reranking for production (best answer relevancy at 59.5%)
- **Reranking adds ~200ms** latency but +6.1% answer quality — worth the tradeoff
- **All costs are $0** — local embeddings, local vector DB, Gemini/Groq free tiers

## Future Considerations

- ~~Batch PDF ingestion~~ ✅ Implemented (folder support + deduplication)
- ~~Modular architecture~~ ✅ Implemented (`core/` module)
- ~~Unit tests~~ ✅ Implemented (pytest with 26 tests)
- ~~Decouple evaluation from app~~ ✅ Implemented (LLM functions moved to `core/llm.py`)
- ~~Performance caching~~ ✅ Implemented (`@st.cache_resource` + cross-encoder model cache)
- ~~FastAPI backend~~ ✅ Implemented (`api.py` with /health, /search, /generate endpoints)
- ~~Expand RAGAS test cases~~ ✅ Implemented (15 test cases across 5 categories)
- ~~UI redesign~~ ✅ Implemented (PantryPal branding, Space Grotesk, Soft Sage theme)
- GPU acceleration for embeddings (change `device: "cpu"` to `"cuda"` in `core/embeddings.py`)
- Recipe deduplication across cookbooks (content-level, not just file-level)
- Caching frequently requested recipes
- Integration tests for full pipeline
