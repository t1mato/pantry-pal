"""
Smart Pantry - AI Recipe Assistant

A chatbot-style recipe finder powered by hybrid search (BM25 + Semantic)
with cross-encoder reranking. Ask naturally about what you want to cook!
"""

import streamlit as st

# ============================================================================
# MINIMAL STYLING (theme colors in .streamlit/config.toml)
# ============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    #MainMenu, header, footer { visibility: hidden; }

    /* Center the page horizontally and vertically */
    .block-container {
        max-width: 600px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 15vh !important;
        padding-bottom: 5vh !important;
    }

    /* Title and subtitle */
    h1 { text-align: center !important; font-size: 5rem !important; color: #5a7d5a !important; margin-bottom: 0.25rem !important; }
    [data-testid="stCaptionContainer"] { font-size: 1.2rem !important; text-align: center !important; width: 100% !important; margin-bottom: 1rem !important; }
    .stCaption { font-size: 1rem !important; }

    /* Form labels */
    .label { font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.5px; color: #6b7d6b; margin-bottom: 0.5rem; text-align: center; }

    /* Input fields */
    .stTextArea textarea, .stTextInput input { font-size: 1.1rem !important; }
    .stButton button { font-size: 1.1rem !important; padding: 0.75rem 1.5rem !important; }

    @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }

    .skeleton { background: linear-gradient(90deg, #e8ede8 25%, #f0f4f0 50%, #e8ede8 75%);
                background-size: 200% 100%; animation: shimmer 1.5s infinite;
                border-radius: 0.25rem; height: 1rem; margin: 0.5rem 0; }
    .skeleton-title { height: 1.25rem; width: 50%; margin-bottom: 1rem; }
    .skeleton-short { width: 75%; }
</style>
"""

SKELETON_HTML = """
<div style="padding:1.5rem; background:#f0f4f0; border-radius:0.5rem; border:1px solid #dce5dc;">
    <div class="skeleton skeleton-title"></div>
    <div class="skeleton"></div>
    <div class="skeleton skeleton-short"></div>
    <div style="height:1rem;"></div>
    <div class="skeleton"></div>
    <div class="skeleton skeleton-short"></div>
</div>
"""

# Import shared utilities from core module
from core import (
    # Config
    NUM_RESULTS,

    # Embeddings & vectorstore
    initialize_vectorstore,
    initialize_hybrid_retriever,

    # Retrieval
    search_hybrid,
    format_context,

    # LLM (shared with evaluation.py via core/)
    initialize_llm,
    generate_recipe,
)

# ============================================================================
# CACHED INITIALIZATION (persist across Streamlit re-renders)
# ============================================================================

@st.cache_resource
def get_search_components():
    """Load vectorstore and build hybrid retrievers (cached across re-renders)."""
    vectorstore = initialize_vectorstore()
    bm25_retriever, semantic_retriever = initialize_hybrid_retriever(vectorstore)
    return bm25_retriever, semantic_retriever


@st.cache_resource
def get_llm():
    """Initialize the Gemini LLM client (cached across re-renders)."""
    return initialize_llm()


# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def search_recipes_hybrid(bm25_retriever, semantic_retriever, user_query, restrictions,
                         num_results=NUM_RESULTS, use_reranking=False):
    """
    Search for recipes using hybrid retrieval.

    This is a thin wrapper around core.search_hybrid that:
    1. Combines user query with dietary restrictions (if any)
    2. Delegates to the core search function

    Args:
        bm25_retriever: BM25 keyword retriever
        semantic_retriever: Semantic vector retriever
        user_query (str): Natural language query from user
        restrictions (str): Dietary restrictions
        num_results (int): Number of results to retrieve
        use_reranking (bool): Whether to apply cross-encoder reranking

    Returns:
        list: List of Document objects with recipe chunks and metadata
    """
    # Use user's natural language query directly
    query = user_query
    if restrictions:
        query += f" (dietary restrictions: {restrictions})"

    # Delegate to core search function
    return search_hybrid(
        bm25_retriever,
        semantic_retriever,
        query,
        num_results=num_results,
        use_reranking=use_reranking
    )


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="PantryPal",
        layout="centered"
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Simple header
    st.title("PantryPal")
    st.caption("Make the most of your pantry with a click of a button.")

    # Sidebar - Minimal
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        Smart Pantry searches real cookbooks using AI to find and adapt recipes based on your ingredients.
        """)

        st.markdown("---")

        st.caption("Built by [t1mato](https://github.com/t1mato)")

    # Settings: use sensible defaults (no UI clutter)
    num_results = NUM_RESULTS  # Default: 5
    use_reranking = True       # Always on — better results
    debug_mode = st.query_params.get("debug", "false").lower() == "true"  # Hidden: ?debug=true


    # Initialize components (cached — only runs once, not on every re-render)
    try:
        bm25_retriever, semantic_retriever = get_search_components()

        # Only initialize LLM if not in debug mode
        if not debug_mode:
            llm = get_llm()
        else:
            llm = None  # Not needed in debug mode

    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("👉 Run `python ingest.py` to create the recipe database first.")
        st.stop()

    except ValueError as e:
        st.error(f"❌ {e}")
        st.info("👉 Add your Google API key to the `.env` file:\n```\nGOOGLE_API_KEY=your_key_here\n```")
        st.stop()

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

    # Main query input
    st.markdown('<p class="label">Ingredients</p>', unsafe_allow_html=True)
    user_query = st.text_area(
        "Ingredients",
        placeholder="I have chicken, rice, and some vegetables...",
        height=100,
        label_visibility="collapsed"
    )

    # Dietary restrictions
    st.markdown('<p class="label">Dietary Restrictions</p>', unsafe_allow_html=True)
    restrictions = st.text_input(
        "Dietary Restrictions",
        placeholder="e.g. vegetarian, gluten-free, no dairy",
        label_visibility="collapsed"
    )

    st.write("")  # Spacing
    search_button = st.button("Find recipes", type="primary", use_container_width=True)

    # Process search
    if search_button:
        if not user_query.strip():
            st.warning("⚠️ Please enter a question or describe what you'd like to cook.")
            return

        # Show skeleton while loading
        skeleton_placeholder = st.empty()
        skeleton_placeholder.markdown(SKELETON_HTML, unsafe_allow_html=True)

        search_results = search_recipes_hybrid(
            bm25_retriever,
            semantic_retriever,
            user_query,
            restrictions,
            num_results=num_results,
            use_reranking=use_reranking
        )

        skeleton_placeholder.empty()

        if not search_results:
            st.warning("😔 No recipes found. Try different ingredients or add more cookbooks.")
            return

        # Debug mode: Show retrieved documents only (no LLM generation)
        if debug_mode:
            st.divider()
            st.subheader(f"🔍 Debug: {len(search_results)} chunks retrieved")

            for i, doc in enumerate(search_results, 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                page = doc.metadata.get('page', '?')
                with st.expander(f"**#{i}** — {source}, p.{page}", expanded=(i == 1)):
                    st.text(doc.page_content)

        # Normal mode: Generate recipe with LLM
        else:
            # Format context
            context = format_context(search_results)

            # Generate recipe with skeleton loading
            recipe_placeholder = st.empty()
            recipe_placeholder.markdown(SKELETON_HTML, unsafe_allow_html=True)
            recipe = generate_recipe(llm, context, user_query, restrictions)
            recipe_placeholder.empty()

            # Display result
            st.markdown("---")
            st.markdown(recipe)

            # Show sources in expander
            with st.expander("📚 Sources from cookbook"):
                for i, doc in enumerate(search_results, 1):
                    source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    page = doc.metadata.get('page', '?')
                    st.caption(f"**[{i}]** {source}, page {page}")
                    st.code(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content, language=None)

    # Footer
    st.write("")
    st.caption("[GitHub](https://github.com/t1mato/smart-pantry)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
