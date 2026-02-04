"""
Smart Pantry & Diet Guardian - Streamlit Application (Hybrid Search Edition)

This app helps you find recipes based on ingredients you have at home.
It retrieves recipes from your local PDF cookbook database and adapts them
to your dietary restrictions using AI.

Features:
- 🔄 **Hybrid Search**: Combines BM25 (keyword) + Semantic (meaning) search
- 🎯 **RRF Fusion**: Reciprocal Rank Fusion for optimal result ranking
- 🍳 **Ingredient-based recipe search**
- 🥗 **Dietary restriction filtering**
- 📚 **Source transparency** (shows which cookbook/page)
"""

import streamlit as st

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

def search_recipes_hybrid(bm25_retriever, semantic_retriever, ingredients, restrictions,
                         num_results=NUM_RESULTS, use_reranking=False):
    """
    Search for recipes using hybrid retrieval.

    This is a thin wrapper around core.search_hybrid that:
    1. Constructs a recipe-specific query from ingredients + restrictions
    2. Delegates to the core search function

    Args:
        bm25_retriever: BM25 keyword retriever
        semantic_retriever: Semantic vector retriever
        ingredients (str): User's available ingredients
        restrictions (str): Dietary restrictions
        num_results (int): Number of results to retrieve
        use_reranking (bool): Whether to apply cross-encoder reranking

    Returns:
        list: List of Document objects with recipe chunks and metadata
    """
    # Construct recipe-specific search query
    query = f"Recipes using: {ingredients}"
    if restrictions:
        query += f" that can be made {restrictions}"

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
        page_title="🍳 Smart Pantry & Diet Guardian",
        page_icon="🍳",
        layout="wide"
    )

    # Header
    st.title("🍳 Smart Pantry & Diet Guardian")
    st.markdown(
        "Find delicious recipes based on what's in your pantry, "
        "adapted to your dietary needs. All recipes are sourced from real cookbooks."
    )

    # Sidebar for configuration and info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            This app uses **Hybrid Search**:
            - 🔤 **BM25**: Keyword matching for exact ingredients
            - 🧠 **Semantic**: Understands meaning & synonyms
            - 🎯 **RRF Fusion**: Combines both for best results
            - 🤖 **AI Adaptation**: Google Gemini adapts recipes
            """
        )

        st.divider()

        st.header("⚙️ Settings")
        num_results = st.slider(
            "Number of recipes to search",
            min_value=1,
            max_value=10,
            value=NUM_RESULTS,
            help="More results = more context but slower"
        )

        st.divider()

        # Cross-encoder reranking toggle
        use_reranking = st.checkbox(
            "🎯 Enable Cross-Encoder Reranking",
            value=True,
            help="Uses a more powerful model to rerank results for better precision. Adds ~100ms latency."
        )

        if use_reranking:
            st.info(
                "**Cross-Encoder is ON** ✅\n\n"
                "Results are reranked using a specialized model "
                "for improved relevance."
            )

        st.divider()

        # Debug mode - retrieval only (no LLM)
        debug_mode = st.checkbox(
            "🔧 Debug Mode (Retrieval Only)",
            value=False,
            help="Show retrieved documents without calling Gemini API. Useful for testing retrieval quality without using API quota."
        )

        if debug_mode:
            st.warning(
                "**Debug Mode is ON** 🔧\n\n"
                "Will show retrieved documents only (no recipe generation). "
                "Use this to compare cross-encoder ON vs OFF without using API quota."
            )


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

    # Main input form
    st.divider()
    st.header("🥘 What's in your pantry?")

    col1, col2 = st.columns(2)

    with col1:
        ingredients = st.text_area(
            "Available Ingredients",
            placeholder="e.g., chicken breast, rice, bell peppers, onion",
            height=100,
            help="List the ingredients you have available"
        )

    with col2:
        restrictions = st.text_area(
            "Dietary Restrictions (Optional)",
            placeholder="e.g., vegetarian, gluten-free, no cilantro",
            height=100,
            help="Any dietary restrictions or preferences"
        )

    # Search button
    search_button = st.button("🔍 Find Recipes", type="primary", use_container_width=True)

    # Process search
    if search_button:
        if not ingredients.strip():
            st.warning("⚠️ Please enter at least one ingredient.")
            return

        # Show search status
        search_message = "🔍 Searching cookbook database with hybrid search (BM25 + Semantic + RRF)"
        if use_reranking:
            search_message += " + Cross-Encoder Reranking"
        search_message += "..."

        with st.spinner(search_message):
            search_results = search_recipes_hybrid(
                bm25_retriever,
                semantic_retriever,
                ingredients,
                restrictions,
                num_results=num_results,
                use_reranking=use_reranking
            )

        if not search_results:
            st.warning("😔 No recipes found. Try different ingredients or add more cookbooks.")
            return

        # Debug mode: Show retrieved documents only (no LLM generation)
        if debug_mode:
            st.divider()
            st.header("🔍 Retrieved Recipe Chunks (Debug Mode)")
            st.info(f"**Found {len(search_results)} relevant chunks**. Toggle cross-encoder ON/OFF to compare results.")

            for i, doc in enumerate(search_results, 1):
                with st.container():
                    st.markdown(f"### 📄 Rank #{i}")
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")
                    st.markdown("**Content:**")
                    st.text_area(
                        f"Chunk {i}",
                        doc.page_content,
                        height=200,
                        key=f"doc_{i}",
                        label_visibility="collapsed"
                    )
                    st.divider()

        # Normal mode: Generate recipe with LLM
        else:
            # Format context
            context = format_context(search_results)

            # Generate recipe
            with st.spinner("🤖 Adapting recipe to your needs..."):
                recipe = generate_recipe(llm, context, ingredients, restrictions)

            # Display result
            st.divider()
            st.header("📖 Your Recipe")
            st.markdown(recipe)

            # Show retrieved context in expander
            with st.expander("🔍 View Source Recipe Chunks"):
                st.markdown("*These are the cookbook excerpts used to generate your recipe:*")
                for i, doc in enumerate(search_results, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')} "
                              f"(Page {doc.metadata.get('page', 'Unknown')})*")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.divider()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
