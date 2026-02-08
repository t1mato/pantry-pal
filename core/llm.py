"""
Smart Pantry - LLM Functions

Business logic for LLM initialization and recipe generation.
Shared by app.py (Streamlit UI) and evaluation.py (RAGAS evaluation).

Separated from app.py so that evaluation can use these functions
without importing the Streamlit UI layer.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from core.config import GEMINI_MODEL, GEMINI_TEMPERATURE

load_dotenv()

# ============================================================================
# QUERY EXPANSION
# ============================================================================

QUERY_EXPANSION_TEMPLATE = """
Extract and expand the cooking-related keywords from this query.

QUERY: {query}

RULES:
1. Extract main ingredients mentioned
2. Add 1-2 synonyms for each ingredient (chicken→poultry, eggplant→aubergine)
3. Add 2-3 likely cooking methods or dish types
4. Output ONLY space-separated keywords, no sentences
5. Maximum 25 words

EXAMPLE:
Query: "I have chicken and rice, maybe some vegetables"
Output: chicken poultry rice grain vegetables veggies stir-fry fried-rice casserole curry

OUTPUT:"""


def expand_query(llm, query):
    """
    Expand a user's natural language query into search keywords.

    Uses the LLM to:
    - Extract key ingredients
    - Add synonyms (chicken → poultry)
    - Add related cooking methods (chicken + rice → stir-fry, curry)

    This improves retrieval by matching more vocabulary in the cookbook.

    Args:
        llm: Language model instance
        query (str): User's natural language query

    Returns:
        str: Expanded query with additional keywords
    """
    prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_TEMPLATE)
    formatted_prompt = prompt.format(query=query)

    response = llm.invoke(formatted_prompt)

    # Handle both response formats (same as generate_recipe)
    content = response.content
    if isinstance(content, list):
        expanded = content[0].get("text", "") if content else ""
    else:
        expanded = content

    # Clean up: remove extra whitespace, keep it concise
    expanded = " ".join(expanded.strip().split())

    # Combine original query with expanded keywords for best results
    # Original query preserves user intent, expanded adds vocabulary
    return f"{query} {expanded}"


# ============================================================================
# RECIPE GENERATION
# ============================================================================

RECIPE_PROMPT_TEMPLATE = """
You are a helpful cooking assistant. Based on the recipe context provided below,
create a recipe recommendation that uses the user's available ingredients and
respects their dietary restrictions.

RECIPE CONTEXT (from cookbooks):
{context}

USER'S QUERY:
{ingredients}

DIETARY RESTRICTIONS:
{restrictions}

INSTRUCTIONS:
1. First, check if the context contains an actual recipe (ingredients list + cooking steps)
2. If YES: recommend the best matching recipe, adapting it for dietary restrictions if needed
3. If the context only contains cookbook tips, general advice, or unrelated content:
   - Be honest that no matching recipe was found in the cookbook
   - Briefly explain what the cookbook does contain based on the context
   - Suggest the user try a different query
4. Always cite the source (cookbook and page number from the context metadata)
5. Be specific about measurements and cooking steps when a recipe exists
6. If restrictions make a recipe impossible (e.g., vegan cake with eggs), say so honestly

IMPORTANT: Always use the format below, even when no recipe is found.

FORMAT YOUR RESPONSE AS:

## [Recipe Name]

*From [Cookbook Name], page [X]*

---

### Ingredients

- [ingredient with measurement]
- [ingredient with measurement]
- ...

### Instructions

1. [First step]
2. [Second step]
3. ...

---

### Chef's Notes

- [Any substitutions made for dietary restrictions]
- [Tips or suggestions]
- **See also:** [If the context contains other potentially relevant recipes, list their names and page numbers here. If none, omit this line.]
"""


def initialize_llm():
    """
    Initialize Google Gemini LLM for recipe generation and adaptation.

    Requires GOOGLE_API_KEY in environment variables.

    Returns:
        ChatGoogleGenerativeAI: LLM instance

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        google_api_key=api_key
    )

    return llm


def generate_recipe(llm, context, ingredients, restrictions):
    """
    Generate a recipe recommendation using the LLM.

    Args:
        llm: Language model instance
        context (str): Recipe context from vector search
        ingredients (str): User's ingredients
        restrictions (str): Dietary restrictions

    Returns:
        str: Generated recipe recommendation
    """
    prompt = ChatPromptTemplate.from_template(RECIPE_PROMPT_TEMPLATE)

    formatted_prompt = prompt.format(
        context=context,
        ingredients=ingredients,
        restrictions=restrictions if restrictions else "None"
    )

    response = llm.invoke(formatted_prompt)

    # Handle both old (string) and new (list of content blocks) response formats
    content = response.content
    if isinstance(content, list):
        # New format: extract text from the first content block
        return content[0].get("text", "") if content else ""
    return content
