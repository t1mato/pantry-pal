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

# Prompt template for recipe generation
RECIPE_PROMPT_TEMPLATE = """
You are a helpful cooking assistant. Based on the recipe context provided below,
create a recipe recommendation that uses the user's available ingredients and
respects their dietary restrictions.

RECIPE CONTEXT (from cookbooks):
{context}

USER'S INGREDIENTS:
{ingredients}

DIETARY RESTRICTIONS:
{restrictions}

INSTRUCTIONS:
1. If the context contains recipes that match the ingredients, recommend the best one
2. Adapt the recipe if needed to respect dietary restrictions
3. If no exact match exists, suggest the closest recipe and explain substitutions
4. Always cite the source (cookbook and page number from the context metadata)
5. Be specific about measurements and cooking steps
6. If restrictions make a recipe impossible (e.g., vegan cake with eggs), say so honestly

FORMAT YOUR RESPONSE AS:
📖 **Recipe Name**
🏷️ *Source: [Cookbook name, Page X]*

**Ingredients:**
- [List ingredients with measurements]

**Instructions:**
1. [Step-by-step instructions]

**Notes:**
- [Any substitutions made for dietary restrictions]
- [Tips or warnings]
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

    return response.content
