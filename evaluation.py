"""
Smart Pantry - RAGAS Evaluation Framework

Uses RAGAS (RAG Assessment) framework to evaluate retrieval quality.

Metrics evaluated:
- Context Precision: Are retrieved documents relevant to the query?
- Context Recall: Did we retrieve all relevant information?
- Faithfulness: Is the generated answer grounded in retrieved context?
- Answer Relevance: Does the answer actually address the query?

Compares:
- Semantic Search only
- BM25 Search only
- Hybrid (BM25 + Semantic + RRF)
- Hybrid + Cross-Encoder Reranking

Usage:
    python evaluation.py [--verbose | --quiet]
"""

import argparse
import logging
import os
import sys
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# RAGAS imports
# NOTE: Using legacy ragas.metrics (not collections) because collections requires InstructorLLM
# Collections doesn't support LangChain's ChatGroq wrapper
from ragas import evaluate, RunConfig
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy
)
from datasets import Dataset

load_dotenv()

# Import shared utilities from core module
from core import (
    # Config
    CHROMA_PATH,
    GEMINI_MODEL,
    EMBEDDING_MODEL,

    # Embeddings & vectorstore
    initialize_vectorstore,
    initialize_hybrid_retriever,

    # Retrieval
    reciprocal_rank_fusion,
    rerank_with_cross_encoder,
    format_context,

    # LLM (now in core/llm.py — no longer imports from app.py!)
    initialize_llm,
    generate_recipe,
)

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# TEST DATASET
# ============================================================================

@dataclass
class TestCase:
    """A test case for RAG evaluation."""
    question: str  # User query
    ground_truth: str  # Expected answer content (for context_recall)
    relevant_keywords: List[str]  # Keywords that should appear in results


# Expanded test cases for statistically meaningful evaluation
# 15 test cases × 4 methods = 60 evaluations
# Covers: ingredients, dietary restrictions, budget, cuisine, meal types
#
# Ground truths are based on actual "Good and Cheap" cookbook content
TEST_CASES = [
    # === SPECIFIC INGREDIENT QUERIES ===
    TestCase(
        question="I have chicken and rice. What can I make for dinner?",
        ground_truth="Filipino Chicken Adobo is a savory chicken dish that pairs well with rice. The cookbook features this as a main dinner recipe.",
        relevant_keywords=["chicken", "adobo", "rice"]
    ),
    TestCase(
        question="What can I make with eggs for breakfast?",
        ground_truth="Tomato Scrambled Eggs, omelettes, or egg sandwiches. The cookbook emphasizes eggs as a versatile, budget-friendly protein.",
        relevant_keywords=["eggs", "scrambled", "omelette", "breakfast"]
    ),
    TestCase(
        question="I have zucchini that's about to go bad. Any recipes?",
        ground_truth="Creamy Zucchini Fettuccine for dinner or Chocolate Zucchini Muffins for a sweet treat. Both use fresh zucchini.",
        relevant_keywords=["zucchini", "fettuccine", "muffins"]
    ),
    TestCase(
        question="What recipes use eggplant?",
        ground_truth="Pasta with Eggplant and Tomato is a vegetarian dinner option featuring roasted or sautéed eggplant with pasta.",
        relevant_keywords=["eggplant", "pasta", "tomato"]
    ),

    # === DIETARY RESTRICTION QUERIES ===
    TestCase(
        question="What vegetarian recipes use beans for protein?",
        ground_truth="Black-Eyed Peas and Collards, Baked Beans, or Chana Masala (chickpeas). These are hearty vegetarian mains.",
        relevant_keywords=["beans", "vegetarian", "protein", "chana", "black-eyed"]
    ),
    TestCase(
        question="What vegan-friendly dinner options are there?",
        ground_truth="Vegetable Jambalaya, Chana Masala, and Pasta with Eggplant and Tomato can all be made vegan.",
        relevant_keywords=["vegan", "vegetable", "jambalaya", "chana"]
    ),
    TestCase(
        question="I need a meatless Monday dinner idea.",
        ground_truth="Creamy Zucchini Fettuccine, Black-Eyed Peas and Collards, or Half-Veggie Burgers are satisfying meatless options.",
        relevant_keywords=["meatless", "vegetarian", "veggie", "burgers"]
    ),

    # === BUDGET/FRUGAL QUERIES ===
    TestCase(
        question="I'm on a tight budget and have pasta. What's a cheap meal?",
        ground_truth="Pasta with Eggplant and Tomato or Creamy Zucchini Fettuccine. The cookbook is designed for $4/day budgets.",
        relevant_keywords=["pasta", "budget", "cheap", "eggplant", "zucchini"]
    ),
    TestCase(
        question="What's the cheapest filling meal I can make?",
        ground_truth="Oatmeal for breakfast, bean-based dishes like Baked Beans or Black-Eyed Peas for dinner. Eggs are also very economical.",
        relevant_keywords=["cheap", "budget", "oatmeal", "beans", "eggs"]
    ),
    TestCase(
        question="How can I eat well on food stamps?",
        ground_truth="The Good and Cheap cookbook is specifically designed for SNAP recipients living on $4/day. Focus on eggs, beans, and seasonal vegetables.",
        relevant_keywords=["budget", "food stamps", "SNAP", "cheap"]
    ),

    # === CUISINE-TYPE QUERIES ===
    TestCase(
        question="Do you have any Indian-inspired recipes?",
        ground_truth="Chana Masala is an Indian chickpea curry. The cookbook also mentions raita and roti as accompaniments.",
        relevant_keywords=["indian", "chana", "masala", "curry", "roti"]
    ),
    TestCase(
        question="What Asian recipes are in the cookbook?",
        ground_truth="Filipino Chicken Adobo is a tangy, savory Filipino dish. The cookbook includes various Asian-influenced recipes.",
        relevant_keywords=["asian", "filipino", "adobo", "chicken"]
    ),

    # === MEAL-TYPE QUERIES ===
    TestCase(
        question="What quick breakfast can I make in 10 minutes?",
        ground_truth="Tomato Scrambled Eggs or a simple omelette. Both can be made quickly with basic pantry ingredients.",
        relevant_keywords=["breakfast", "quick", "eggs", "scrambled", "omelette"]
    ),
    TestCase(
        question="I need healthy snack ideas for kids.",
        ground_truth="Peanut Butter and Jelly Granola Bars or Chocolate Zucchini Muffins are kid-friendly, portable snacks.",
        relevant_keywords=["snack", "granola", "bars", "muffins", "kids"]
    ),
    TestCase(
        question="What's a good comfort food for a cold day?",
        ground_truth="French Onion Soup is a warm, comforting classic. Oatmeal also makes a cozy, warming meal.",
        relevant_keywords=["comfort", "soup", "onion", "warm", "oatmeal"]
    ),
]


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================

def retrieve_semantic_only(vectorstore: Chroma, query: str, k: int = 5) -> List[Document]:
    """Pure semantic search using embeddings."""
    return vectorstore.similarity_search(query, k=k)


def retrieve_bm25_only(bm25_retriever: BM25Retriever, query: str, k: int = 5) -> List[Document]:
    """Pure BM25 keyword search."""
    bm25_retriever.k = k * 2  # Get more for better fusion
    return bm25_retriever.invoke(query)[:k]


def retrieve_hybrid_rrf(bm25_retriever: BM25Retriever, semantic_retriever, query: str, k: int = 5) -> List[Document]:
    """Hybrid search: BM25 + Semantic with RRF."""
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)
    fused = reciprocal_rank_fusion([bm25_results, semantic_results])
    return fused[:k]


def retrieve_hybrid_with_reranking(bm25_retriever: BM25Retriever, semantic_retriever, query: str, k: int = 5) -> List[Document]:
    """Hybrid + Cross-Encoder: BM25 + Semantic with RRF, then cross-encoder reranking."""
    # First get hybrid results (retrieve more for reranking)
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)
    fused = reciprocal_rank_fusion([bm25_results, semantic_results])

    # Get top 10 candidates for reranking
    candidates = fused[:10]

    # Rerank with cross-encoder
    return rerank_with_cross_encoder(query, candidates, top_k=k)


# ============================================================================
# RAGAS EVALUATION
# ============================================================================

def create_ragas_dataset(test_cases: List[TestCase], retrieval_method, llm, method_name: str) -> Dataset:
    """
    Create a RAGAS-compatible dataset from test cases.

    RAGAS expects:
    - question: User query
    - answer: Generated answer from RAG
    - contexts: Retrieved documents (list of strings)
    - ground_truth: Expected answer (for context_recall)
    """
    logger.info(f"🔍 Generating dataset for: {method_name}")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"   [{i}/{len(test_cases)}] Processing: \"{test_case.question[:50]}...\"")

        # Retrieve documents
        retrieved_docs = retrieval_method(test_case.question)

        # Format context for LLM
        context_str = format_context(retrieved_docs)

        # Generate answer using LLM
        answer = generate_recipe(
            llm,
            context_str,
            test_case.question,
            restrictions=""  # No restrictions for evaluation
        )

        # Ensure answer is a string (not list or other type)
        if not isinstance(answer, str):
            answer = str(answer)

        # Prepare for RAGAS
        questions.append(test_case.question)
        answers.append(answer)
        contexts_list.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(test_case.ground_truth)

    # Create dataset using pandas to handle nested lists better
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }

    # Convert to DataFrame first, then to Dataset (handles nested lists better)
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)


def evaluate_retrieval_method(
    method_name: str,
    retrieval_func,
    vectorstore: Chroma,
    llm,
    test_cases: List[TestCase]
) -> Dict:
    """
    Evaluate a single retrieval method using RAGAS.

    Returns dict of metric scores.
    """
    logger.info("=" * 80)
    logger.info(f"📊 Evaluating: {method_name}")
    logger.info("=" * 80)

    # Create dataset
    dataset = create_ragas_dataset(test_cases, retrieval_func, llm, method_name)

    logger.info("🧪 Running RAGAS evaluation...")
    logger.info("   This may take a few minutes (using LLM for metrics)...")

    # Configure RAGAS with extended timeouts and better error handling
    run_config = RunConfig(
        timeout=300,  # 5 minutes per operation (increased from default 180s)
        max_retries=3,  # Retry up to 3 times on failures
        max_wait=30,  # Wait up to 30s between retries
        max_workers=2,  # Limit concurrent workers to avoid rate limits
        log_tenacity=False  # Disable retry logging for cleaner output
    )

    # Initialize all 4 RAGAS metrics for comprehensive RAG evaluation
    # Using legacy API (ragas.metrics) - LLM is passed to evaluate(), not to metrics
    # CRITICAL: Set strictness=1 for AnswerRelevancy to work with Groq
    # Groq only supports n=1, and AnswerRelevancy generates N questions by default (default=3)
    # See: https://github.com/explodinggradients/ragas/issues/1072
    metrics = [
        ContextPrecision(),             # Retrieval: Are retrieved docs relevant?
        ContextRecall(),                # Retrieval: Did we retrieve all relevant info?
        Faithfulness(),                 # Generation: Is answer grounded in context?
        AnswerRelevancy(strictness=1)   # Generation: strictness=1 for Groq compatibility
    ]

    # Run RAGAS evaluation
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        run_config=run_config,
        raise_exceptions=False  # Continue even if some metrics fail
    )

    logger.info(f"✅ Evaluation complete for {method_name}")

    return results


def compare_methods(all_results):
    """Print comparison table of all methods."""
    logger.info("=" * 100)
    logger.info("📊 RAGAS EVALUATION RESULTS - COMPARISON")
    logger.info("=" * 100)

    # Extract metrics
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    # Convert EvaluationResult objects to metric dictionaries
    results_dict = {}
    for method_name, eval_result in all_results.items():
        # Convert to DataFrame and get mean scores
        df = eval_result.to_pandas()
        results_dict[method_name] = {metric: df[metric].mean() for metric in metrics}

    # Create comparison table header
    header = f"{'Method':<25}"
    for metric in metrics:
        header += f"{metric:<20}"
    logger.info(header)
    logger.info("-" * 100)

    # Log each method's results
    for method_name, scores in results_dict.items():
        row = f"{method_name:<25}"
        for metric in metrics:
            value = scores.get(metric, 0.0)
            row += f"{value:.4f}{' '*15}"
        logger.info(row)

    logger.info("=" * 100)

    # Calculate and display best performer for each metric
    logger.info("🏆 Best Performers:")
    for metric in metrics:
        best_method = max(results_dict.items(), key=lambda x: x[1].get(metric, 0))
        best_value = best_method[1].get(metric, 0)
        logger.info(f"   {metric}: {best_method[0]} ({best_value:.4f})")


def save_results_to_csv(all_results, filename: str = "evaluation_results.csv"):
    """Save results to CSV for easy analysis."""
    # Convert EvaluationResult objects to metric dictionaries
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    results_dict = {}
    for method_name, eval_result in all_results.items():
        # Convert to DataFrame and get mean scores
        df_temp = eval_result.to_pandas()
        results_dict[method_name] = {metric: df_temp[metric].mean() for metric in metrics}

    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T
    df.index.name = "Method"

    # Save to CSV
    df.to_csv(filename)
    logger.info(f"💾 Results saved to: {filename}")

    # Also save detailed report
    report_filename = filename.replace('.csv', '_report.txt')
    with open(report_filename, 'w') as f:
        f.write("SMART PANTRY - RAGAS EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")

        f.write("RESULTS:\n")
        f.write(df.to_string())
        f.write("\n\n")

        f.write("BEST PERFORMERS:\n")
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        for metric in metrics:
            # Use DataFrame instead of EvaluationResult objects
            if metric in df.columns:
                best_method = df[metric].idxmax()
                best_value = df.loc[best_method, metric]
                if not pd.isna(best_value):
                    f.write(f"  {metric}: {best_method} ({best_value:.4f})\n")
                else:
                    f.write(f"  {metric}: No valid results\n")
            else:
                f.write(f"  {metric}: Metric not found\n")

    logger.info(f"📄 Report saved to: {report_filename}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation framework on Smart Pantry retrieval methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation.py              # Run evaluation with default output
  python evaluation.py --verbose    # Show detailed debug output
  python evaluation.py --quiet      # Only show errors
        """
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only show errors"
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",  # Simple format — just the message, no timestamps
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """Run the RAGAS evaluation framework."""
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    logger.info("=" * 80)
    logger.info("🧪 SMART PANTRY - RAGAS EVALUATION FRAMEWORK")
    logger.info("=" * 80)

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        logger.error(f"❌ Vector database not found at {CHROMA_PATH}")
        logger.error("   Run 'python ingest.py' first to create it.")
        sys.exit(1)

    # Initialize components
    logger.info("📚 Initializing components...")
    vectorstore = initialize_vectorstore()
    bm25_retriever, semantic_retriever = initialize_hybrid_retriever(vectorstore)

    # Use Groq API for fast, reliable evaluation
    logger.debug("   Initializing Groq LLM (llama-3.1-8b-instant)...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Efficient 8B model for faster evaluation with lower token usage
        temperature=0,  # Deterministic for evaluation
        max_tokens=4096,  # Sufficient for recipe generation
        # Note: Groq only supports n=1 by default, no need to specify
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    logger.info("✓ Vector database loaded")
    logger.info("✓ BM25 and Semantic retrievers ready")
    logger.info("✓ LLM initialized (Groq Llama-3.1-8B, n=1 for RAGAS compatibility)")

    # Define retrieval methods to evaluate
    methods = {
        "Semantic Only": lambda q: retrieve_semantic_only(vectorstore, q),
        "BM25 Only": lambda q: retrieve_bm25_only(bm25_retriever, q),
        "Hybrid (RRF)": lambda q: retrieve_hybrid_rrf(bm25_retriever, semantic_retriever, q),
        "Hybrid + Reranking": lambda q: retrieve_hybrid_with_reranking(bm25_retriever, semantic_retriever, q)
    }

    logger.info(f"📋 Test cases: {len(TEST_CASES)}")
    logger.info(f"🔬 Methods to evaluate: {len(methods)}")
    logger.info("   1. Semantic Only (baseline)")
    logger.info("   2. BM25 Only (keyword)")
    logger.info("   3. Hybrid (RRF)")
    logger.info("   4. Hybrid + Cross-Encoder Reranking ⭐")

    # Run evaluation for each method
    all_results = {}

    for method_name, retrieval_func in methods.items():
        try:
            results = evaluate_retrieval_method(
                method_name,
                retrieval_func,
                vectorstore,
                llm,
                TEST_CASES
            )
            all_results[method_name] = results

        except Exception as e:
            logger.error(f"❌ Error evaluating {method_name}: {e}")
            logger.exception("Traceback:")

    # Display comparison
    if all_results:
        compare_methods(all_results)
        save_results_to_csv(all_results)

    logger.info("✅ Evaluation complete!")
    logger.info("")
    logger.info("💡 Next steps:")
    logger.info("   1. Review the results above")
    logger.info("   2. Check evaluation_results.csv for raw data")
    logger.info("   3. Compare methods and tune if needed")
    logger.info("")
    logger.info("📖 Learn more about RAGAS: https://docs.ragas.io/")


if __name__ == "__main__":
    main()
