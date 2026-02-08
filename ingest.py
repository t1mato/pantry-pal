"""
Smart Pantry - Document Ingestion Script

This script processes PDF cookbooks and stores them in a local vector database (ChromaDB).
It uses local embeddings (HuggingFace) to avoid API rate limits and costs.

Supports both single file and batch ingestion with duplicate detection.

Usage:
    python ingest.py [path] [--verbose | --quiet]

    path can be:
    - A single PDF file: python ingest.py data/my-cookbook.pdf
    - A directory: python ingest.py data/   (processes all PDFs in folder)
    - Omitted: Uses default PDF

Examples:
    python ingest.py data/my-cookbook.pdf     # Single file
    python ingest.py data/                   # All PDFs in folder
    python ingest.py                         # Default PDF
    python ingest.py --verbose               # Show debug output
    python ingest.py --quiet                 # Only show errors
"""

import argparse
import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

# Import shared config from core module
from core import (
    DEFAULT_PDF_PATH,
    CHROMA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    BATCH_SIZE,
    create_embeddings,
)

# Load environment variables (for future API keys if needed)
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def discover_pdfs(input_path):
    """
    Discover PDF files from a file path or directory.

    This enables batch ingestion: pass a folder and all PDFs are found.

    Args:
        input_path (str): Path to a PDF file or directory containing PDFs

    Returns:
        list[Path]: List of validated PDF file paths

    Raises:
        FileNotFoundError: If the path doesn't exist
        ValueError: If no PDFs are found
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if path.is_file():
        # Single file mode
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {input_path}")
        return [path]

    if path.is_dir():
        # Batch mode — find all PDFs in directory
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"No PDF files found in directory: {input_path}")
        logger.info(f"📁 Found {len(pdfs)} PDF(s) in {input_path}")
        for pdf in pdfs:
            logger.info(f"   - {pdf.name}")
        return pdfs

    raise ValueError(f"Path is neither a file nor a directory: {input_path}")


def validate_pdf_path(pdf_path):
    """
    Validates that the PDF file exists and is readable.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        Path: Validated Path object

    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If file is not a PDF
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")

    return path


def get_ingested_sources():
    """
    Check which PDF sources are already in the vector database.

    Reads metadata from ChromaDB to determine which files have been
    ingested. This prevents duplicate ingestion when re-running the script.

    Returns:
        set[str]: Set of source filenames already in the database.
                  Empty set if database doesn't exist yet.
    """
    if not Path(CHROMA_PATH).exists():
        return set()

    try:
        embeddings = create_embeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        # Get all metadata from the database
        all_metadatas = vectorstore.get()['metadatas']
        # Extract unique source filenames
        sources = {meta.get('source', '') for meta in all_metadatas}
        return sources
    except Exception as e:
        logger.warning(f"Could not read existing database, starting fresh: {e}")
        return set()


def load_pdf_documents(pdf_path):
    """
    Loads a PDF file and extracts text content page by page.

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        list: List of Document objects (one per page)

    Raises:
        RuntimeError: If PDF loading fails
    """
    logger.info(f"📖 Loading PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        logger.info(f"✓ Loaded {len(docs)} pages")
        return docs

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF {pdf_path}") from e


def split_documents(docs):
    """
    Splits documents into smaller chunks optimized for recipe retrieval.

    The splitter:
    1. Tries to split on double newlines first (paragraph boundaries)
    2. Then tries recipe-specific markers (Title:, Ingredients:)
    3. Falls back to character count if needed
    4. Overlaps chunks to avoid cutting recipes in half

    Args:
        docs (list): List of Document objects

    Returns:
        list: List of smaller Document chunks
    """
    logger.info("✂️  Splitting documents into chunks...")
    logger.debug(f"   Chunk size: {CHUNK_SIZE} characters")
    logger.debug(f"   Overlap: {CHUNK_OVERLAP} characters")
    logger.debug(f"   Separators: {CHUNK_SEPARATORS}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS
    )

    splits = text_splitter.split_documents(docs)
    logger.info(f"✓ Created {len(splits)} chunks")

    return splits


def initialize_embeddings():
    """
    Initialize the local embedding model using the shared core configuration.

    This is a thin wrapper around core.create_embeddings() that adds
    user-friendly console output for the ingestion process.

    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    """
    logger.info("🤖 Initializing local embedding model...")
    logger.debug("   (First run will download ~120MB model)")

    # Use shared embedding configuration from core
    embeddings = create_embeddings()

    logger.info("✓ Embedding model ready")
    return embeddings


def store_in_vectordb(splits, embeddings):
    """
    Stores document chunks in ChromaDB with embeddings.

    Uses add_documents() to append to an existing database instead of
    overwriting. This enables batch ingestion of multiple PDFs.

    Args:
        splits (list): Document chunks to embed and store
        embeddings (HuggingFaceEmbeddings): Embedding model

    Returns:
        Chroma: Vector store instance
    """
    logger.info(f"💾 Storing chunks in vector database: {CHROMA_PATH}")
    logger.debug(f"   Processing {len(splits)} chunks in batches of {BATCH_SIZE}...")

    try:
        # Connect to existing database or create new one
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        # Append documents (doesn't overwrite existing data)
        with tqdm(total=len(splits), desc="Embedding & storing", unit="chunk") as pbar:
            vectorstore.add_documents(documents=splits)
            pbar.update(len(splits))

        logger.info(f"✓ Successfully stored {len(splits)} chunks")
        return vectorstore

    except Exception as e:
        raise RuntimeError(f"Failed to store documents in vector database") from e


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def ingest_single_pdf(pdf_path, embeddings):
    """
    Ingest a single PDF: load, split, embed, and store.

    Args:
        pdf_path (Path): Validated path to PDF file
        embeddings: Initialized embedding model

    Returns:
        tuple: (num_pages, num_chunks) for statistics
    """
    docs = load_pdf_documents(pdf_path)
    splits = split_documents(docs)
    _ = store_in_vectordb(splits, embeddings)
    return len(docs), len(splits)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF cookbooks into the Smart Pantry vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                        # Ingest default cookbook
  python ingest.py data/cookbook.pdf      # Ingest single file
  python ingest.py data/                  # Ingest all PDFs in folder
  python ingest.py --verbose              # Show detailed output
  python ingest.py --quiet                # Only show errors
        """
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PDF_PATH,
        help=f"Path to PDF file or directory (default: {DEFAULT_PDF_PATH})"
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
    """
    Main execution function with batch support.

    Workflow:
    1. Parse command line arguments (file or directory)
    2. Discover PDFs to process
    3. Check for already-ingested files (deduplication)
    4. Initialize embedding model (once, shared across all PDFs)
    5. Process each PDF: load → split → embed → store (skip duplicates)
    6. Report batch statistics
    """
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    logger.info("=" * 70)
    logger.info("🍳 Smart Pantry - Document Ingestion")
    logger.info("=" * 70)

    input_path = args.path
    logger.info(f"Input path: {input_path}")

    try:
        # Step 1: Discover PDFs (works with files and directories)
        pdf_paths = discover_pdfs(input_path)

        # Step 2: Check which sources are already ingested
        existing_sources = get_ingested_sources()
        if existing_sources:
            logger.info(f"📋 Already in database: {len(existing_sources)} source(s)")

        # Step 3: Initialize embedding model ONCE (shared across all PDFs)
        embeddings = initialize_embeddings()

        # Step 4: Process each PDF
        total_pages = 0
        total_chunks = 0
        skipped = 0
        processed = 0

        for pdf_path in pdf_paths:
            # Deduplication: skip PDFs already in the database
            if str(pdf_path) in existing_sources:
                logger.warning(f"Skipping {pdf_path.name} (already ingested)")
                skipped += 1
                continue

            pages, chunks = ingest_single_pdf(pdf_path, embeddings)
            total_pages += pages
            total_chunks += chunks
            processed += 1

        # Step 5: Report results
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ SUCCESS!")
        logger.info("=" * 70)
        logger.info("📊 Batch Statistics:")
        logger.info(f"   - PDFs processed: {processed}")
        logger.info(f"   - PDFs skipped (already ingested): {skipped}")
        logger.info(f"   - Total pages: {total_pages}")
        logger.info(f"   - Total chunks: {total_chunks}")
        logger.info(f"   - Database location: {CHROMA_PATH}")
        logger.info("")
        logger.info("💡 Next step: Run 'streamlit run app.py' to query your recipes!")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"❌ ERROR: {e}")
        logger.error("💡 Make sure your PDF is in the correct location.")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"❌ ERROR: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        logger.error("💡 Check the error message above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
