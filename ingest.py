"""
Smart Pantry - Document Ingestion Script

This script processes PDF cookbooks and stores them in a local vector database (ChromaDB).
It uses local embeddings (HuggingFace) to avoid API rate limits and costs.

Supports both single file and batch ingestion with duplicate detection.

Usage:
    python ingest.py [path]

    path can be:
    - A single PDF file: python ingest.py data/my-cookbook.pdf
    - A directory: python ingest.py data/   (processes all PDFs in folder)
    - Omitted: Uses default PDF

Examples:
    python ingest.py data/my-cookbook.pdf     # Single file
    python ingest.py data/                   # All PDFs in folder
    python ingest.py                         # Default PDF
"""

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
        print(f"📁 Found {len(pdfs)} PDF(s) in {input_path}")
        for pdf in pdfs:
            print(f"   - {pdf.name}")
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
    except Exception:
        # If database is corrupted or unreadable, start fresh
        return set()


def load_pdf_documents(pdf_path):
    """
    Loads a PDF file and extracts text content page by page.

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        list: List of Document objects (one per page)

    Raises:
        Exception: If PDF loading fails
    """
    print(f"\n📖 Loading PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        print(f"✓ Loaded {len(docs)} pages")
        return docs

    except Exception as e:
        raise Exception(f"Failed to load PDF: {e}")


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
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Overlap: {CHUNK_OVERLAP} characters")
    print(f"   Separators: {CHUNK_SEPARATORS}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS
    )

    splits = text_splitter.split_documents(docs)
    print(f"✓ Created {len(splits)} chunks")

    return splits


def initialize_embeddings():
    """
    Initialize the local embedding model using the shared core configuration.

    This is a thin wrapper around core.create_embeddings() that adds
    user-friendly console output for the ingestion process.

    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    """
    print("\n🤖 Initializing local embedding model...")
    print("   (First run will download ~120MB model)")

    # Use shared embedding configuration from core
    embeddings = create_embeddings()

    print("✓ Embedding model ready")
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
    print(f"\n💾 Storing chunks in vector database: {CHROMA_PATH}")
    print(f"   Processing {len(splits)} chunks in batches of {BATCH_SIZE}...")

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

        print(f"✓ Successfully stored {len(splits)} chunks")
        return vectorstore

    except Exception as e:
        raise Exception(f"Failed to store documents in vector database: {e}")


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
    print("=" * 70)
    print("🍳 Smart Pantry - Document Ingestion")
    print("=" * 70)

    # Step 1: Get path from command line or use default
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        print(f"Input path: {input_path}")
    else:
        input_path = DEFAULT_PDF_PATH
        print(f"Using default: {input_path}")

    try:
        # Step 2: Discover PDFs (works with files and directories)
        pdf_paths = discover_pdfs(input_path)

        # Step 3: Check which sources are already ingested
        existing_sources = get_ingested_sources()
        if existing_sources:
            print(f"\n📋 Already in database: {len(existing_sources)} source(s)")

        # Step 4: Initialize embedding model ONCE (shared across all PDFs)
        embeddings = initialize_embeddings()

        # Step 5: Process each PDF
        total_pages = 0
        total_chunks = 0
        skipped = 0
        processed = 0

        for pdf_path in pdf_paths:
            # Deduplication: skip PDFs already in the database
            if str(pdf_path) in existing_sources:
                print(f"\nSkipping {pdf_path.name} (already ingested)")
                skipped += 1
                continue

            pages, chunks = ingest_single_pdf(pdf_path, embeddings)
            total_pages += pages
            total_chunks += chunks
            processed += 1

        # Step 6: Report results
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"📊 Batch Statistics:")
        print(f"   - PDFs processed: {processed}")
        print(f"   - PDFs skipped (already ingested): {skipped}")
        print(f"   - Total pages: {total_pages}")
        print(f"   - Total chunks: {total_chunks}")
        print(f"   - Database location: {CHROMA_PATH}")
        print(f"\n💡 Next step: Run 'streamlit run app.py' to query your recipes!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"💡 Make sure your PDF is in the correct location.")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"💡 Check the error message above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
