"""
Tests for ingest.py

Tests the PDF discovery and document splitting logic.
Uses tmp_path fixtures to create temporary files — no real PDFs needed.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from ingest import discover_pdfs, split_documents


class TestDiscoverPdfs:
    """Tests for PDF file discovery (single file and batch mode)."""

    def test_single_pdf_file(self, tmp_path):
        """A valid PDF path should return a list containing that path."""
        pdf = tmp_path / "cookbook.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        result = discover_pdfs(str(pdf))

        assert result == [pdf]

    def test_directory_returns_sorted_pdfs(self, tmp_path):
        """A directory should return all PDFs sorted alphabetically."""
        (tmp_path / "b_recipes.pdf").write_bytes(b"%PDF")
        (tmp_path / "a_recipes.pdf").write_bytes(b"%PDF")
        (tmp_path / "notes.txt").write_text("not a pdf")

        result = discover_pdfs(str(tmp_path))

        assert len(result) == 2
        assert result[0].name == "a_recipes.pdf"
        assert result[1].name == "b_recipes.pdf"

    def test_nonexistent_path_raises(self):
        """A path that doesn't exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_pdfs("/nonexistent/path.pdf")

    def test_non_pdf_file_raises(self, tmp_path):
        """A non-PDF file should raise ValueError."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("not a pdf")

        with pytest.raises(ValueError, match="not a PDF"):
            discover_pdfs(str(txt_file))

    def test_empty_directory_raises(self, tmp_path):
        """A directory with no PDFs should raise ValueError."""
        (tmp_path / "notes.txt").write_text("not a pdf")

        with pytest.raises(ValueError, match="No PDF files found"):
            discover_pdfs(str(tmp_path))


class TestSplitDocuments:
    """Tests for document chunking logic."""

    def test_splits_long_document(self):
        """Documents longer than chunk_size should be split into multiple chunks."""
        # Must include \n\n separators — the splitter uses ["\n\n", "Title:", "Ingredients:"]
        # Without valid split points, it keeps the document as one chunk
        paragraph = "Recipe content with details about cooking steps and ingredients.\n\n"
        long_content = paragraph * 50  # ~3200 chars with paragraph breaks
        docs = [Document(page_content=long_content, metadata={"source": "test.pdf", "page": 1})]

        splits = split_documents(docs)

        assert len(splits) > 1

    def test_preserves_metadata(self):
        """Metadata should survive the splitting process."""
        docs = [Document(
            page_content="Short recipe content that fits in one chunk.",
            metadata={"source": "cookbook.pdf", "page": 42}
        )]

        splits = split_documents(docs)

        assert all(split.metadata["source"] == "cookbook.pdf" for split in splits)

    def test_empty_input(self):
        """Splitting an empty list should return an empty list."""
        splits = split_documents([])

        assert splits == []
