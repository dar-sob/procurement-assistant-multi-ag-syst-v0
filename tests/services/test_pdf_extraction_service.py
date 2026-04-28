# tests/services/test_pdf_extraction_service.py
# pytest tests/services/test_pdf_extraction_service.py -v

import pytest
from unittest.mock import Mock, patch, MagicMock

from procurement_system.services.pdf_extraction_service import PDFExtractionService
from procurement_system.exceptions import PDFProcessingError
from procurement_system.constants import PDF_MAX_CHARS, PDF_MAX_PAGES


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_repo():
    return Mock()


@pytest.fixture
def service(mock_repo):
    """Service with injected mock repository — full isolation."""
    return PDFExtractionService(pdf_repo=mock_repo)


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

class TestPDFExtractionServiceInit:
    def test_uses_injected_repository(self, mock_repo):
        svc = PDFExtractionService(pdf_repo=mock_repo)
        assert svc._pdf_repo is mock_repo

    def test_creates_default_repository_when_none(self):
        """When no repository is provided, the service creates its own."""
        with patch("procurement_system.services.pdf_extraction_service.PDFRepository") as MockRepo:
            svc = PDFExtractionService()
            MockRepo.assert_called_once()
            assert svc._pdf_repo is MockRepo.return_value


# -----------------------------------------------------------------------------
# extract_text — successful text extraction (no OCR)
# -----------------------------------------------------------------------------

class TestExtractTextSuccess:
    def test_extracts_text_when_available(self, service, mock_repo):
        """When extract_text_from_pdf returns text, OCR is not called."""
        mock_repo.extract_text_from_pdf.return_value = "Example invoice PLN 1234.56"

        result = service.extract_text("dummy.pdf")

        assert "Example invoice" in result
        mock_repo.extract_text_from_pdf.assert_called_once_with("dummy.pdf", max_pages=PDF_MAX_PAGES)
        mock_repo.ocr_pdf.assert_not_called()

    def test_respects_max_pages_parameter(self, service, mock_repo):
        mock_repo.extract_text_from_pdf.return_value = "Text from PDF"

        service.extract_text("dummy.pdf", max_pages=5)

        mock_repo.extract_text_from_pdf.assert_called_once_with("dummy.pdf", max_pages=5)

    def test_respects_max_chars_parameter(self, service, mock_repo):
        long_text = "Alice has a cat. " * 1000
        mock_repo.extract_text_from_pdf.return_value = long_text

        result = service.extract_text("dummy.pdf", max_chars=50)

        assert len(result) <= 50 + len("... [truncated]")
        assert result.endswith("... [truncated]")

    def test_truncates_preserving_word_boundaries(self, service, mock_repo):
        mock_repo.extract_text_from_pdf.return_value = "This is a very long text that should be truncated at the right place."

        result = service.extract_text("dummy.pdf", max_chars=20)

        assert result.endswith("... [truncated]")
        # should cut after a whole word
        assert "place" not in result


# -----------------------------------------------------------------------------
# extract_text — fallback to OCR
# -----------------------------------------------------------------------------

class TestExtractTextOCRFallback:
    def test_falls_back_to_ocr_when_text_extraction_fails(self, service, mock_repo):
        """When extract_text_from_pdf raises an exception → fallback to OCR."""
        mock_repo.extract_text_from_pdf.side_effect = PDFProcessingError("Text extraction failed")
        mock_repo.ocr_pdf.return_value = "Text obtained by OCR"

        result = service.extract_text("dummy.pdf")

        assert result == "Text obtained by OCR"
        mock_repo.ocr_pdf.assert_called_once_with("dummy.pdf", max_pages=PDF_MAX_PAGES)

    def test_falls_back_to_ocr_when_text_is_empty(self, service, mock_repo):
        """When extract_text_from_pdf returns only whitespace → fallback to OCR."""
        mock_repo.extract_text_from_pdf.return_value = "   \n\t  "
        mock_repo.ocr_pdf.return_value = "Text from OCR"

        result = service.extract_text("dummy.pdf")

        assert result == "Text from OCR"
        mock_repo.ocr_pdf.assert_called_once()

    def test_force_ocr_skips_text_extraction(self, service, mock_repo):
        """When force_ocr=True, skip extract_text_from_pdf entirely."""
        mock_repo.ocr_pdf.return_value = "Result of forced OCR"

        result = service.extract_text("dummy.pdf", force_ocr=True)

        assert result == "Result of forced OCR"
        mock_repo.extract_text_from_pdf.assert_not_called()
        mock_repo.ocr_pdf.assert_called_once()


# -----------------------------------------------------------------------------
# extract_text — OCR success / failure
# -----------------------------------------------------------------------------

class TestExtractTextOCR:
    def test_raises_when_ocr_fails(self, service, mock_repo):
        """When both text extraction and OCR fail — propagate the exception."""
        mock_repo.extract_text_from_pdf.side_effect = PDFProcessingError("Text failed")
        mock_repo.ocr_pdf.side_effect = PDFProcessingError("OCR also failed")

        with pytest.raises(PDFProcessingError, match="OCR also failed"):
            service.extract_text("dummy.pdf")

    def test_raises_when_ocr_returns_empty_text(self, service, mock_repo):
        """OCR cannot return empty text."""
        mock_repo.extract_text_from_pdf.side_effect = PDFProcessingError("Text failed")
        mock_repo.ocr_pdf.return_value = "   \n  "

        with pytest.raises(PDFProcessingError, match="OCR produced no text"):
            service.extract_text("dummy.pdf")


# -----------------------------------------------------------------------------
# Source types handling
# -----------------------------------------------------------------------------

class TestExtractTextSourceTypes:
    def test_accepts_bytes_as_source(self, service, mock_repo):
        mock_repo.extract_text_from_pdf.return_value = "Text from bytes"
        pdf_bytes = b"%PDF-1.4 fake content"

        result = service.extract_text(pdf_bytes)

        assert "Text from bytes" in result
        mock_repo.extract_text_from_pdf.assert_called_once_with(pdf_bytes, max_pages=PDF_MAX_PAGES)

    def test_accepts_url_as_source(self, service, mock_repo):
        mock_repo.extract_text_from_pdf.return_value = "Text from URL"
        url = "https://example.com/invoice.pdf"

        result = service.extract_text(url)

        assert "Text from URL" in result
        mock_repo.extract_text_from_pdf.assert_called_once_with(url, max_pages=PDF_MAX_PAGES)


# -----------------------------------------------------------------------------
# Truncation logic (private method)
# -----------------------------------------------------------------------------

class TestTruncateText:
    def test_does_not_truncate_short_text(self, service):
        text = "Short text"
        result = service._truncate_text(text, 100)
        assert result == text

    def test_truncates_long_text_with_ellipsis(self, service):
        text = "This is a very long text " * 10
        result = service._truncate_text(text, 30)
        assert result.endswith("... [truncated]")

    def test_truncates_at_word_boundary(self, service):
        text = "Alice has a cat and the cat has Alice and they live happily together."
        result = service._truncate_text(text, 15)
        assert result.endswith("... [truncated]")
        prefix = result[:-len("... [truncated]")]
        next_idx = len(prefix)
        if next_idx < len(text):
            assert text[next_idx].isspace()
