# tests/repositories/test_pdf_repository.py
# pytest tests/repositories/test_pdf_repository.py -v

import io
import logging
import os
import pytest
from unittest.mock import MagicMock, Mock, patch
from concurrent.futures import TimeoutError as FuturesTimeoutError

from procurement_system.repositories.pdf_repository import PDFRepository
from procurement_system.exceptions import PDFProcessingError
from procurement_system.constants import (
    PDF_DOWNLOAD_TIMEOUT,
    PDF_MAX_FILE_SIZE_BYTES,
    PDF_STRICT_CONTENT_TYPE,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_requests_get():
    with patch("procurement_system.repositories.pdf_repository.requests.get") as mock:
        yield mock


@pytest.fixture
def mock_pdfplumber():
    with patch("procurement_system.repositories.pdf_repository.pdfplumber") as mock:
        yield mock


@pytest.fixture
def mock_pypdf():
    with patch("procurement_system.repositories.pdf_repository.PdfReader") as mock:
        yield mock


@pytest.fixture
def mock_tesseract_cmd():
    with patch("procurement_system.repositories.pdf_repository.get_tesseract_cmd") as mock:
        mock.return_value = "/usr/bin/tesseract"
        yield mock


@pytest.fixture
def pdf_repository(mock_tesseract_cmd):
    return PDFRepository()


@pytest.fixture
def sample_pdf_bytes():
    return b"%PDF-1.4\n%test content"


# -----------------------------------------------------------------------------
# Test initialization
# -----------------------------------------------------------------------------

class TestPDFRepositoryInit:
    def test_init_default_values(self, mock_tesseract_cmd):
        """
        BUG FIX: Poprzednia wersja hardkodowała wartości (30, 10*1024*1024, True).
        Teraz importujemy stałe bezpośrednio, żeby test nie złamał się przy zmianie
        wartości w constants.py bez zmiany testu.
        """
        repo = PDFRepository()
        assert repo.download_timeout == PDF_DOWNLOAD_TIMEOUT
        assert repo.max_file_size_bytes == PDF_MAX_FILE_SIZE_BYTES
        assert repo.strict_content_type == PDF_STRICT_CONTENT_TYPE

    def test_init_tesseract_path(self, mock_tesseract_cmd):
        PDFRepository()
        mock_tesseract_cmd.assert_called_once()

    def test_ocr_available_with_pdf2image(self, mock_tesseract_cmd):
        """
        BUG FIX: Brakujący mock_tesseract_cmd powodował, że get_tesseract_cmd()
        był wołany naprawdę — potencjalnie failując w środowiskach bez tesseract.
        """
        with patch.dict("sys.modules", {"pdf2image": Mock()}):
            repo = PDFRepository()
            assert repo._ocr_available is True

    def test_ocr_unavailable_without_pdf2image(self, mock_tesseract_cmd):
        """
        BUG FIX: j.w. — brakujący mock_tesseract_cmd.
        """
        with patch.dict("sys.modules", {"pdf2image": None}):
            repo = PDFRepository()
            assert repo._ocr_available is False


# -----------------------------------------------------------------------------
# Content Type Validation
# -----------------------------------------------------------------------------

class TestValidateContentType:
    def test_valid_pdf_content_type(self, pdf_repository):
        pdf_repository._validate_content_type("application/pdf", "http://example.com/doc.pdf")
        pdf_repository._validate_content_type("application/x-pdf", "http://example.com/doc.pdf")
        pdf_repository._validate_content_type("application/octet-stream", "http://example.com/doc.pdf")
        pdf_repository._validate_content_type("application/force-download", "http://example.com/doc.pdf")

    def test_suspicious_content_type_warning(self, pdf_repository, caplog):
        """
        POPRAWA: Dodano caplog.set_level, żeby jawnie potwierdzić poziom przechwytywania
        logów. Domyślny caplog może nie przechwytywać WARNING jeśli logger ma inny level.
        """
        pdf_repository.strict_content_type = False
        with caplog.at_level(logging.WARNING, logger="procurement_system.repositories.pdf_repository"):
            pdf_repository._validate_content_type("text/html", "http://example.com/doc.pdf")
        assert "Suspicious Content-Type" in caplog.text

    def test_invalid_content_type_strict_raises(self, pdf_repository):
        """
        BUG FIX: PDF_STRICT_CONTENT_TYPE może być False w constants.py, więc
        fixture tworzy repo z strict_content_type=False. Test musi jawnie
        włączyć tryb strict przed wywołaniem — nie może zakładać wartości stałej.
        """
        pdf_repository.strict_content_type = True
        with pytest.raises(PDFProcessingError, match="Invalid Content-Type"):
            pdf_repository._validate_content_type("image/jpeg", "http://example.com/doc.pdf")

    def test_missing_content_type_allowed(self, pdf_repository, caplog):
        """
        POPRAWA: j.w. — explicit caplog level.
        """
        with caplog.at_level(logging.WARNING, logger="procurement_system.repositories.pdf_repository"):
            pdf_repository._validate_content_type("", "http://example.com/doc.pdf")
        assert "No Content-Type" in caplog.text


# -----------------------------------------------------------------------------
# PDF Signature Check
# -----------------------------------------------------------------------------

class TestIsPdfBytes:
    def test_valid_pdf_header(self, pdf_repository):
        assert pdf_repository._is_pdf_bytes(b"%PDF-1.4") is True
        assert pdf_repository._is_pdf_bytes(b"%PDF-2.0") is True

    def test_invalid_pdf_header(self, pdf_repository):
        assert pdf_repository._is_pdf_bytes(b"not a pdf") is False
        assert pdf_repository._is_pdf_bytes(b"") is False


# -----------------------------------------------------------------------------
# Download Layer
# -----------------------------------------------------------------------------

class TestDownloadPDF:
    def test_download_success(self, pdf_repository, mock_requests_get, sample_pdf_bytes):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf", "content-length": "1024"}
        mock_response.iter_content.return_value = [sample_pdf_bytes]
        mock_requests_get.return_value = mock_response
        data = pdf_repository._download_pdf("http://example.com/doc.pdf")
        assert data == sample_pdf_bytes

    def test_download_timeout(self, pdf_repository, mock_requests_get):
        from requests.exceptions import Timeout
        mock_requests_get.side_effect = Timeout("Timeout")
        with pytest.raises(PDFProcessingError, match="Timeout downloading PDF"):
            pdf_repository._download_pdf("http://example.com/doc.pdf")

    def test_download_request_exception(self, pdf_repository, mock_requests_get):
        from requests.exceptions import RequestException
        mock_requests_get.side_effect = RequestException("Connection error")
        with pytest.raises(PDFProcessingError, match="Download failed"):
            pdf_repository._download_pdf("http://example.com/doc.pdf")

    def test_content_length_exceeds_limit(self, pdf_repository, mock_requests_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf", "content-length": "20000000"}
        mock_requests_get.return_value = mock_response
        with pytest.raises(PDFProcessingError, match="exceeds max size"):
            pdf_repository._download_pdf("http://example.com/doc.pdf")

    def test_download_size_exceeds_limit_during_stream(self, pdf_repository, mock_requests_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        large_chunk = b"x" * (pdf_repository.max_file_size_bytes + 1)
        mock_response.iter_content.return_value = [large_chunk]
        mock_requests_get.return_value = mock_response
        with pytest.raises(PDFProcessingError, match="exceeds max size"):
            pdf_repository._download_pdf("http://example.com/doc.pdf")

    def test_invalid_pdf_content(self, pdf_repository, mock_requests_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content.return_value = [b"Not a pdf"]
        mock_requests_get.return_value = mock_response
        with pytest.raises(PDFProcessingError, match="not a valid PDF"):
            pdf_repository._download_pdf("http://example.com/doc.pdf")

    def test_cached_download(self, pdf_repository, mock_requests_get, sample_pdf_bytes):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf", "content-length": "1024"}
        mock_response.iter_content.return_value = [sample_pdf_bytes]
        mock_requests_get.return_value = mock_response
        data1 = pdf_repository._download_pdf_cached("http://example.com/doc.pdf")
        data2 = pdf_repository._download_pdf_cached("http://example.com/doc.pdf")
        assert data1 == data2
        assert mock_requests_get.call_count == 1


# -----------------------------------------------------------------------------
# Source Normalization (_get_pdf_bytes)
# -----------------------------------------------------------------------------

class TestGetPdfBytes:
    def test_from_bytes_valid(self, pdf_repository, sample_pdf_bytes):
        result = pdf_repository._get_pdf_bytes(sample_pdf_bytes)
        assert result == sample_pdf_bytes

    def test_from_bytes_exceeds_size(self, pdf_repository):
        large_bytes = b"x" * (pdf_repository.max_file_size_bytes + 1)
        with pytest.raises(PDFProcessingError, match="exceed size limit"):
            pdf_repository._get_pdf_bytes(large_bytes)

    def test_from_bytes_invalid_signature(self, pdf_repository):
        with pytest.raises(PDFProcessingError, match="do not represent a PDF"):
            pdf_repository._get_pdf_bytes(b"Not PDF")

    def test_from_url(self, pdf_repository, mock_requests_get, sample_pdf_bytes):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content.return_value = [sample_pdf_bytes]
        mock_requests_get.return_value = mock_response
        result = pdf_repository._get_pdf_bytes("http://example.com/doc.pdf")
        assert result == sample_pdf_bytes

    def test_from_local_file(self, pdf_repository, tmp_path, sample_pdf_bytes):
        file_path = tmp_path / "test.pdf"
        file_path.write_bytes(sample_pdf_bytes)
        result = pdf_repository._get_pdf_bytes(str(file_path))
        assert result == sample_pdf_bytes

    def test_local_file_not_found(self, pdf_repository):
        with pytest.raises(PDFProcessingError, match="File not found"):
            pdf_repository._get_pdf_bytes("/nonexistent/file.pdf")

    def test_local_file_too_large(self, pdf_repository, tmp_path):
        file_path = tmp_path / "large.pdf"
        large_bytes = b"x" * (pdf_repository.max_file_size_bytes + 100)
        file_path.write_bytes(large_bytes)
        with pytest.raises(PDFProcessingError, match="exceeds size limit"):
            pdf_repository._get_pdf_bytes(str(file_path))

    def test_local_file_invalid_pdf(self, pdf_repository, tmp_path):
        file_path = tmp_path / "invalid.pdf"
        file_path.write_bytes(b"not pdf")
        with pytest.raises(PDFProcessingError, match="does not appear to be a PDF"):
            pdf_repository._get_pdf_bytes(str(file_path))

    def test_unsupported_source_type(self, pdf_repository):
        with pytest.raises(PDFProcessingError, match="Unsupported source type"):
            pdf_repository._get_pdf_bytes(123)


# -----------------------------------------------------------------------------
# Text Extraction
# -----------------------------------------------------------------------------

class TestExtractTextFromPDF:
    def test_extract_with_pdfplumber_success(self, pdf_repository, mock_pdfplumber, sample_pdf_bytes):
        mock_pdf = Mock()
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        text = pdf_repository.extract_text_from_pdf(sample_pdf_bytes, max_pages=2)
        assert text == "Page 1 content\n\nPage 2 content"

    def test_extract_with_pdfplumber_fallback_to_pypdf(self, pdf_repository, mock_pdfplumber, mock_pypdf, sample_pdf_bytes):
        mock_pdfplumber.open.side_effect = Exception("pdfplumber error")
        mock_reader = Mock()
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Fallback content"
        mock_reader.pages = [mock_page1]
        mock_pypdf.return_value = mock_reader
        text = pdf_repository.extract_text_from_pdf(sample_pdf_bytes, max_pages=1)
        assert text == "Fallback content"

    def test_extract_pypdf_failure(self, pdf_repository, mock_pdfplumber, mock_pypdf, sample_pdf_bytes):
        mock_pdfplumber.open.side_effect = Exception("pdfplumber error")
        mock_pypdf.side_effect = Exception("pypdf error")
        with pytest.raises(PDFProcessingError, match="pypdf extraction failed"):
            pdf_repository.extract_text_from_pdf(sample_pdf_bytes)

    def test_extract_respects_max_pages(self, pdf_repository, mock_pdfplumber, sample_pdf_bytes):
        mock_pdf = Mock()
        mock_pdf.pages = [Mock() for _ in range(5)]
        for i, page in enumerate(mock_pdf.pages):
            page.extract_text.return_value = f"Page {i+1}"
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        text = pdf_repository.extract_text_from_pdf(sample_pdf_bytes, max_pages=3)
        assert text.count("\n\n") == 2
        assert "Page 4" not in text

    def test_extract_skips_empty_pages(self, pdf_repository, mock_pdfplumber, sample_pdf_bytes):
        mock_pdf = Mock()
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = None
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        text = pdf_repository.extract_text_from_pdf(sample_pdf_bytes, max_pages=2)
        assert text == "Content"


# -----------------------------------------------------------------------------
# OCR (with correct patching of pdf2image.convert_from_bytes)
# -----------------------------------------------------------------------------

class TestOCR:
    def test_ocr_not_available(self, pdf_repository):
        pdf_repository._ocr_available = False
        with pytest.raises(PDFProcessingError, match="OCR not available"):
            pdf_repository.ocr_pdf(b"%PDF-1.4")

    def test_ocr_success(self, pdf_repository, sample_pdf_bytes):
        pdf_repository._ocr_available = True
        with patch("pdf2image.convert_from_bytes") as mock_convert:
            mock_image = Mock()
            mock_convert.return_value = [mock_image]
            with patch("pytesseract.image_to_string") as mock_ocr:
                mock_ocr.return_value = "Recognized text"
                text = pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=1, timeout_seconds=None)
                assert "Recognized text" in text
                assert "Page 1" in text

    def test_ocr_with_timeout(self, pdf_repository, sample_pdf_bytes):
        pdf_repository._ocr_available = True
        with patch("pdf2image.convert_from_bytes") as mock_convert:
            mock_image = Mock()
            mock_convert.return_value = [mock_image]
            with patch("pytesseract.image_to_string") as mock_ocr:
                mock_ocr.return_value = "Text"
                text = pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=1, timeout_seconds=1)
                assert "Text" in text

    def test_ocr_timeout_exceeded(self, pdf_repository, sample_pdf_bytes):
        """
        BUG FIX: ThreadPoolExecutor jest importowany w module jako
        'from concurrent.futures import ThreadPoolExecutor', więc lokalna nazwa
        jest już zbindowana w momencie importu. Patch pod 'concurrent.futures.ThreadPoolExecutor'
        nie ma żadnego efektu — moduł używa swojej lokalnej referencji.
        Poprawna ścieżka patcha: 'procurement_system.repositories.pdf_repository.ThreadPoolExecutor'.
        """
        pdf_repository._ocr_available = True
        with patch("procurement_system.repositories.pdf_repository.ThreadPoolExecutor") as mock_executor:
            mock_future = Mock()
            mock_future.result.side_effect = FuturesTimeoutError()
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            with pytest.raises(PDFProcessingError, match="OCR timeout after 1s"):
                pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=1, timeout_seconds=1)

    def test_ocr_handles_exception_without_timeout(self, pdf_repository, sample_pdf_bytes):
        """
        BUG FIX: Poprzednia wersja matchowała 'OCR failed', ale przy timeout_seconds=None
        kod woła _perform_ocr bezpośrednio, który rzuca PDFProcessingError('OCR processing
        failed: ...'). Fraza 'OCR failed' NIE jest podciągiem 'OCR processing failed',
        więc test zawsze failował. Poprawiony match: 'OCR processing failed'.
        """
        pdf_repository._ocr_available = True
        with patch("pdf2image.convert_from_bytes") as mock_convert:
            mock_convert.side_effect = Exception("Conversion failed")
            with pytest.raises(PDFProcessingError, match="OCR processing failed"):
                pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=1, timeout_seconds=None)

    def test_ocr_handles_exception_with_timeout(self, pdf_repository, sample_pdf_bytes):
        """
        NOWY TEST: Weryfikuje komunikat błędu gdy wyjątek jest przechwycony przez
        ścieżkę z ThreadPoolExecutor (timeout_seconds podany). Tu message to 'OCR failed'.
        """
        pdf_repository._ocr_available = True
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            mock_future = Mock()
            mock_future.result.side_effect = RuntimeError("Unexpected error")
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            with pytest.raises(PDFProcessingError, match="OCR failed"):
                pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=1, timeout_seconds=5)

    def test_ocr_respects_max_pages(self, pdf_repository, sample_pdf_bytes):
        pdf_repository._ocr_available = True
        with patch("pdf2image.convert_from_bytes") as mock_convert:
            mock_convert.return_value = [Mock(), Mock(), Mock()]  # 3 images
            with patch("pytesseract.image_to_string") as mock_ocr:
                mock_ocr.return_value = "Text"
                pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=2, timeout_seconds=None)
                mock_convert.assert_called_once()
                args, kwargs = mock_convert.call_args
                assert kwargs.get("last_page") == 2


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_pdf_bytes(self, pdf_repository):
        with pytest.raises(PDFProcessingError, match="do not represent a PDF"):
            pdf_repository._get_pdf_bytes(b"")

    def test_text_extraction_from_empty_pdf(self, pdf_repository, mock_pdfplumber, sample_pdf_bytes):
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = None
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        text = pdf_repository.extract_text_from_pdf(sample_pdf_bytes)
        assert text == ""

    def test_ocr_clears_images(self, pdf_repository, sample_pdf_bytes):
        """
        BUG FIX: Poprzednia wersja używała mock_images = [Mock(), Mock()] — prawdziwej
        listy Pythona. Wywołanie mock_images.clear.assert_called_once() failowało z
        AttributeError, bo list.clear to wbudowana metoda, nie Mock.

        Naprawione przez użycie MagicMock(spec=list), który ma .clear jako Mock method
        i jest poprawnie konfigurowany jako return_value convert_from_bytes.
        """
        pdf_repository._ocr_available = True
        mock_images = MagicMock(spec=list)
        mock_images.__iter__.return_value = iter([Mock(), Mock()])
        with patch("pdf2image.convert_from_bytes") as mock_convert:
            mock_convert.return_value = mock_images
            with patch("pytesseract.image_to_string") as mock_ocr:
                mock_ocr.return_value = ""
                pdf_repository.ocr_pdf(sample_pdf_bytes, max_pages=2, timeout_seconds=None)
                mock_images.clear.assert_called_once()
