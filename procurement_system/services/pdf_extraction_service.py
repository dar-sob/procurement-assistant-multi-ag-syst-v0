# procurement_system/services/pdf_extraction_service.py

import logging
from typing import Optional, Union

from procurement_system.constants import PDF_MAX_CHARS, PDF_MAX_PAGES
from procurement_system.exceptions import PDFProcessingError
from procurement_system.repositories.pdf_repository import PDFRepository

logger = logging.getLogger(__name__)


class PDFExtractionService:
    """Service to extract text from PDFs with fallback to OCR."""

    def __init__(self, pdf_repo: Optional[PDFRepository] = None):
        self._pdf_repo = pdf_repo or PDFRepository()

    def extract_text(
        self,
        source: Union[str, bytes],
        max_pages: Optional[int] = None,
        max_chars: Optional[int] = None,
        force_ocr: bool = False,
    ) -> str:
        """
        Extract text from a PDF.

        Args:
            source: File path, URL, or bytes content.
            max_pages: Maximum pages to process.
            max_chars: Maximum characters to return.
            force_ocr: If True, always use OCR (even if text extraction succeeded).

        Returns:
            Extracted text, truncated to max_chars if specified.
        """
        if max_pages is None:
            max_pages = PDF_MAX_PAGES
        if max_chars is None:
            max_chars = PDF_MAX_CHARS

        # Try text extraction first
        if not force_ocr:
            try:
                text = self._pdf_repo.extract_text_from_pdf(source, max_pages=max_pages)
                if text.strip():
                    logger.info(f"Text extraction succeeded, length: {len(text)}")
                    return self._truncate_text(text, max_chars)
            except PDFProcessingError as e:
                logger.warning(f"Text extraction failed: {e}. Falling back to OCR.")

        # Fallback to OCR
        logger.info("Attempting OCR extraction.")
        try:
            ocr_text = self._pdf_repo.ocr_pdf(source, max_pages=max_pages)
            if not ocr_text.strip():
                raise PDFProcessingError("OCR produced no text.")
            return self._truncate_text(ocr_text, max_chars)
        except PDFProcessingError as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Truncate text to max_chars while preserving word boundaries."""
        if len(text) <= max_chars:
            return text
        # Try to cut at last space before limit
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + "... [truncated]"
