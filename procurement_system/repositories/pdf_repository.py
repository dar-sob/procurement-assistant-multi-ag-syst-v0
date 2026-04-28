# procurement_system/repositories/pdf_repository.py
"""
Production-grade repository for PDF handling:
- Safe download with limits and validation
- Text extraction (pdfplumber + pypdf fallback)
- OCR with timeouts and safeguards

Design principles:
- No business logic (no OCR fallback decisions)
- Hard safety limits (memory, size)
- Idempotent input handling
- No truncation of extracted text (responsibility of service layer)
"""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from typing import Optional, Union
from urllib.parse import urlparse

import pdfplumber
import pytesseract
import requests
from pypdf import PdfReader
from requests.exceptions import RequestException, Timeout

from procurement_system.constants import (
    PDF_DOWNLOAD_TIMEOUT,
    PDF_MAX_FILE_SIZE_BYTES,
    PDF_MAX_PAGES,
    PDF_OCR_LANGUAGE,
    PDF_OCR_TIMEOUT_SECONDS,
    PDF_STRICT_CONTENT_TYPE,
    PDF_CACHE_MAXSIZE,
)
from procurement_system.exceptions import PDFProcessingError
from procurement_system.settings import get_tesseract_cmd

logger = logging.getLogger(__name__)


class PDFRepository:
    """Low-level, production-safe PDF operations."""

    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        self.download_timeout = PDF_DOWNLOAD_TIMEOUT
        self.max_file_size_bytes = PDF_MAX_FILE_SIZE_BYTES
        self.strict_content_type = PDF_STRICT_CONTENT_TYPE

        # Configure tesseract
        tesseract_path = tesseract_cmd or get_tesseract_cmd()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self._ocr_available = self._check_ocr_dependencies()

        logger.info(
            f"PDFRepository initialized: timeout={self.download_timeout}s, "
            f"max_size={self.max_file_size_bytes // (1024*1024)}MB, "
            f"strict_content_type={self.strict_content_type}, "
            f"ocr_available={self._ocr_available}"
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _check_ocr_dependencies(self) -> bool:
        try:
            from pdf2image import convert_from_bytes  # noqa
            return True
        except ImportError:
            logger.warning("pdf2image not installed. OCR unavailable.")
            return False

    def _validate_content_type(self, content_type: str, url: str) -> None:
        """Validate HTTP Content-Type header. Raises on invalid in strict mode."""
        ct = (content_type or "").lower()
        if not ct:
            logger.warning(f"No Content-Type for {url}, assuming PDF")
            return

        allowed = (
            "application/pdf",
            "application/x-pdf",
            "application/octet-stream",
            "application/force-download",
        )
        if any(ct.startswith(prefix) for prefix in allowed):
            return

        if self.strict_content_type:
            raise PDFProcessingError(f"Invalid Content-Type '{ct}' for PDF")

        logger.warning(f"Suspicious Content-Type '{ct}' for {url}")

    def _is_pdf_bytes(self, data: bytes) -> bool:
        """Check if bytes start with PDF signature."""
        return data.startswith(b'%PDF')

    # ------------------------------------------------------------------
    # Download layer (cached ONLY for URLs)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=PDF_CACHE_MAXSIZE)
    def _download_pdf_cached(self, url: str) -> bytes:
        logger.debug(f"PDF download cache miss: {url}")
        data = self._download_pdf(url)
        logger.debug(f"Downloaded {len(data)} bytes from {url}")
        return data

    def _download_pdf(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=self.download_timeout, stream=True)
            response.raise_for_status()

            self._validate_content_type(
                response.headers.get("content-type", ""), url
            )

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.max_file_size_bytes:
                raise PDFProcessingError("PDF exceeds max size (Content-Length)")

            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > self.max_file_size_bytes:
                        raise PDFProcessingError("PDF exceeds max size during download")

            data = b"".join(chunks)
            if not self._is_pdf_bytes(data):
                raise PDFProcessingError("Downloaded data is not a valid PDF")
            return data

        except Timeout as e:
            raise PDFProcessingError(f"Timeout downloading PDF: {e}") from e
        except RequestException as e:
            raise PDFProcessingError(f"Download failed: {e}") from e

    def _get_pdf_bytes(self, source: Union[str, bytes]) -> bytes:
        """Idempotent source normalization with PDF signature check."""
        if isinstance(source, bytes):
            if len(source) > self.max_file_size_bytes:
                raise PDFProcessingError("PDF bytes exceed size limit")
            if not self._is_pdf_bytes(source):
                raise PDFProcessingError("Provided bytes do not represent a PDF")
            return source

        if isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme in ("http", "https"):
                return self._download_pdf_cached(source)

            # Local file
            if not os.path.exists(source):
                raise PDFProcessingError(f"File not found: {source}")
            file_size = os.path.getsize(source)
            if file_size > self.max_file_size_bytes:
                raise PDFProcessingError("Local file exceeds size limit")

            with open(source, "rb") as f:
                header = f.read(5)
                if not header.startswith(b'%PDF'):
                    raise PDFProcessingError(f"File does not appear to be a PDF: {source}")
                f.seek(0)
                data = f.read()
                if not self._is_pdf_bytes(data):
                    raise PDFProcessingError(f"File {source} is not a valid PDF")
                return data

        raise PDFProcessingError(f"Unsupported source type: {type(source)}")

    # ------------------------------------------------------------------
    # Text extraction (returns full text, no truncation)
    # ------------------------------------------------------------------

    def extract_text_from_pdf(
        self,
        source: Union[str, bytes],
        max_pages: int = PDF_MAX_PAGES,
    ) -> str:
        """Extract text from a PDF (full content, not truncated)."""
        pdf_bytes = self._get_pdf_bytes(source)

        with io.BytesIO(pdf_bytes) as stream:
            try:
                with pdfplumber.open(stream) as pdf:
                    pages = min(len(pdf.pages), max_pages)
                    out = []
                    for i in range(pages):
                        text = pdf.pages[i].extract_text()
                        if text:
                            out.append(text)
                    return "\n\n".join(out)
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}, falling back to pypdf")
                return self._extract_text_with_pypdf(pdf_bytes, max_pages)

    def _extract_text_with_pypdf(self, pdf_bytes: bytes, max_pages: int) -> str:
        """Fallback extraction using pypdf."""
        with io.BytesIO(pdf_bytes) as stream:
            try:
                reader = PdfReader(stream)
                pages = min(len(reader.pages), max_pages)
                out = []
                for i in range(pages):
                    text = reader.pages[i].extract_text()
                    if text:
                        out.append(text)
                return "\n\n".join(out)
            except Exception as e:
                raise PDFProcessingError(f"pypdf extraction failed: {e}") from e

    # ------------------------------------------------------------------
    # OCR (for scanned PDFs)
    # ------------------------------------------------------------------

    def ocr_pdf(
        self,
        source: Union[str, bytes],
        max_pages: int = PDF_MAX_PAGES,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Run OCR on a PDF (scanned document). Returns full text."""
        if not self._ocr_available:
            raise PDFProcessingError("OCR not available (pdf2image missing)")

        pdf_bytes = self._get_pdf_bytes(source)
        timeout = timeout_seconds if timeout_seconds is not None else PDF_OCR_TIMEOUT_SECONDS

        if timeout:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._perform_ocr, pdf_bytes, max_pages)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeoutError:
                    raise PDFProcessingError(f"OCR timeout after {timeout}s")
                except Exception as e:
                    raise PDFProcessingError(f"OCR failed: {e}") from e
        else:
            return self._perform_ocr(pdf_bytes, max_pages)

    def _perform_ocr(self, pdf_bytes: bytes, max_pages: int) -> str:
        """Internal OCR execution (without timeout)."""
        try:
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(
                pdf_bytes,
                first_page=1,
                last_page=min(max_pages, 100),  # safety cap
            )
            out = []
            for i, img in enumerate(images):
                # pytesseract has its own internal timeout (10s per page)
                text = pytesseract.image_to_string(
                    img,
                    lang=PDF_OCR_LANGUAGE,
                    timeout=10,
                )
                if text.strip():
                    out.append(f"Page {i+1}:\n{text}")
            images.clear()
            return "\n\n".join(out)
        except Exception as e:
            raise PDFProcessingError(f"OCR processing failed: {e}") from e
