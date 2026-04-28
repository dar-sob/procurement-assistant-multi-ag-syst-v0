# procurement_system/tools/pdf_reader.py

import logging
from typing import Optional, Type, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from procurement_system.exceptions import PDFProcessingError
from procurement_system.schemas.tool_schemas import ReadPDFInput
from procurement_system.services.pdf_extraction_service import PDFExtractionService
from procurement_system.constants import PDF_MAX_PAGES, PDF_MAX_CHARS

logger = logging.getLogger(__name__)


class ReadPDFTool(BaseTool):
    """Tool to extract text from a PDF document (local file or URL)."""

    name: str = "read_pdf"
    description: str = (
        "Extract text content from a PDF document. Use this tool to read supplier offers, "
        "technical specifications, or any PDF attached to a request. "
        "Provide the source (local file path or URL). "
        "Optionally limit the number of pages and characters. "
        "Returns the extracted text (may be truncated). "
        "**Note:** PDF file size must not exceed 5 MB."
    )
    args_schema: Type[BaseModel] = ReadPDFInput

    def __init__(self, pdf_service: Optional[PDFExtractionService] = None, **kwargs):
        super().__init__(**kwargs)
        self._pdf_service = pdf_service or PDFExtractionService()

    def _run(
        self,
        source: Union[str, bytes],
        max_pages: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """Run the tool."""
        logger.info(f"Reading PDF from source: {source[:100]}...")
        try:
            text = self._pdf_service.extract_text(
                source=source,
                max_pages=max_pages or PDF_MAX_PAGES,
                max_chars=max_chars or PDF_MAX_CHARS,
            )
            return text
        except PDFProcessingError as e:
            logger.exception("PDF processing failed")
            return f"Error reading PDF: {e}"
