# procurement_system/tools/__init__.py

"""
Central tool definitions for all agents.

This module provides:
    - Individual tool functions (e.g., convert_to_usd)
    - Per‑agent tool lists (INTAKE_TOOLS, PROCUREMENT_TOOLS, etc.)
    - ALL_TOOLS for global registration
"""

from langchain_core.tools import BaseTool
from typing import List

# Tools
from .currency_converter import ConvertToUSDTool
from .supplier_web_search import SearchWebSuppliersTool
from .pdf_reader import ReadPDFTool

# Services
from procurement_system.services.currency_service import CurrencyService
from procurement_system.services.pdf_extraction_service import PDFExtractionService
from procurement_system.services.supplier_web_search_service import SupplierWebSearchService


# ---------------------------------------------------------------------------
# Tool factories - accept services, return finished tools
# ---------------------------------------------------------------------------

def make_currency_tool (currency_service: CurrencyService) -> ConvertToUSDTool:
    return ConvertToUSDTool(currency_service=currency_service)


def make_pdf_reader_tool (pdf_service: PDFExtractionService) -> ReadPDFTool:
    return ReadPDFTool(pdf_service=pdf_service)


def make_supplier_web_search_tool (supplier_search_service: SupplierWebSearchService) -> SearchWebSuppliersTool:
    return SearchWebSuppliersTool(supplier_search_service=supplier_search_service)


# ---------------------------------------------------------------------------
# Agent Toolkit Factory
# ---------------------------------------------------------------------------

def make_intake_tools(services: dict) -> List[BaseTool]:
    return []


def make_procurement_tools(services: dict) -> List[BaseTool]:    
    return[
        make_currency_tool(services.get("currency_service")),
        make_pdf_reader_tool(services.get("pdf_extraction_service")),
        make_supplier_web_search_tool(services.get("supplier_web_search_service"))
    ]


def make_analyst_tools(services: dict) -> List[BaseTool]:
    return [
        make_pdf_reader_tool(services.get("pdf_extraction_service")),
    ]


def make_orchestrator_tools(services: dict) -> List[BaseTool]:
    return[]
