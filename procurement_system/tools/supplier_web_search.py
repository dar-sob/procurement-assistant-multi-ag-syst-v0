# procurement_system/tools/supplier_web_search.py

import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from procurement_system.exceptions import TavilySearchError
from procurement_system.schemas.tool_schemas import SearchSuppliersInput
from procurement_system.services.supplier_web_search_service import SupplierWebSearchService
from procurement_system.constants import MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


class SearchWebSuppliersTool(BaseTool):
    """Tool to search for suppliers on the internet using Tavily."""

    name: str = "search_suppliers"
    description: str = (
        "Search for suppliers on the internet using Tavily. "
        "Use this tool when you need to find potential suppliers for a product or service, "
        "especially if no internal suppliers are available. "
        "Provide a detailed product_description (e.g., 'industrial 3D printer'), "
        "optionally a category to refine results. "
        "Returns a list of relevant suppliers with name, website URL, and a short snippet."
    )
    args_schema: Type[BaseModel] = SearchSuppliersInput

    def __init__(self, supplier_search_service: Optional[SupplierWebSearchService] = None, **kwargs):
        super().__init__(**kwargs)
        self._service = supplier_search_service or SupplierWebSearchService()

    def _run(
        self,
        product_description: str,
        category: Optional[str] = None,
    ) -> str:
        """Execute the search and return formatted results."""
        logger.info(
            f"Tool 'search_suppliers' invoked: product='{product_description[:50]}...', category={category}"
        )

        try:
            results = self._service.search_suppliers(product_description, category)
        except TavilySearchError as e:
            logger.exception(f"Supplier search failed: {e}")
            return f"Error searching for suppliers: {e}"

        if not results:
            return "No suppliers found."

        lines = []
        for idx, res in enumerate(results[:5], start=1):
            snippet = res.content[:MAX_CONTENT_LENGTH]
            if len(res.content) > MAX_CONTENT_LENGTH:
                snippet += "..."
            lines.append(
                f"{idx}. {res.title}\n"
                f"   URL: {res.url}\n"
                f"   Snippet: {snippet}"
            )
        return "\n\n".join(lines)
