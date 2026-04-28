# procurement_system/services/supplier_search_service.py

import logging
from typing import List, Optional

from procurement_system.constants import TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE
from procurement_system.exceptions import TavilySearchError
from procurement_system.repositories.tavily_repository import TavilyRepository
from procurement_system.schemas.tool_schemas import SupplierWebSearchResult

logger = logging.getLogger(__name__)


class SupplierWebSearchService:
    """Service for searching suppliers using Tavily."""

    def __init__(self, tavily_repo: Optional[TavilyRepository] = None):
        self._tavily_repo = tavily_repo or TavilyRepository()

    def search_suppliers(
        self,
        product_description: str,
        category: Optional[str] = None,
    ) -> List[SupplierWebSearchResult]:
        """Search for suppliers using Tavily."""
        query = TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE.format(
            product_description=product_description
        )
        if category:
            query += f" {category}"

        logger.info(f"Searching suppliers with query: {query}")

        try:
            results = self._tavily_repo.search(query)
        except TavilySearchError as e:
            logger.error(f"Tavily search failed: {e}")
            raise

        logger.info(f"Found {len(results)} supplier results.")
        return results
