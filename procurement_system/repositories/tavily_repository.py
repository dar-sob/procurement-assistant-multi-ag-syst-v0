# procurement_system/repositories/tavily_repository.py

import logging
from typing import List

import requests
from requests.exceptions import RequestException, Timeout

from procurement_system.constants import (
    TAVILY_DEFAULT_MAX_RESULTS,
    TAVILY_DEFAULT_TIMEOUT,
    TAVILY_API_URL
)
from procurement_system.exceptions import TavilySearchError
from procurement_system.schemas.tool_schemas import SupplierWebSearchResult
from procurement_system.settings import get_tavily_api_key

logger = logging.getLogger(__name__)


class TavilyRepository:
    """Handles communication with Tavily API."""

    def __init__(self):
        self.api_key = get_tavily_api_key()
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not set in environment.")
        self.base_url = TAVILY_API_URL
        self.timeout = TAVILY_DEFAULT_TIMEOUT
        self.max_results = TAVILY_DEFAULT_MAX_RESULTS

    def search(self, query: str) -> List[SupplierWebSearchResult]:
        """Perform a search query against Tavily."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
        }

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Timeout as e:
            logger.error(f"Tavily API timeout: {e}")
            raise TavilySearchError(f"Tavily API timeout: {e}") from e
        except RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
            raise TavilySearchError(f"Tavily API request failed: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid JSON response from Tavily: {e}")
            raise TavilySearchError(f"Invalid JSON response: {e}") from e

        results = data.get("results", [])
        parsed = []
        for item in results[:self.max_results]:
            parsed.append(
                SupplierWebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score"),
                )
            )
        return parsed
