# tests/services/test_supplier_web_search_service.py
# pytest tests/services/test_supplier_web_search_service.py -v
import pytest
from unittest.mock import Mock, patch
from procurement_system.services.supplier_web_search_service import SupplierWebSearchService   # ← poprawiona nazwa
from procurement_system.exceptions import TavilySearchError
from procurement_system.schemas.tool_schemas import SupplierWebSearchResult


class TestSupplierWebSearchService:
    """Test suite for SupplierWebSearchService."""

    @pytest.fixture
    def mock_tavily_repo(self):
        return Mock()

    @pytest.fixture
    def service(self, mock_tavily_repo):
        return SupplierWebSearchService(tavily_repo=mock_tavily_repo)

    @patch("procurement_system.services.supplier_web_search_service.TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE",
           "Find suppliers for {product_description}")
    def test_search_suppliers_success_no_category(self, service, mock_tavily_repo):
        product_description = "steel beams"
        expected_query = "Find suppliers for steel beams"
        mock_results = [SupplierWebSearchResult(title="Supplier A", url="http://example.com", content="...")]
        mock_tavily_repo.search.return_value = mock_results

        results = service.search_suppliers(product_description)

        mock_tavily_repo.search.assert_called_once_with(expected_query)
        assert results == mock_results

    @patch("procurement_system.services.supplier_web_search_service.TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE",
           "Find suppliers for {product_description}")
    def test_search_suppliers_success_with_category(self, service, mock_tavily_repo):
        product_description = "laptops"
        category = "electronics"
        expected_query = "Find suppliers for laptops electronics"
        mock_results = [SupplierWebSearchResult(title="Supplier B", url="http://example2.com", content="...")]
        mock_tavily_repo.search.return_value = mock_results

        results = service.search_suppliers(product_description, category=category)

        mock_tavily_repo.search.assert_called_once_with(expected_query)
        assert results == mock_results

    @patch("procurement_system.services.supplier_web_search_service.TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE",
           "Find suppliers for {product_description}")
    def test_search_suppliers_tavily_search_error(self, service, mock_tavily_repo):
        product_description = "concrete"
        mock_tavily_repo.search.side_effect = TavilySearchError("API key missing")

        with pytest.raises(TavilySearchError, match="API key missing"):
            service.search_suppliers(product_description)

        mock_tavily_repo.search.assert_called_once_with("Find suppliers for concrete")

    def test_search_suppliers_default_repository(self):
        with patch("procurement_system.services.supplier_web_search_service.TavilyRepository") as MockRepo:
            instance = MockRepo.return_value
            instance.search.return_value = []
            with patch("procurement_system.services.supplier_web_search_service.TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE",
                       "Find {product_description}"):
                service = SupplierWebSearchService()
                service.search_suppliers("test")
                MockRepo.assert_called_once()
                instance.search.assert_called_once()

    @patch("procurement_system.services.supplier_web_search_service.logger")
    def test_logging_on_success(self, mock_logger, service, mock_tavily_repo):
        product_description = "wood"
        mock_tavily_repo.search.return_value = []
        with patch("procurement_system.services.supplier_web_search_service.TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE",
                   "Find {product_description}"):
            service.search_suppliers(product_description)

            mock_logger.info.assert_any_call("Searching suppliers with query: Find wood")
            mock_logger.info.assert_any_call("Found 0 supplier results.")

    @patch("procurement_system.services.supplier_web_search_service.logger")
    def test_logging_on_error(self, mock_logger, service, mock_tavily_repo):
        product_description = "plastic"
        mock_tavily_repo.search.side_effect = TavilySearchError("Network error")
        with patch("procurement_system.services.supplier_web_search_service.TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE",
                   "Find {product_description}"):
            with pytest.raises(TavilySearchError):
                service.search_suppliers(product_description)

            mock_logger.error.assert_called_once_with("Tavily search failed: Network error")
