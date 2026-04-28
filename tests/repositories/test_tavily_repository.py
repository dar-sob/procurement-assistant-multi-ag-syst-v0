# tests/repositories/test_tavily_repository.py
# pytest tests/repositories/test_tavily_repository.py -v

import pytest
from unittest.mock import Mock, patch
from requests.exceptions import Timeout, RequestException
import json

from procurement_system.repositories.tavily_repository import TavilyRepository
from procurement_system.exceptions import TavilySearchError
from procurement_system.schemas.tool_schemas import SupplierWebSearchResult


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_api_key():
    """Mock the Tavily API key."""
    with patch("procurement_system.repositories.tavily_repository.get_tavily_api_key") as mock:
        mock.return_value = "test-api-key"
        yield mock


@pytest.fixture
def mock_requests_post():
    """Mock requests.post."""
    with patch("procurement_system.repositories.tavily_repository.requests.post") as mock:
        yield mock


@pytest.fixture
def repository(mock_api_key, mock_requests_post):
    """Return TavilyRepository instance with mocked dependencies."""
    return TavilyRepository()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestTavilyRepositoryInit:
    """Test class initialisation and API key validation."""

    def test_init_success(self, mock_api_key):
        """Initialisation succeeds when API key is present."""
        mock_api_key.return_value = "valid-key"
        repo = TavilyRepository()
        assert repo.api_key == "valid-key"
        assert repo.base_url == "https://api.tavily.com/search"
        assert repo.timeout == 10 
        assert repo.max_results == 5

    def test_init_missing_api_key(self, mock_api_key):
        """Initialisation raises ValueError when API key is missing."""
        mock_api_key.return_value = ""
        with pytest.raises(ValueError, match="TAVILY_API_KEY not set in environment"):
            TavilyRepository()

        mock_api_key.return_value = None
        with pytest.raises(ValueError, match="TAVILY_API_KEY not set in environment"):
            TavilyRepository()


class TestTavilyRepositorySearch:
    """Test the search method."""

    def test_search_success(self, repository, mock_requests_post):
        """Successful search returns list of SupplierWebSearchResult."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "TechDirect Official Site",
                    "url": "https://techdirect.com",
                    "content": "Leading supplier of IT equipment",
                    "score": 0.95,
                },
                {
                    "title": "Review of TechDirect",
                    "url": "https://example.com/review",
                    "content": "Positive feedback",
                    "score": 0.82,
                },
            ]
        }
        mock_requests_post.return_value = mock_response

        results = repository.search("TechDirect supplier review")
        assert len(results) == 2
        assert isinstance(results[0], SupplierWebSearchResult)
        assert results[0].title == "TechDirect Official Site"
        # Zamieniamy na porównanie stringów
        assert str(results[0].url) == "https://techdirect.com/"
        assert results[0].content == "Leading supplier of IT equipment"
        assert results[0].score == 0.95

        # Verify request payload
        mock_requests_post.assert_called_once()
        call_args = mock_requests_post.call_args
        assert call_args[1]["json"]["api_key"] == "test-api-key"
        assert call_args[1]["json"]["query"] == "TechDirect supplier review"
        assert call_args[1]["json"]["max_results"] == 5
        assert call_args[1]["timeout"] == 10

    def test_search_empty_results(self, repository, mock_requests_post):
        """Search returns empty list when no results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_requests_post.return_value = mock_response

        results = repository.search("nonexistent supplier")
        assert results == []

    def test_search_missing_results_key(self, repository, mock_requests_post):
        """Search returns empty list if 'results' key missing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_requests_post.return_value = mock_response

        results = repository.search("anything")
        assert results == []

    def test_search_partial_fields(self, repository, mock_requests_post):
        """Missing optional fields are replaced with defaults (but URL must be valid)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Only title and url provided",
                    "url": "https://example.com",  
                }
            ]
        }
        mock_requests_post.return_value = mock_response

        results = repository.search("query")
        assert len(results) == 1
        assert results[0].title == "Only title and url provided"
        assert str(results[0].url) == "https://example.com/"
        assert results[0].content == ""       
        assert results[0].score is None       

    def test_search_timeout(self, repository, mock_requests_post):
        """Timeout triggers TavilySearchError."""
        mock_requests_post.side_effect = Timeout("Connection timed out")

        with pytest.raises(TavilySearchError) as exc_info:
            repository.search("test")
        assert "timeout" in str(exc_info.value).lower()

    def test_search_request_exception(self, repository, mock_requests_post):
        """RequestException triggers TavilySearchError."""
        mock_requests_post.side_effect = RequestException("Network error")

        with pytest.raises(TavilySearchError) as exc_info:
            repository.search("test")
        assert "request failed" in str(exc_info.value).lower()

    def test_search_invalid_json(self, repository, mock_requests_post):
        """Invalid JSON response triggers TavilySearchError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_requests_post.return_value = mock_response

        with pytest.raises(TavilySearchError) as exc_info:
            repository.search("test")
        assert "Invalid JSON response" in str(exc_info.value)

    def test_search_http_error(self, repository, mock_requests_post):
        """HTTP error (4xx, 5xx) triggers TavilySearchError."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = RequestException("500 Server Error")
        mock_requests_post.return_value = mock_response

        with pytest.raises(TavilySearchError) as exc_info:
            repository.search("test")
        assert "request failed" in str(exc_info.value).lower()

    def test_search_max_results_respected(self, repository, mock_requests_post):
        """Only max_results items are returned even if API returns more."""
        repository.max_results = 2
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": "https://example.com"}  # dodajemy url
                for i in range(10)
            ]
        }
        mock_requests_post.return_value = mock_response

        results = repository.search("query")
        assert len(results) == 2

    def test_search_logging_on_error(self, repository, mock_requests_post, caplog):
        """Errors are logged appropriately."""
        import logging
        caplog.set_level(logging.ERROR)
        mock_requests_post.side_effect = Timeout("Timeout")

        with pytest.raises(TavilySearchError):
            repository.search("test")

        assert "Tavily API timeout" in caplog.text
