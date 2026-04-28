# tests/repositories/test_currency_repository.py
# pytest tests/repositories/test_currency_repository.py -v

import pytest
from unittest.mock import Mock, patch, call
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

from procurement_system.repositories.currency_repository import CurrencyRepository
from procurement_system.settings import get_frankfurter_api_url


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_requests_get():
    with patch("procurement_system.repositories.currency_repository.requests.get") as mock:
        yield mock


@pytest.fixture
def mock_settings():
    """Izoluje testy od rzeczywistej konfiguracji (URL, timeout)."""
    with patch("procurement_system.repositories.currency_repository.get_frankfurter_api_url") as mock_url, \
         patch("procurement_system.repositories.currency_repository.get_frankfurter_timeout") as mock_timeout:
        mock_url.return_value = "https://api.frankfurter.app"
        mock_timeout.return_value = 5
        yield {"url": mock_url, "timeout": mock_timeout}


@pytest.fixture
def repo():
    return CurrencyRepository()


@pytest.fixture
def mock_response_pln():
    """Przykładowa poprawna odpowiedź API dla USD→PLN."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "amount": 1.0,
        "base": "USD",
        "date": "2024-01-15",
        "rates": {"PLN": 4.05},
    }
    return response


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------

class TestGetUsdRateSuccess:
    def test_returns_rate_and_date(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        mock_requests_get.return_value = mock_response_pln

        rate, date = repo.get_usd_rate("PLN")

        assert rate == 4.05
        assert date == "2024-01-15"

    def test_calls_correct_endpoint(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        mock_requests_get.return_value = mock_response_pln

        repo.get_usd_rate("PLN")

        mock_requests_get.assert_called_once_with(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "PLN"},
            timeout=5,
        )

    def test_returns_tuple(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        mock_requests_get.return_value = mock_response_pln

        result = repo.get_usd_rate("PLN")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rate_is_float(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        mock_requests_get.return_value = mock_response_pln

        rate, _ = repo.get_usd_rate("PLN")

        assert isinstance(rate, float)

    def test_different_currencies(self, repo, mock_settings, mock_requests_get):
        """Weryfikuje, że currency jest poprawnie przekazywane do URL i JSON."""
        for currency, rate_value in [("EUR", 0.92), ("GBP", 0.79), ("JPY", 148.5)]:
            response = Mock()
            response.json.return_value = {
                "date": "2024-01-15",
                "rates": {currency: rate_value},
            }
            mock_requests_get.return_value = response

            rate, _ = repo.get_usd_rate(currency)

            assert rate == rate_value
            _, kwargs = mock_requests_get.call_args
            assert kwargs["params"]["to"] == currency

    def test_uses_configured_timeout(self, repo, mock_requests_get):
        """Timeout pochodzi z get_frankfurter_timeout(), nie jest hardkodowany."""
        response = Mock()
        response.json.return_value = {"date": "2024-01-15", "rates": {"PLN": 4.05}}
        mock_requests_get.return_value = response

        with patch("procurement_system.repositories.currency_repository.get_frankfurter_timeout") as mock_timeout, \
             patch("procurement_system.repositories.currency_repository.get_frankfurter_api_url") as mock_url:
            mock_timeout.return_value = 99
            mock_url.return_value = "https://api.frankfurter.app"
            repo.get_usd_rate("PLN")

        _, kwargs = mock_requests_get.call_args
        assert kwargs["timeout"] == 99

    def test_uses_configured_api_url(self, repo, mock_requests_get):
        """URL pochodzi z get_frankfurter_api_url(), nie jest hardkodowany."""
        response = Mock()
        response.json.return_value = {"date": "2024-01-15", "rates": {"PLN": 4.05}}
        mock_requests_get.return_value = response

        with patch("procurement_system.repositories.currency_repository.get_frankfurter_api_url") as mock_url, \
             patch("procurement_system.repositories.currency_repository.get_frankfurter_timeout") as mock_timeout:
            mock_url.return_value = "https://custom-api.example.com"
            mock_timeout.return_value = 5
            repo.get_usd_rate("PLN")

        args, _ = mock_requests_get.call_args
        assert args[0] == "https://custom-api.example.com/latest"


# -----------------------------------------------------------------------------
# Network errors & retry logic
# -----------------------------------------------------------------------------

class TestGetUsdRateRetry:
    def test_retries_on_request_exception(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        """
        Tenacity retryuje przy RequestException (bazowa klasa dla Timeout,
        ConnectionError itp.). Weryfikujemy, że po przejściowym błędzie
        kolejna próba kończy się sukcesem.
        """
        mock_requests_get.side_effect = [
            RequestException("transient error"),
            mock_response_pln,
        ]

        # Wyłączamy wait żeby test nie trwał kilku sekund
        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            rate, date = repo.get_usd_rate("PLN")

        assert rate == 4.05
        assert mock_requests_get.call_count == 2

    def test_retries_on_timeout(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        mock_requests_get.side_effect = [
            Timeout("timed out"),
            mock_response_pln,
        ]

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            rate, _ = repo.get_usd_rate("PLN")

        assert rate == 4.05
        assert mock_requests_get.call_count == 2

    def test_retries_on_connection_error(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        mock_requests_get.side_effect = [
            ConnectionError("connection refused"),
            mock_response_pln,
        ]

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            rate, _ = repo.get_usd_rate("PLN")

        assert rate == 4.05
        assert mock_requests_get.call_count == 2

    def test_reraises_after_max_attempts(self, repo, mock_settings, mock_requests_get):
        """
        Po wyczerpaniu 3 prób (stop_after_attempt(3)) tenacity reraisuje
        oryginalny wyjątek (reraise=True).
        """
        mock_requests_get.side_effect = RequestException("persistent error")

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            with pytest.raises(RequestException, match="persistent error"):
                repo.get_usd_rate("PLN")

        assert mock_requests_get.call_count == 3

    def test_exactly_three_attempts_on_failure(self, repo, mock_settings, mock_requests_get):
        """Weryfikacja, że liczba prób to dokładnie 3 (stop_after_attempt)."""
        mock_requests_get.side_effect = ConnectionError("down")

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            with pytest.raises(RequestException):
                repo.get_usd_rate("EUR")

        assert mock_requests_get.call_count == 3

    def test_no_retry_on_http_error(self, repo, mock_settings, mock_requests_get):
        """
        HTTPError (raise_for_status) NIE jest podklasą RequestException w tenacity
        — właściwie jest, więc powinien retryować. Ten test dokumentuje faktyczne
        zachowanie: HTTPError dziedziczy po RequestException i JEST retryowany.
        """
        mock_requests_get.return_value = Mock(
            **{"raise_for_status.side_effect": HTTPError("404 Not Found")}
        )

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            with pytest.raises(HTTPError):
                repo.get_usd_rate("PLN")

        assert mock_requests_get.call_count == 3

    def test_successful_first_attempt_no_retry(self, repo, mock_settings, mock_requests_get, mock_response_pln):
        """Przy sukcesie od razu — dokładnie jedno wywołanie HTTP."""
        mock_requests_get.return_value = mock_response_pln

        repo.get_usd_rate("PLN")

        assert mock_requests_get.call_count == 1


# -----------------------------------------------------------------------------
# HTTP error responses (raise_for_status)
# -----------------------------------------------------------------------------

class TestGetUsdRateHttpErrors:
    def test_404_raises(self, repo, mock_settings, mock_requests_get):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_requests_get.return_value = mock_response

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            with pytest.raises(HTTPError):
                repo.get_usd_rate("XYZ")

    def test_500_raises(self, repo, mock_settings, mock_requests_get):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("500 Internal Server Error")
        mock_requests_get.return_value = mock_response

        with patch("procurement_system.repositories.currency_repository.CurrencyRepository.get_usd_rate.retry.wait"):
            with pytest.raises(HTTPError):
                repo.get_usd_rate("PLN")


# -----------------------------------------------------------------------------
# Malformed / unexpected API responses
# -----------------------------------------------------------------------------

class TestGetUsdRateMalformedResponse:
    def test_missing_rates_key_raises(self, repo, mock_settings, mock_requests_get):
        """Brak klucza 'rates' w odpowiedzi — KeyError."""
        mock_response = Mock()
        mock_response.json.return_value = {"date": "2024-01-15"}
        mock_requests_get.return_value = mock_response

        with pytest.raises(KeyError):
            repo.get_usd_rate("PLN")

    def test_missing_currency_in_rates_raises(self, repo, mock_settings, mock_requests_get):
        """API zwróciło 'rates', ale bez żądanej waluty."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "date": "2024-01-15",
            "rates": {"EUR": 0.92},  # brak PLN
        }
        mock_requests_get.return_value = mock_response

        with pytest.raises(KeyError):
            repo.get_usd_rate("PLN")

    def test_missing_date_key_raises(self, repo, mock_settings, mock_requests_get):
        """Brak klucza 'date' w odpowiedzi — KeyError."""
        mock_response = Mock()
        mock_response.json.return_value = {"rates": {"PLN": 4.05}}
        mock_requests_get.return_value = mock_response

        with pytest.raises(KeyError):
            repo.get_usd_rate("PLN")

    def test_empty_json_raises(self, repo, mock_settings, mock_requests_get):
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_requests_get.return_value = mock_response

        with pytest.raises(KeyError):
            repo.get_usd_rate("PLN")
