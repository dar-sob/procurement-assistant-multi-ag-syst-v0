# tests/services/test_currency_service.py
# pytest tests/services/test_currency_service.py -v

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from procurement_system.services.currency_service import CurrencyService
from procurement_system.constants import SUPPORTED_CURRENCIES


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_repo():
    return Mock()


@pytest.fixture
def service(mock_repo):
    """Serwis z wstrzykniętym mock repozytorium — izolacja od warstwy HTTP."""
    return CurrencyService(repository=mock_repo)


@pytest.fixture(autouse=True)
def clear_lru_cache(service):
    """
    KLUCZOWE: lru_cache jest przywiązany do instancji przez (self, currency, today).
    Bez czyszczenia cache między testami wynik z poprzedniego testu może być
    zwrócony zamiast nowego — testy zaczynają się wzajemnie psuć w nieprzewidywalny
    sposób zależny od kolejności wykonania.
    """
    yield
    service._get_rate_cached.cache_clear()


@pytest.fixture
def frozen_today():
    """
    Stała data używana wszędzie tam, gdzie test musi kontrolować date.today().
    lru_cache używa 'today' jako część klucza — bez zamrożenia daty dwa wywołania
    w różnych dniach dadzą dwa wpisy w cache zamiast jednego.
    """
    return date(2024, 1, 15)


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

class TestCurrencyServiceInit:
    def test_uses_injected_repository(self, mock_repo):
        svc = CurrencyService(repository=mock_repo)
        assert svc._repo is mock_repo

    def test_creates_default_repository_when_none(self):
        """Bez wstrzyknięcia serwis tworzy własne CurrencyRepository."""
        with patch("procurement_system.services.currency_service.CurrencyRepository") as MockRepo:
            svc = CurrencyService()
            MockRepo.assert_called_once()
            assert svc._repo is MockRepo.return_value


# -----------------------------------------------------------------------------
# convert_to_usd — USD passthrough
# -----------------------------------------------------------------------------

class TestConvertToUsdPassthrough:
    def test_usd_returns_amount_unchanged(self, service, mock_repo):
        result = service.convert_to_usd(100.0, "USD")
        assert result == 100.0
        mock_repo.get_usd_rate.assert_not_called()

    def test_usd_lowercase_returns_amount_unchanged(self, service, mock_repo):
        result = service.convert_to_usd(50.0, "usd")
        assert result == 50.0
        mock_repo.get_usd_rate.assert_not_called()

    def test_usd_with_whitespace_returns_amount_unchanged(self, service, mock_repo):
        result = service.convert_to_usd(200.0, "  usd  ")
        assert result == 200.0
        mock_repo.get_usd_rate.assert_not_called()

    def test_none_currency_treated_as_usd(self, service, mock_repo):
        """None → domyślnie 'USD' zgodnie z logiką `(currency or 'USD')`."""
        result = service.convert_to_usd(75.0, None)
        assert result == 75.0
        mock_repo.get_usd_rate.assert_not_called()

    def test_empty_string_currency_treated_as_usd(self, service, mock_repo):
        result = service.convert_to_usd(10.0, "")
        assert result == 10.0
        mock_repo.get_usd_rate.assert_not_called()

    def test_zero_amount_usd(self, service):
        assert service.convert_to_usd(0.0, "USD") == 0.0

    def test_negative_amount_usd(self, service):
        """Serwis nie waliduje znaku kwoty — zwraca jak jest."""
        assert service.convert_to_usd(-50.0, "USD") == -50.0


# -----------------------------------------------------------------------------
# convert_to_usd — currency conversion
# -----------------------------------------------------------------------------

class TestConvertToUsdConversion:
    def test_basic_conversion(self, service, mock_repo, frozen_today):
        """1 PLN = 1/4.0 USD = 0.25 USD."""
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result = service.convert_to_usd(100.0, "PLN")

        assert result == pytest.approx(25.0)

    def test_conversion_formula(self, service, mock_repo, frozen_today):
        """
        Weryfikacja formuły: rate_usd_to_cur = X, więc amount_usd = amount / X.
        Przy rate=5.0 i amount=10.0 → 10/5 = 2.0 USD.
        """
        mock_repo.get_usd_rate.return_value = (5.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result = service.convert_to_usd(10.0, "EUR")

        assert result == pytest.approx(2.0)

    def test_lowercase_currency_is_normalized(self, service, mock_repo, frozen_today):
        """Waluta jest uppercasowana przed użyciem — 'pln' i 'PLN' to to samo."""
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result_lower = service.convert_to_usd(100.0, "pln")

        service._get_rate_cached.cache_clear()
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result_upper = service.convert_to_usd(100.0, "PLN")

        assert result_lower == pytest.approx(result_upper)

    def test_currency_with_whitespace_normalized(self, service, mock_repo, frozen_today):
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result = service.convert_to_usd(100.0, "  PLN  ")

        assert result == pytest.approx(25.0)

    def test_zero_amount_converts_to_zero(self, service, mock_repo, frozen_today):
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result = service.convert_to_usd(0.0, "PLN")

        assert result == pytest.approx(0.0)

    def test_repository_called_with_correct_currency(self, service, mock_repo, frozen_today):
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            service.convert_to_usd(100.0, "PLN")

        mock_repo.get_usd_rate.assert_called_once_with("PLN")


# -----------------------------------------------------------------------------
# Unsupported currencies
# -----------------------------------------------------------------------------

class TestConvertToUsdUnsupportedCurrency:
    def test_unsupported_currency_raises_value_error(self, service):
        with pytest.raises(ValueError, match="Unsupported currency"):
            service.convert_to_usd(100.0, "XYZ")

    def test_unsupported_currency_error_contains_currency_name(self, service):
        with pytest.raises(ValueError, match="FAKE"):
            service.convert_to_usd(100.0, "fake")

    def test_unsupported_currency_does_not_call_repo(self, service, mock_repo):
        with pytest.raises(ValueError):
            service.convert_to_usd(100.0, "XYZ")
        mock_repo.get_usd_rate.assert_not_called()

    def test_all_supported_currencies_do_not_raise(self, service, mock_repo, frozen_today):
        """
        Smoke test: każda waluta z SUPPORTED_CURRENCIES (poza USD) przechodzi
        walidację i trafia do repo. Łapie regres przy zmianie stałej.
        """
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            for currency in SUPPORTED_CURRENCIES:
                service._get_rate_cached.cache_clear()
                service.convert_to_usd(1.0, currency)  # nie powinno rzucić


# -----------------------------------------------------------------------------
# LRU cache
# -----------------------------------------------------------------------------

class TestGetRateCached:
    def test_repo_called_once_for_same_currency_and_date(self, service, mock_repo, frozen_today):
        """
        Dwa wywołania z tą samą walutą i datą → jedno trafienie do repo.
        To jest podstawowa gwarancja cache'u.
        """
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            service.convert_to_usd(100.0, "PLN")
            service.convert_to_usd(200.0, "PLN")

        mock_repo.get_usd_rate.assert_called_once_with("PLN")

    def test_repo_called_separately_for_different_currencies(self, service, mock_repo, frozen_today):
        """Różne waluty → osobne wpisy w cache → repo wołane raz na każdą walutę."""
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            service.convert_to_usd(100.0, "PLN")
            service.convert_to_usd(100.0, "EUR")

        assert mock_repo.get_usd_rate.call_count == 2

    def test_repo_called_separately_for_different_dates(self, service, mock_repo):
        """
        Różne daty → różne klucze cache → repo wołane dwa razy.
        Gdyby 'today' nie było częścią klucza, kurs z wcześniejszego dnia
        byłby serwowany na kolejny dzień — cichy błąd w przeliczeniach.
        """
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = date(2024, 1, 15)
            service.convert_to_usd(100.0, "PLN")

        service._get_rate_cached.cache_clear()
        mock_repo.get_usd_rate.return_value = (4.1, "2024-01-16")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = date(2024, 1, 16)
            service.convert_to_usd(100.0, "PLN")

        assert mock_repo.get_usd_rate.call_count == 2

    def test_cache_returns_rate_not_tuple(self, service, mock_repo, frozen_today):
        """
        _get_rate_cached zwraca tylko float (rate), nie tuple (rate, date).
        Sprawdzamy, że rozpakowywanie `rate, _ = repo.get_usd_rate(...)` działa
        i do cache trafia sama liczba.
        """
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            rate = service._get_rate_cached("PLN", frozen_today)

        assert rate == 4.0
        assert isinstance(rate, float)

    def test_cache_clear_forces_repo_call(self, service, mock_repo, frozen_today):
        """Po cache_clear() kolejne wywołanie musi trafić do repo, nie do cache."""
        mock_repo.get_usd_rate.return_value = (4.0, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            service.convert_to_usd(100.0, "PLN")

        service._get_rate_cached.cache_clear()
        mock_repo.get_usd_rate.return_value = (4.5, "2024-01-15")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            result = service.convert_to_usd(100.0, "PLN")

        assert result == pytest.approx(100.0 / 4.5)
        assert mock_repo.get_usd_rate.call_count == 2


# -----------------------------------------------------------------------------
# Repository error propagation
# -----------------------------------------------------------------------------

class TestConvertToUsdErrorPropagation:
    def test_repo_exception_propagates(self, service, mock_repo, frozen_today):
        """
        Serwis nie łapie wyjątków z repozytorium — propagują do wywołującego.
        Weryfikujemy, że żaden try/except nie połknie błędu po cichu.
        """
        mock_repo.get_usd_rate.side_effect = RuntimeError("API down")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            with pytest.raises(RuntimeError, match="API down"):
                service.convert_to_usd(100.0, "PLN")

    def test_repo_value_error_propagates(self, service, mock_repo, frozen_today):
        mock_repo.get_usd_rate.side_effect = ValueError("bad response")

        with patch("procurement_system.services.currency_service.date") as mock_date:
            mock_date.today.return_value = frozen_today
            with pytest.raises(ValueError, match="bad response"):
                service.convert_to_usd(100.0, "PLN")
