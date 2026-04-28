
# services/currency_service.py
import logging
from datetime import date
from functools import lru_cache
from procurement_system.repositories.currency_repository import CurrencyRepository
from procurement_system.constants import SUPPORTED_CURRENCIES

logger = logging.getLogger(__name__)

class CurrencyService:
    def __init__(self, repository: CurrencyRepository = None):
        self._repo = repository or CurrencyRepository()

    @lru_cache(maxsize=64)
    def _get_rate_cached(self, currency: str, today: date) -> float:
        """Downloads the USD → currency rate from the repository, caches it for a day."""
        rate, _ = self._repo.get_usd_rate(currency)
        return rate

    def convert_to_usd(self, amount: float, currency: str) -> float:
        """
        Converts an amount from a given currency to USD. 
        If currency == 'USD', returns the amount unchanged..
        """
        currency = (currency or "USD").upper().strip()

        if currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}")

        if currency == "USD":
            return amount

        # Kurs: 1 USD = X currency -> 1 currency = 1/X USD
        rate_usd_to_cur = self._get_rate_cached(currency, date.today())
        rate_cur_to_usd = 1 / rate_usd_to_cur
        return amount * rate_cur_to_usd
