
# repositories/currency_repository.py
import requests
import logging
from typing import Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from procurement_system.settings import get_frankfurter_api_url, get_frankfurter_timeout

logger = logging.getLogger(__name__)

class CurrencyRepository:
    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=4),
        reraise=True,
    )
    def get_usd_rate(self, currency: str) -> Tuple[float, str]:
        """
        Retrieves the exchange rate 1 USD → currency from the Frankfurter API. 
        Returns (rate, date).
        """
        response = requests.get(
            f"{get_frankfurter_api_url()}/latest",
            params={"from": "USD", "to": currency},
            timeout=get_frankfurter_timeout(),
        )
        response.raise_for_status()
        data = response.json()
        return data["rates"][currency], data["date"]
