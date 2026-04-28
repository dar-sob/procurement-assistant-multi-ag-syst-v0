

# tools/currency_converter.py
import logging
from typing import Type
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from procurement_system.schemas.tool_schemas import CurrencyInput
from procurement_system.services.currency_service import CurrencyService
from procurement_system.constants import SUPPORTED_CURRENCIES

logger = logging.getLogger(__name__)

class ConvertToUSDTool(BaseTool):
    name: str = "convert_to_usd"
    description: str = "Converts a given amount from any supported currency to USD using live exchange rates."
    args_schema: Type[BaseModel] = CurrencyInput


    def __init__(self, currency_service:CurrencyService, **kwargs):
        super().__init__(**kwargs)
        self._currency_service = currency_service


    def _run(self, amount: float, currency: str = "USD") -> str:

        """
        Converts a given amount from a source currency into USD.
        Uses live exchange rates from the Frankfurter API (ECB data).
        If no currency is provided or the source is already USD, returns as-is.
        """

        currency = (currency or "USD").upper().strip()

        if currency == "USD":
                return f"{amount:.2f} USD — no conversion needed, already in USD."  

        logger.info("Tool invoked: convert_to_usd(amount=%.2f, currency=%s)", amount, currency)

        try:
            result_usd = self._currency_service._run(amount, currency) 
            # Optional date
            return f"{amount:.2f} {currency} = {result_usd:.2f} USD"

        except ValueError as e:
            logger.warning("Validation error: %s", e)
            return f"Error: {e}. Supported codes: {', '.join(sorted(SUPPORTED_CURRENCIES))}"


        except Exception as e:
            logger.exception("Unexpected error in convert_to_usd")
            return f"Error: could not convert currency — {e}"
