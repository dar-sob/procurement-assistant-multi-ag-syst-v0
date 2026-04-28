from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List


# ─────────────────────────────────────────────────────────
# CURRENCY CONVERT
# ─────────────────────────────────────────────────────────

class CurrencyInput(BaseModel):
    """Input schema for the convert_to_usd tool."""
    amount: float = Field(
        description="The amount to convert. Must be a positive number.",
        gt=0,
        le=1_000_000_000,
    )
    currency: Optional[str] = Field(
        default="USD",
        description=(
            "ISO 4217 source currency code (e.g. PLN, EUR, GBP). "
            "Defaults to USD. If USD is provided, no conversion is performed."
        ),
    )


# ─────────────────────────────────────────────────────────
# SUPPLIER WEB SEARCH
# ─────────────────────────────────────────────────────────

class SupplierWebSearchResult(BaseModel):
    """A single search result from Tavily."""
    title: str
    url: HttpUrl
    content: str
    score: Optional[float] = None

class SearchSuppliersInput(BaseModel):
    """Input schema for the supplier search tool."""
    product_description: str
    category: Optional[str] = None


# ─────────────────────────────────────────────────────────
# READ PDF
# ─────────────────────────────────────────────────────────

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Union

class ReadPDFInput(BaseModel):
    """Input schema for reading a PDF."""
    source: Union[str, HttpUrl] = Field(
        description="File path (local) or URL to the PDF document."
    )
    max_pages: Optional[int] = Field(
        default=None,
        description="Maximum number of pages to extract (default: all)."
    )
    max_chars: Optional[int] = Field(
        default=None,
        description="Maximum characters to return (default: 5000)."
    )
