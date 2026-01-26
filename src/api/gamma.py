"""Polymarket Gamma API client.

This module provides access to the Gamma API for:
- Event discovery and metadata
- Market details and tokens
- Historical price data

The Gamma API is used for market discovery and metadata lookup.
No authentication required.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

from .base import BaseAPIClient
from .rate_limiter import RateLimiter
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Token:
    """A conditional token representing an outcome."""

    token_id: str
    outcome: str
    price: Optional[Decimal] = None
    winner: Optional[bool] = None

    @classmethod
    def from_api(cls, data: dict) -> "Token":
        """Create from API response."""
        price = data.get("price")
        return cls(
            token_id=data.get("token_id", data.get("tokenId", "")),
            outcome=data.get("outcome", ""),
            price=Decimal(str(price)) if price else None,
            winner=data.get("winner"),
        )


@dataclass
class Market:
    """A prediction market."""

    condition_id: str
    question: str
    slug: str
    tokens: list[Token] = field(default_factory=list)
    description: Optional[str] = None
    end_date: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    volume: Decimal = Decimal("0")
    volume_24h: Decimal = Decimal("0")
    liquidity: Decimal = Decimal("0")

    @property
    def yes_token_id(self) -> Optional[str]:
        """Get YES token ID (first token)."""
        return self.tokens[0].token_id if self.tokens else None

    @property
    def no_token_id(self) -> Optional[str]:
        """Get NO token ID (second token)."""
        return self.tokens[1].token_id if len(self.tokens) > 1 else None

    @classmethod
    def from_api(cls, data: dict) -> "Market":
        """Create from API response."""
        # Parse end date
        end_date_str = data.get("endDate", data.get("end_date_iso"))
        end_date = None
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Parse tokens
        tokens = []
        for t in data.get("tokens", data.get("outcomes", [])):
            if isinstance(t, dict):
                tokens.append(Token.from_api(t))
            else:
                # Simple outcome string
                tokens.append(Token(token_id="", outcome=str(t)))

        # Also check clobTokenIds format
        clob_tokens = data.get("clobTokenIds", [])
        outcomes = data.get("outcomes", ["Yes", "No"])
        if clob_tokens and not tokens:
            for i, token_id in enumerate(clob_tokens):
                outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
                tokens.append(Token(token_id=token_id, outcome=outcome))

        return cls(
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            question=data.get("question", data.get("title", "")),
            slug=data.get("slug", data.get("market_slug", "")),
            tokens=tokens,
            description=data.get("description"),
            end_date=end_date,
            active=data.get("active", True),
            closed=data.get("closed", False),
            volume=Decimal(str(data.get("volume", data.get("volumeNum", 0)))),
            volume_24h=Decimal(str(data.get("volume24hr", 0))),
            liquidity=Decimal(str(data.get("liquidity", data.get("liquidityNum", 0)))),
        )


@dataclass
class Event:
    """An event containing one or more markets."""

    id: str
    slug: str
    title: str
    markets: list[Market] = field(default_factory=list)
    description: Optional[str] = None
    end_date: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    volume: Decimal = Decimal("0")
    liquidity: Decimal = Decimal("0")

    @classmethod
    def from_api(cls, data: dict) -> "Event":
        """Create from API response."""
        # Parse end date
        end_date_str = data.get("endDate", data.get("end_date"))
        end_date = None
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Parse markets
        markets = []
        for m in data.get("markets", []):
            try:
                markets.append(Market.from_api(m))
            except Exception as e:
                logger.warning("market_parse_error", error=str(e))

        return cls(
            id=data.get("id", data.get("event_id", "")),
            slug=data.get("slug", ""),
            title=data.get("title", data.get("name", "")),
            markets=markets,
            description=data.get("description"),
            end_date=end_date,
            active=data.get("active", True),
            closed=data.get("closed", False),
            volume=Decimal(str(data.get("volume", data.get("volumeNum", 0)))),
            liquidity=Decimal(str(data.get("liquidity", data.get("liquidityNum", 0)))),
        )


class GammaAPIClient(BaseAPIClient):
    """Client for Polymarket Gamma API.

    Provides market discovery and metadata.
    No authentication required.

    Example:
        async with GammaAPIClient() as client:
            events = await client.get_events(limit=10)
            market = await client.get_market(condition_id)
    """

    DEFAULT_BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 10.0,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize Gamma API client.

        Args:
            base_url: Gamma API base URL
            timeout_s: Request timeout in seconds
            rate_limiter: Optional rate limiter
        """
        if rate_limiter is None:
            rate_limiter = RateLimiter()

        super().__init__(
            base_url=base_url,
            timeout_s=timeout_s,
            rate_limiter=rate_limiter,
        )

    async def get_events(
        self,
        active: Optional[bool] = True,
        closed: Optional[bool] = False,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume",
        ascending: bool = False,
    ) -> list[Event]:
        """Get events with optional filtering.

        Args:
            active: Filter by active status
            closed: Filter by closed status
            limit: Maximum results
            offset: Pagination offset
            order: Sort field (volume, liquidity, startDate, endDate)
            ascending: Sort direction

        Returns:
            List of Event objects
        """
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        response = await self.get("/events", params=params)

        events = []
        data = response.data if isinstance(response.data, list) else response.data.get("events", [])
        for item in data:
            try:
                events.append(Event.from_api(item))
            except Exception as e:
                logger.warning("event_parse_error", error=str(e))

        return events

    async def get_event(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event object or None if not found
        """
        response = await self.get(f"/events/{event_id}")
        if response.data:
            return Event.from_api(response.data)
        return None

    async def get_event_by_slug(self, slug: str) -> Optional[Event]:
        """Get an event by slug.

        Args:
            slug: Event slug

        Returns:
            Event object or None if not found
        """
        response = await self.get("/events", params={"slug": slug})
        data = response.data if isinstance(response.data, list) else response.data.get("events", [])
        if data:
            return Event.from_api(data[0])
        return None

    async def get_markets(
        self,
        active: Optional[bool] = True,
        closed: Optional[bool] = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Get markets with optional filtering.

        Args:
            active: Filter by active status
            closed: Filter by closed status
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of Market objects
        """
        params = {
            "limit": limit,
            "offset": offset,
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        response = await self.get("/markets", params=params)

        markets = []
        data = response.data if isinstance(response.data, list) else response.data.get("markets", [])
        for item in data:
            try:
                markets.append(Market.from_api(item))
            except Exception as e:
                logger.warning("market_parse_error", error=str(e))

        return markets

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """Get a specific market by condition ID.

        Args:
            condition_id: Market condition ID

        Returns:
            Market object or None if not found

        Note:
            The Gamma API has a bug where it returns a default market (typically
            the oldest one) when a condition ID is not found, instead of 404/empty.
            We verify the returned market's condition_id matches to avoid this.
        """
        # Query by conditionId parameter (not path) since the API expects numeric IDs in path
        response = await self.get("/markets", params={"conditionId": condition_id, "limit": 1})
        data = response.data if isinstance(response.data, list) else response.data.get("markets", [])
        if data:
            market = Market.from_api(data[0])
            # CRITICAL: Verify condition ID matches to avoid Gamma API's fallback behavior
            # The API returns a default market instead of 404 when condition ID not found
            if market.condition_id.lower() == condition_id.lower():
                return market
            # Condition ID mismatch - API returned wrong market
            return None
        return None

    async def get_market_by_slug(self, slug: str) -> Optional[Market]:
        """Get a market by slug.

        Args:
            slug: Market slug

        Returns:
            Market object or None if not found
        """
        response = await self.get("/markets", params={"slug": slug})
        data = response.data if isinstance(response.data, list) else response.data.get("markets", [])
        if data:
            return Market.from_api(data[0])
        return None

    async def search_markets(
        self,
        query: str,
        limit: int = 20,
    ) -> list[Market]:
        """Search for markets by keyword.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching Market objects
        """
        response = await self.get("/markets", params={
            "q": query,
            "limit": limit,
        })

        markets = []
        data = response.data if isinstance(response.data, list) else response.data.get("markets", [])
        for item in data:
            try:
                markets.append(Market.from_api(item))
            except Exception as e:
                logger.warning("market_parse_error", error=str(e))

        return markets

    async def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 60,
    ) -> list[dict]:
        """Get price history for a token.

        Args:
            token_id: Token ID
            interval: Time interval (1h, 1d, 1w, max)
            fidelity: Data point frequency in minutes

        Returns:
            List of price history data points
        """
        response = await self.get("/prices-history", params={
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        })

        return response.data.get("history", response.data) if isinstance(response.data, dict) else response.data
