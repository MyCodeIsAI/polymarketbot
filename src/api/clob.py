"""Polymarket CLOB (Central Limit Order Book) API client.

This module provides access to the CLOB API for:
- Order book data
- Price information
- Order placement and cancellation
- Market information

The CLOB API requires authentication for order operations.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from .base import BaseAPIClient
from .rate_limiter import RateLimiter
from .auth import PolymarketAuth
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order time-in-force type."""

    GTC = "GTC"  # Good-til-cancelled
    GTD = "GTD"  # Good-til-date
    FOK = "FOK"  # Fill-or-kill
    IOC = "IOC"  # Immediate-or-cancel (same as FAK)


class OrderStatus(str, Enum):
    """Order status."""

    LIVE = "LIVE"
    MATCHED = "MATCHED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderBookLevel:
    """A single level in the order book."""

    price: Decimal
    size: Decimal

    @classmethod
    def from_api(cls, data: dict | list) -> "OrderBookLevel":
        """Create from API response."""
        if isinstance(data, list):
            return cls(price=Decimal(str(data[0])), size=Decimal(str(data[1])))
        return cls(
            price=Decimal(str(data.get("price", data.get("p", 0)))),
            size=Decimal(str(data.get("size", data.get("s", 0)))),
        )


@dataclass
class OrderBook:
    """Order book for a token."""

    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def midpoint(self) -> Optional[Decimal]:
        """Get midpoint price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def get_depth(self, side: OrderSide, price_levels: int = 5) -> Decimal:
        """Get total depth up to N price levels."""
        levels = self.bids if side == OrderSide.BUY else self.asks
        return sum(level.size for level in levels[:price_levels])

    @classmethod
    def from_api(cls, token_id: str, data: dict) -> "OrderBook":
        """Create from API response."""
        bids = [OrderBookLevel.from_api(b) for b in data.get("bids", [])]
        asks = [OrderBookLevel.from_api(a) for a in data.get("asks", [])]

        # Sort bids descending (best bid first), asks ascending (best ask first)
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return cls(
            token_id=token_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
        )


@dataclass
class PriceInfo:
    """Price information for a token."""

    token_id: str
    price: Decimal
    spread: Optional[Decimal] = None
    midpoint: Optional[Decimal] = None

    @classmethod
    def from_api(cls, token_id: str, data: dict) -> "PriceInfo":
        """Create from API response."""
        return cls(
            token_id=token_id,
            price=Decimal(str(data.get("price", 0))),
            spread=Decimal(str(data["spread"])) if data.get("spread") else None,
            midpoint=Decimal(str(data["mid"])) if data.get("mid") else None,
        )


@dataclass
class Order:
    """An order on the CLOB."""

    id: str
    token_id: str
    side: OrderSide
    price: Decimal
    size: Decimal
    size_matched: Decimal
    status: OrderStatus
    order_type: OrderType
    created_at: datetime
    expiration: Optional[datetime] = None

    @property
    def remaining_size(self) -> Decimal:
        """Get unfilled size."""
        return self.size - self.size_matched

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.size_matched >= self.size

    @classmethod
    def from_api(cls, data: dict) -> "Order":
        """Create from API response."""
        created = data.get("created_at", data.get("createdAt", 0))
        if isinstance(created, str):
            created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        else:
            created_at = datetime.fromtimestamp(created / 1000 if created > 1e10 else created)

        exp = data.get("expiration")
        expiration = None
        if exp:
            if isinstance(exp, str):
                expiration = datetime.fromisoformat(exp.replace("Z", "+00:00"))
            else:
                expiration = datetime.fromtimestamp(exp / 1000 if exp > 1e10 else exp)

        return cls(
            id=data.get("id", data.get("order_id", "")),
            token_id=data.get("asset_id", data.get("assetId", data.get("token_id", ""))),
            side=OrderSide(data.get("side", "BUY")),
            price=Decimal(str(data.get("price", 0))),
            size=Decimal(str(data.get("original_size", data.get("size", 0)))),
            size_matched=Decimal(str(data.get("size_matched", data.get("sizeMatched", 0)))),
            status=OrderStatus(data.get("status", "LIVE")),
            order_type=OrderType(data.get("type", data.get("order_type", "GTC"))),
            created_at=created_at,
            expiration=expiration,
        )


@dataclass
class MarketInfo:
    """Basic market information from CLOB."""

    condition_id: str
    token_id: str
    outcome: str
    tick_size: Decimal
    min_tick_size: Decimal
    active: bool = True

    @classmethod
    def from_api(cls, data: dict) -> "MarketInfo":
        """Create from API response."""
        return cls(
            condition_id=data.get("condition_id", data.get("conditionId", "")),
            token_id=data.get("token_id", data.get("tokenId", data.get("asset_id", ""))),
            outcome=data.get("outcome", ""),
            tick_size=Decimal(str(data.get("tick_size", data.get("tickSize", "0.01")))),
            min_tick_size=Decimal(str(data.get("minimum_tick_size", data.get("minTickSize", "0.001")))),
            active=data.get("active", True),
        )


class CLOBClient(BaseAPIClient):
    """Client for Polymarket CLOB API.

    Provides order book data, pricing, and order management.
    Requires authentication for order operations.

    Example:
        auth = PolymarketAuth(private_key="0x...")
        async with CLOBClient(auth=auth) as client:
            book = await client.get_order_book(token_id)
            order = await client.place_order(...)
    """

    DEFAULT_BASE_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        auth: Optional[PolymarketAuth] = None,
        timeout_s: float = 10.0,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize CLOB API client.

        Args:
            base_url: CLOB API base URL
            auth: Authentication handler (required for orders)
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

        self.auth = auth

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for request."""
        headers = await super()._get_headers()
        return headers

    async def _auth_request(
        self,
        method: str,
        path: str,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an authenticated request.

        Args:
            method: HTTP method
            path: Request path
            json_data: JSON body data
            params: Query parameters

        Returns:
            Response data

        Raises:
            AuthenticationError: If auth not configured
        """
        if not self.auth or not self.auth.has_credentials:
            from ..core.exceptions import AuthenticationError
            raise AuthenticationError("CLOB authentication not configured")

        # Build body string for signing
        body = json.dumps(json_data, separators=(",", ":")) if json_data else ""

        # Get auth headers
        auth_headers = self.auth.get_l2_headers(method, path, body)

        response = await self._request(
            method=method,
            path=path,
            json_data=json_data,
            params=params,
            headers=auth_headers,
        )

        return response.data

    # =========================================================================
    # Public endpoints (no auth required)
    # =========================================================================

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Get order book for a token.

        Args:
            token_id: The token ID (ERC1155 conditional token)

        Returns:
            OrderBook with bids and asks
        """
        response = await self.get("/book", params={"token_id": token_id})
        return OrderBook.from_api(token_id, response.data)

    async def get_order_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Get order books for multiple tokens.

        Args:
            token_ids: List of token IDs

        Returns:
            Dictionary mapping token_id to OrderBook
        """
        # API expects comma-separated token IDs
        response = await self.get("/books", params={"token_ids": ",".join(token_ids)})

        books = {}
        for token_id, book_data in response.data.items():
            books[token_id] = OrderBook.from_api(token_id, book_data)

        return books

    async def get_price(self, token_id: str, side: OrderSide) -> PriceInfo:
        """Get price for a token.

        Args:
            token_id: The token ID
            side: BUY or SELL

        Returns:
            PriceInfo with current price
        """
        response = await self.get("/price", params={
            "token_id": token_id,
            "side": side.value,
        })
        return PriceInfo.from_api(token_id, response.data)

    async def get_midpoint(self, token_id: str) -> Decimal:
        """Get midpoint price for a token.

        Args:
            token_id: The token ID

        Returns:
            Midpoint price
        """
        response = await self.get("/midpoint", params={"token_id": token_id})
        return Decimal(str(response.data.get("mid", 0)))

    async def get_spread(self, token_id: str) -> Decimal:
        """Get spread for a token.

        Args:
            token_id: The token ID

        Returns:
            Spread value
        """
        response = await self.get("/spread", params={"token_id": token_id})
        return Decimal(str(response.data.get("spread", 0)))

    async def get_tick_size(self, token_id: str) -> Decimal:
        """Get tick size for a token.

        Args:
            token_id: The token ID

        Returns:
            Tick size
        """
        response = await self.get("/tick-size", params={"token_id": token_id})
        return Decimal(str(response.data.get("minimum_tick_size", "0.01")))

    async def get_markets(self) -> list[MarketInfo]:
        """Get all active markets.

        Returns:
            List of MarketInfo objects
        """
        response = await self.get("/markets")

        markets = []
        data = response.data if isinstance(response.data, list) else response.data.get("markets", [])
        for item in data:
            try:
                markets.append(MarketInfo.from_api(item))
            except Exception as e:
                logger.warning("market_parse_error", error=str(e))

        return markets

    async def get_simplified_markets(self) -> list[dict]:
        """Get simplified market data.

        Returns:
            List of market dictionaries
        """
        response = await self.get("/simplified-markets")
        return response.data if isinstance(response.data, list) else response.data.get("markets", [])

    # =========================================================================
    # Authenticated endpoints
    # =========================================================================

    async def get_api_keys(self) -> list[dict]:
        """Get API keys for the authenticated user.

        Returns:
            List of API key metadata
        """
        return await self._auth_request("GET", "/auth/api-keys")

    async def get_open_orders(
        self,
        market: Optional[str] = None,
        token_id: Optional[str] = None,
    ) -> list[Order]:
        """Get open orders for the authenticated user.

        Args:
            market: Filter by market condition ID
            token_id: Filter by token ID

        Returns:
            List of Order objects
        """
        params = {}
        if market:
            params["market"] = market
        if token_id:
            params["asset_id"] = token_id

        data = await self._auth_request("GET", "/orders", params=params)

        orders = []
        order_list = data if isinstance(data, list) else data.get("orders", [])
        for item in order_list:
            try:
                orders.append(Order.from_api(item))
            except Exception as e:
                logger.warning("order_parse_error", error=str(e))

        return orders

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
        order_type: OrderType = OrderType.GTC,
        expiration: Optional[int] = None,
    ) -> dict:
        """Place a new order.

        Note: This is a simplified interface. For production use,
        you should use the py-clob-client library which handles
        order signing correctly.

        Args:
            token_id: Token ID to trade
            side: BUY or SELL
            price: Order price (0-1)
            size: Order size in shares
            order_type: Order type (GTC, GTD, FOK, IOC)
            expiration: Expiration timestamp for GTD orders

        Returns:
            Order response data
        """
        order_data = {
            "tokenID": token_id,
            "side": side.value,
            "price": str(price),
            "size": str(size),
            "type": order_type.value,
            "funderAddress": self.auth.funder_address if self.auth else None,
        }

        if expiration and order_type == OrderType.GTD:
            order_data["expiration"] = expiration

        return await self._auth_request("POST", "/order", json_data=order_data)

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order.

        Args:
            order_id: ID of order to cancel

        Returns:
            Cancellation response
        """
        return await self._auth_request("DELETE", "/order", json_data={"id": order_id})

    async def cancel_orders(self, order_ids: list[str]) -> dict:
        """Cancel multiple orders.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            Cancellation response
        """
        return await self._auth_request("DELETE", "/orders", json_data={"ids": order_ids})

    async def cancel_all_orders(self) -> dict:
        """Cancel all open orders.

        Returns:
            Cancellation response
        """
        return await self._auth_request("DELETE", "/cancel-all")

    async def get_trades_for_order(self, order_id: str) -> list[dict]:
        """Get trades for a specific order.

        Args:
            order_id: The order ID

        Returns:
            List of trade data
        """
        data = await self._auth_request("GET", f"/order/{order_id}/trades")
        return data if isinstance(data, list) else data.get("trades", [])
