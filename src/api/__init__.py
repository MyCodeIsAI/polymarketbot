"""API clients for Polymarket services.

This module provides high-performance async clients for all Polymarket APIs:
- DataAPIClient: Positions, activity, trades
- CLOBClient: Order book, orders, pricing
- GammaAPIClient: Market discovery, events

All clients support:
- Connection pooling
- Automatic retries with exponential backoff
- Rate limiting
- Structured logging
"""

from .base import BaseAPIClient, APIResponse
from .rate_limiter import RateLimiter, AdaptiveRateLimiter, RateLimitConfig
from .auth import (
    PolymarketAuth,
    APICredentials,
    L1Authenticator,
    L2Authenticator,
    load_credentials_from_env,
    load_private_key_from_env,
)
from .data import (
    DataAPIClient,
    Position,
    Activity,
    Trade,
    ActivityType,
    TradeSide,
    PositionSortBy,
    SortDirection,
)
from .clob import (
    CLOBClient,
    OrderBook,
    OrderBookLevel,
    PriceInfo,
    Order,
    MarketInfo,
    OrderSide,
    OrderType,
    OrderStatus,
)
from .gamma import (
    GammaAPIClient,
    Event,
    Market,
    Token,
)
from .connectivity import (
    test_all_endpoints,
    test_endpoint,
    test_polygon_rpc,
    ConnectivityResult,
    format_connectivity_results,
)

__all__ = [
    # Base
    "BaseAPIClient",
    "APIResponse",
    # Rate limiting
    "RateLimiter",
    "AdaptiveRateLimiter",
    "RateLimitConfig",
    # Authentication
    "PolymarketAuth",
    "APICredentials",
    "L1Authenticator",
    "L2Authenticator",
    "load_credentials_from_env",
    "load_private_key_from_env",
    # Data API
    "DataAPIClient",
    "Position",
    "Activity",
    "Trade",
    "ActivityType",
    "TradeSide",
    "PositionSortBy",
    "SortDirection",
    # CLOB API
    "CLOBClient",
    "OrderBook",
    "OrderBookLevel",
    "PriceInfo",
    "Order",
    "MarketInfo",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Gamma API
    "GammaAPIClient",
    "Event",
    "Market",
    "Token",
    # Connectivity
    "test_all_endpoints",
    "test_endpoint",
    "test_polygon_rpc",
    "ConnectivityResult",
    "format_connectivity_results",
]
