"""WebSocket module for real-time data feeds.

This module provides:
- WebSocket client with auto-reconnection
- Market feed for order book updates
- User feed for authenticated order/trade updates
- Local order book maintenance
- Connection health monitoring
"""

from .client import (
    WebSocketClient,
    AuthenticatedWebSocketClient,
    ConnectionState,
    ReconnectConfig,
    ConnectionStats,
    WS_ENDPOINT,
    WS_ENDPOINT_USER,
)
from .market_feed import (
    MarketFeed,
    MultiMarketFeed,
    LocalOrderBook,
    BookLevel,
    BookUpdate,
    SubscriptionStats,
)
from .user_feed import (
    UserFeed,
    OrderUpdate,
    TradeUpdate,
    PositionUpdate,
    OrderStatus,
    FillType,
    FillTracker,
    UserFeedStats,
)
from .health import (
    HealthMonitor,
    HealthStatus,
    ConnectionHealth,
    SystemHealth,
    LatencyMetrics,
    LatencyTracker,
    ConnectionWatchdog,
)

__all__ = [
    # Client
    "WebSocketClient",
    "AuthenticatedWebSocketClient",
    "ConnectionState",
    "ReconnectConfig",
    "ConnectionStats",
    "WS_ENDPOINT",
    "WS_ENDPOINT_USER",
    # Market feed
    "MarketFeed",
    "MultiMarketFeed",
    "LocalOrderBook",
    "BookLevel",
    "BookUpdate",
    "SubscriptionStats",
    # User feed
    "UserFeed",
    "OrderUpdate",
    "TradeUpdate",
    "PositionUpdate",
    "OrderStatus",
    "FillType",
    "FillTracker",
    "UserFeedStats",
    # Health
    "HealthMonitor",
    "HealthStatus",
    "ConnectionHealth",
    "SystemHealth",
    "LatencyMetrics",
    "LatencyTracker",
    "ConnectionWatchdog",
]
