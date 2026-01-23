"""Core components for PolymarketBot."""

from .exceptions import (
    PolymarketBotError,
    ConfigurationError,
    APIError,
    AuthenticationError,
    RateLimitError,
    OrderError,
    SlippageExceededError,
    InsufficientBalanceError,
    InsufficientLiquidityError,
    PositionError,
    CircuitBreakerTripped,
    WebSocketError,
)

__all__ = [
    "PolymarketBotError",
    "ConfigurationError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "OrderError",
    "SlippageExceededError",
    "InsufficientBalanceError",
    "InsufficientLiquidityError",
    "PositionError",
    "CircuitBreakerTripped",
    "WebSocketError",
]
