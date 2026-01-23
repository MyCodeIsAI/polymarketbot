"""Custom exceptions for PolymarketBot.

This module defines a hierarchy of exceptions for precise error handling
throughout the application. Each exception carries context information
to aid debugging and logging.
"""

from decimal import Decimal
from typing import Any, Optional


class PolymarketBotError(Exception):
    """Base exception for all PolymarketBot errors.

    All custom exceptions inherit from this class, allowing broad
    exception catching when needed.
    """

    def __init__(self, message: str, **context: Any):
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            **self.context,
        }


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(PolymarketBotError):
    """Raised when configuration is invalid or missing."""

    pass


# =============================================================================
# API Errors
# =============================================================================


class APIError(PolymarketBotError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **context: Any,
    ):
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(
            message,
            endpoint=endpoint,
            status_code=status_code,
            **context,
        )


class AuthenticationError(APIError):
    """Raised when API authentication fails.

    This typically means:
    - Invalid API credentials
    - Expired credentials
    - Missing required headers
    """

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded.

    Contains information about when to retry.
    """

    def __init__(
        self,
        message: str,
        retry_after_ms: Optional[int] = None,
        **context: Any,
    ):
        self.retry_after_ms = retry_after_ms
        super().__init__(message, retry_after_ms=retry_after_ms, **context)


# =============================================================================
# Order Errors
# =============================================================================


class OrderError(PolymarketBotError):
    """Base class for order-related errors."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        market_id: Optional[str] = None,
        **context: Any,
    ):
        self.order_id = order_id
        self.market_id = market_id
        super().__init__(
            message,
            order_id=order_id,
            market_id=market_id,
            **context,
        )


class SlippageExceededError(OrderError):
    """Raised when price slippage exceeds the configured threshold.

    This is a safety mechanism to prevent executing at unfavorable prices.
    """

    def __init__(
        self,
        message: str,
        target_price: Decimal,
        current_price: Decimal,
        slippage_percent: Decimal,
        max_slippage: Decimal,
        **context: Any,
    ):
        self.target_price = target_price
        self.current_price = current_price
        self.slippage_percent = slippage_percent
        self.max_slippage = max_slippage
        super().__init__(
            message,
            target_price=str(target_price),
            current_price=str(current_price),
            slippage_percent=str(slippage_percent),
            max_slippage=str(max_slippage),
            **context,
        )


class InsufficientBalanceError(OrderError):
    """Raised when account balance is too low for an order."""

    def __init__(
        self,
        message: str,
        required: Decimal,
        available: Decimal,
        **context: Any,
    ):
        self.required = required
        self.available = available
        super().__init__(
            message,
            required=str(required),
            available=str(available),
            **context,
        )


class InsufficientLiquidityError(OrderError):
    """Raised when order book doesn't have enough depth."""

    def __init__(
        self,
        message: str,
        required_depth: Decimal,
        available_depth: Decimal,
        **context: Any,
    ):
        self.required_depth = required_depth
        self.available_depth = available_depth
        super().__init__(
            message,
            required_depth=str(required_depth),
            available_depth=str(available_depth),
            **context,
        )


class OrderTimeoutError(OrderError):
    """Raised when an order doesn't fill within the timeout period."""

    def __init__(
        self,
        message: str,
        timeout_seconds: int,
        **context: Any,
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, timeout_seconds=timeout_seconds, **context)


class OrderRejectedError(OrderError):
    """Raised when an order is rejected by the exchange."""

    def __init__(
        self,
        message: str,
        rejection_reason: Optional[str] = None,
        **context: Any,
    ):
        self.rejection_reason = rejection_reason
        super().__init__(message, rejection_reason=rejection_reason, **context)


# =============================================================================
# Position Errors
# =============================================================================


class PositionError(PolymarketBotError):
    """Base class for position-related errors."""

    def __init__(
        self,
        message: str,
        target_wallet: Optional[str] = None,
        market_id: Optional[str] = None,
        **context: Any,
    ):
        self.target_wallet = target_wallet
        self.market_id = market_id
        super().__init__(
            message,
            target_wallet=target_wallet,
            market_id=market_id,
            **context,
        )


class PositionDriftError(PositionError):
    """Raised when position drift exceeds acceptable threshold."""

    def __init__(
        self,
        message: str,
        expected_ratio: Decimal,
        actual_ratio: Decimal,
        drift_percent: Decimal,
        **context: Any,
    ):
        self.expected_ratio = expected_ratio
        self.actual_ratio = actual_ratio
        self.drift_percent = drift_percent
        super().__init__(
            message,
            expected_ratio=str(expected_ratio),
            actual_ratio=str(actual_ratio),
            drift_percent=str(drift_percent),
            **context,
        )


# =============================================================================
# Safety Errors
# =============================================================================


class CircuitBreakerTripped(PolymarketBotError):
    """Raised when a circuit breaker condition is triggered.

    This halts all trading until manually reset.
    """

    def __init__(
        self,
        message: str,
        breaker_name: str,
        trigger_value: Any,
        threshold: Any,
        **context: Any,
    ):
        self.breaker_name = breaker_name
        self.trigger_value = trigger_value
        self.threshold = threshold
        super().__init__(
            message,
            breaker_name=breaker_name,
            trigger_value=str(trigger_value),
            threshold=str(threshold),
            **context,
        )


# =============================================================================
# WebSocket Errors
# =============================================================================


class WebSocketError(PolymarketBotError):
    """Base class for WebSocket-related errors."""

    pass


class WebSocketConnectionError(WebSocketError):
    """Raised when WebSocket connection fails."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        reconnect_attempts: int = 0,
        **context: Any,
    ):
        self.url = url
        self.reconnect_attempts = reconnect_attempts
        super().__init__(
            message,
            url=url,
            reconnect_attempts=reconnect_attempts,
            **context,
        )


class WebSocketSubscriptionError(WebSocketError):
    """Raised when subscription to a WebSocket channel fails."""

    def __init__(
        self,
        message: str,
        channel: Optional[str] = None,
        **context: Any,
    ):
        self.channel = channel
        super().__init__(message, channel=channel, **context)
