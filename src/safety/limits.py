"""Trading limits and circuit breaker conditions.

This module provides concrete implementations of circuit breaker
conditions for various risk scenarios.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, Awaitable

from ..utils.logging import get_logger
from .circuit_breaker import BreakerCondition, TripReason

logger = get_logger(__name__)


# =============================================================================
# Value Providers (dependency injection for conditions)
# =============================================================================

# Type aliases for value providers
BalanceProvider = Callable[[], Awaitable[Decimal]]
PnLProvider = Callable[[], Awaitable[Decimal]]
DriftProvider = Callable[[], Awaitable[Decimal]]
IntProvider = Callable[[], Awaitable[int]]
BoolProvider = Callable[[], Awaitable[bool]]


# =============================================================================
# Loss-Based Conditions
# =============================================================================


class MaxDailyLoss(BreakerCondition):
    """Trips when daily P&L loss exceeds threshold.

    Example:
        condition = MaxDailyLoss(
            threshold_usd=Decimal("500"),
            pnl_provider=get_daily_pnl,
        )
    """

    def __init__(
        self,
        threshold_usd: Decimal,
        pnl_provider: PnLProvider,
        auto_reset: bool = False,
    ):
        """Initialize condition.

        Args:
            threshold_usd: Maximum loss in USD before trip
            pnl_provider: Async function that returns daily P&L
            auto_reset: Whether to auto-reset at day boundary
        """
        super().__init__(
            name="max_daily_loss",
            reason=TripReason.MAX_DAILY_LOSS,
            auto_reset=auto_reset,
        )
        self.threshold_usd = threshold_usd
        self._pnl_provider = pnl_provider
        self._last_pnl: Optional[Decimal] = None

    async def is_triggered(self) -> bool:
        """Check if daily loss exceeds threshold."""
        pnl = await self._pnl_provider()
        self._last_pnl = pnl

        # Negative P&L means loss
        if pnl < -self.threshold_usd:
            logger.warning(
                "daily_loss_limit_triggered",
                loss=str(abs(pnl)),
                threshold=str(self.threshold_usd),
            )
            return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        return f"Daily loss ${abs(self._last_pnl):.2f} exceeded limit ${self.threshold_usd:.2f}"


class MaxDrawdown(BreakerCondition):
    """Trips when drawdown from peak exceeds threshold.

    Tracks high-water mark and calculates drawdown from it.
    """

    def __init__(
        self,
        threshold_percent: Decimal,
        balance_provider: BalanceProvider,
    ):
        """Initialize condition.

        Args:
            threshold_percent: Max drawdown percentage (e.g., 0.10 = 10%)
            balance_provider: Async function that returns current balance
        """
        super().__init__(
            name="max_drawdown",
            reason=TripReason.MAX_DAILY_LOSS,
        )
        self.threshold_percent = threshold_percent
        self._balance_provider = balance_provider
        self._high_water_mark = Decimal("0")
        self._current_balance: Optional[Decimal] = None
        self._current_drawdown: Optional[Decimal] = None

    async def is_triggered(self) -> bool:
        """Check if drawdown exceeds threshold."""
        balance = await self._balance_provider()
        self._current_balance = balance

        # Update high water mark
        if balance > self._high_water_mark:
            self._high_water_mark = balance

        # Calculate drawdown
        if self._high_water_mark > 0:
            drawdown = (self._high_water_mark - balance) / self._high_water_mark
            self._current_drawdown = drawdown

            if drawdown > self.threshold_percent:
                logger.warning(
                    "drawdown_limit_triggered",
                    drawdown_percent=f"{drawdown*100:.1f}%",
                    threshold_percent=f"{self.threshold_percent*100:.1f}%",
                )
                return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        dd = self._current_drawdown or Decimal("0")
        return f"Drawdown {dd*100:.1f}% exceeded limit {self.threshold_percent*100:.1f}%"


# =============================================================================
# Failure-Based Conditions
# =============================================================================


class MaxConsecutiveFailures(BreakerCondition):
    """Trips when consecutive order failures exceed threshold.

    Tracks consecutive failures and resets on success.
    """

    def __init__(
        self,
        max_failures: int,
        auto_reset: bool = True,
        reset_after_s: int = 300,
    ):
        """Initialize condition.

        Args:
            max_failures: Maximum consecutive failures before trip
            auto_reset: Whether to auto-reset
            reset_after_s: Seconds before auto-reset
        """
        super().__init__(
            name="max_consecutive_failures",
            reason=TripReason.MAX_CONSECUTIVE_FAILURES,
            auto_reset=auto_reset,
            reset_after_s=reset_after_s,
        )
        self.max_failures = max_failures
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failure."""
        self._failure_count += 1
        logger.debug(
            "failure_recorded",
            count=self._failure_count,
            max=self.max_failures,
        )

    def record_success(self) -> None:
        """Record a success (resets counter)."""
        if self._failure_count > 0:
            logger.debug("failure_count_reset")
        self._failure_count = 0

    async def is_triggered(self) -> bool:
        """Check if consecutive failures exceed threshold."""
        if self._failure_count >= self.max_failures:
            logger.warning(
                "consecutive_failures_triggered",
                count=self._failure_count,
                max=self.max_failures,
            )
            return True
        return False

    def get_details(self) -> str:
        """Get details about current state."""
        return f"{self._failure_count} consecutive failures (limit: {self.max_failures})"

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count


class MaxSlippageEvents(BreakerCondition):
    """Trips when too many slippage events occur in a window.

    Tracks slippage events with timestamps in a sliding window.
    """

    def __init__(
        self,
        max_events: int,
        window_minutes: int,
        auto_reset: bool = True,
        reset_after_s: int = 600,
    ):
        """Initialize condition.

        Args:
            max_events: Maximum slippage events in window
            window_minutes: Time window in minutes
            auto_reset: Whether to auto-reset
            reset_after_s: Seconds before auto-reset
        """
        super().__init__(
            name="max_slippage_events",
            reason=TripReason.MAX_SLIPPAGE_EVENTS,
            auto_reset=auto_reset,
            reset_after_s=reset_after_s,
        )
        self.max_events = max_events
        self.window_minutes = window_minutes
        self._events: deque = deque()

    def record_slippage_event(self, slippage_percent: Optional[Decimal] = None) -> None:
        """Record a slippage event.

        Args:
            slippage_percent: Optional slippage percentage for logging
        """
        self._events.append(datetime.utcnow())
        self._cleanup_old_events()

        logger.debug(
            "slippage_event_recorded",
            count=len(self._events),
            max=self.max_events,
            slippage=f"{slippage_percent*100:.1f}%" if slippage_percent else None,
        )

    def _cleanup_old_events(self) -> None:
        """Remove events outside the window."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    async def is_triggered(self) -> bool:
        """Check if slippage events exceed threshold."""
        self._cleanup_old_events()

        if len(self._events) >= self.max_events:
            logger.warning(
                "slippage_events_triggered",
                count=len(self._events),
                max=self.max_events,
                window_minutes=self.window_minutes,
            )
            return True
        return False

    def get_details(self) -> str:
        """Get details about current state."""
        return f"{len(self._events)} slippage events in {self.window_minutes}min (limit: {self.max_events})"

    @property
    def event_count(self) -> int:
        """Get current event count in window."""
        self._cleanup_old_events()
        return len(self._events)


class APIErrorRate(BreakerCondition):
    """Trips when API error rate exceeds threshold.

    Tracks success/failure rate over a sliding window.
    """

    def __init__(
        self,
        threshold: Decimal,
        window_minutes: int,
        min_requests: int = 10,
        auto_reset: bool = True,
        reset_after_s: int = 300,
    ):
        """Initialize condition.

        Args:
            threshold: Error rate threshold (0.5 = 50%)
            window_minutes: Time window for rate calculation
            min_requests: Minimum requests before checking rate
            auto_reset: Whether to auto-reset
            reset_after_s: Seconds before auto-reset
        """
        super().__init__(
            name="api_error_rate",
            reason=TripReason.API_ERROR_RATE,
            auto_reset=auto_reset,
            reset_after_s=reset_after_s,
        )
        self.threshold = threshold
        self.window_minutes = window_minutes
        self.min_requests = min_requests

        self._successes: deque = deque()
        self._failures: deque = deque()
        self._current_rate: Optional[Decimal] = None

    def record_success(self) -> None:
        """Record a successful API call."""
        self._successes.append(datetime.utcnow())
        self._cleanup_old()

    def record_failure(self) -> None:
        """Record a failed API call."""
        self._failures.append(datetime.utcnow())
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        """Remove events outside the window."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)

        while self._successes and self._successes[0] < cutoff:
            self._successes.popleft()

        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    async def is_triggered(self) -> bool:
        """Check if error rate exceeds threshold."""
        self._cleanup_old()

        total = len(self._successes) + len(self._failures)
        if total < self.min_requests:
            return False

        error_rate = Decimal(len(self._failures)) / Decimal(total)
        self._current_rate = error_rate

        if error_rate > self.threshold:
            logger.warning(
                "api_error_rate_triggered",
                error_rate=f"{error_rate*100:.1f}%",
                threshold=f"{self.threshold*100:.1f}%",
                failures=len(self._failures),
                total=total,
            )
            return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        rate = self._current_rate or Decimal("0")
        return f"API error rate {rate*100:.1f}% exceeded threshold {self.threshold*100:.1f}%"

    @property
    def error_rate(self) -> Decimal:
        """Get current error rate."""
        self._cleanup_old()
        total = len(self._successes) + len(self._failures)
        if total == 0:
            return Decimal("0")
        return Decimal(len(self._failures)) / Decimal(total)


# =============================================================================
# Position-Based Conditions
# =============================================================================


class PositionDriftTooHigh(BreakerCondition):
    """Trips when position drift exceeds threshold.

    Checks maximum drift across all positions.
    """

    def __init__(
        self,
        threshold: Decimal,
        drift_provider: DriftProvider,
    ):
        """Initialize condition.

        Args:
            threshold: Maximum drift as decimal (0.2 = 20%)
            drift_provider: Async function returning max drift
        """
        super().__init__(
            name="position_drift_too_high",
            reason=TripReason.POSITION_DRIFT,
        )
        self.threshold = threshold
        self._drift_provider = drift_provider
        self._current_drift: Optional[Decimal] = None

    async def is_triggered(self) -> bool:
        """Check if position drift exceeds threshold."""
        drift = await self._drift_provider()
        self._current_drift = drift

        if drift > self.threshold:
            logger.warning(
                "position_drift_triggered",
                drift=f"{drift*100:.1f}%",
                threshold=f"{self.threshold*100:.1f}%",
            )
            return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        drift = self._current_drift or Decimal("0")
        return f"Position drift {drift*100:.1f}% exceeded threshold {self.threshold*100:.1f}%"


class BalanceTooLow(BreakerCondition):
    """Trips when balance falls below threshold.

    Prevents trading when balance is too low.
    """

    def __init__(
        self,
        threshold_usd: Decimal,
        balance_provider: BalanceProvider,
    ):
        """Initialize condition.

        Args:
            threshold_usd: Minimum balance in USD
            balance_provider: Async function returning balance
        """
        super().__init__(
            name="balance_too_low",
            reason=TripReason.LOW_BALANCE,
        )
        self.threshold_usd = threshold_usd
        self._balance_provider = balance_provider
        self._current_balance: Optional[Decimal] = None

    async def is_triggered(self) -> bool:
        """Check if balance is too low."""
        balance = await self._balance_provider()
        self._current_balance = balance

        if balance < self.threshold_usd:
            logger.warning(
                "balance_too_low_triggered",
                balance=str(balance),
                threshold=str(self.threshold_usd),
            )
            return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        bal = self._current_balance or Decimal("0")
        return f"Balance ${bal:.2f} below minimum ${self.threshold_usd:.2f}"


# =============================================================================
# Connection-Based Conditions
# =============================================================================


class WebSocketDisconnected(BreakerCondition):
    """Trips when WebSocket has been disconnected too long.

    Monitors WebSocket connection health.
    """

    def __init__(
        self,
        max_disconnect_s: int,
        connected_provider: BoolProvider,
        auto_reset: bool = True,
        reset_after_s: int = 60,
    ):
        """Initialize condition.

        Args:
            max_disconnect_s: Maximum disconnect time in seconds
            connected_provider: Async function returning connection status
            auto_reset: Whether to auto-reset when reconnected
            reset_after_s: Seconds before auto-reset check
        """
        super().__init__(
            name="websocket_disconnected",
            reason=TripReason.WEBSOCKET_DISCONNECTED,
            auto_reset=auto_reset,
            reset_after_s=reset_after_s,
        )
        self.max_disconnect_s = max_disconnect_s
        self._connected_provider = connected_provider
        self._disconnected_since: Optional[datetime] = None

    async def is_triggered(self) -> bool:
        """Check if WebSocket has been disconnected too long."""
        is_connected = await self._connected_provider()

        if is_connected:
            self._disconnected_since = None
            return False

        now = datetime.utcnow()

        if self._disconnected_since is None:
            self._disconnected_since = now
            return False

        disconnect_duration = (now - self._disconnected_since).total_seconds()

        if disconnect_duration > self.max_disconnect_s:
            logger.warning(
                "websocket_disconnect_triggered",
                duration_s=disconnect_duration,
                max_s=self.max_disconnect_s,
            )
            return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        if self._disconnected_since:
            duration = (datetime.utcnow() - self._disconnected_since).total_seconds()
            return f"WebSocket disconnected for {duration:.0f}s (max: {self.max_disconnect_s}s)"
        return "WebSocket connected"


class RateLimitExceeded(BreakerCondition):
    """Trips when rate limits are critically low.

    Monitors rate limit headroom to prevent API blocks.
    """

    def __init__(
        self,
        min_headroom_percent: Decimal,
        headroom_provider: Callable[[], Awaitable[Decimal]],
    ):
        """Initialize condition.

        Args:
            min_headroom_percent: Minimum headroom (0.1 = 10%)
            headroom_provider: Async function returning headroom percent
        """
        super().__init__(
            name="rate_limit_exceeded",
            reason=TripReason.RATE_LIMIT_EXCEEDED,
            auto_reset=True,
            reset_after_s=60,
        )
        self.min_headroom_percent = min_headroom_percent
        self._headroom_provider = headroom_provider
        self._current_headroom: Optional[Decimal] = None

    async def is_triggered(self) -> bool:
        """Check if rate limit headroom is too low."""
        headroom = await self._headroom_provider()
        self._current_headroom = headroom

        if headroom < self.min_headroom_percent:
            logger.warning(
                "rate_limit_headroom_low",
                headroom=f"{headroom*100:.1f}%",
                min=f"{self.min_headroom_percent*100:.1f}%",
            )
            return True

        return False

    def get_details(self) -> str:
        """Get details about current state."""
        headroom = self._current_headroom or Decimal("0")
        return f"Rate limit headroom {headroom*100:.1f}% below minimum {self.min_headroom_percent*100:.1f}%"
