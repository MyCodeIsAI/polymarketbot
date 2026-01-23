"""Circuit breaker system for trading protection.

This module provides:
- Circuit breaker framework with pluggable conditions
- Automatic trading halt on dangerous conditions
- State snapshot on trip
- Recovery procedures
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable, Awaitable, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Tripped, trading halted
    HALF_OPEN = "half_open"  # Testing recovery


class TripReason(str, Enum):
    """Reasons for circuit breaker trip."""

    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_FAILURES = "max_consecutive_failures"
    MAX_SLIPPAGE_EVENTS = "max_slippage_events"
    API_ERROR_RATE = "api_error_rate"
    POSITION_DRIFT = "position_drift"
    LOW_BALANCE = "low_balance"
    WEBSOCKET_DISCONNECTED = "websocket_disconnected"
    MANUAL = "manual"
    HEALTH_CHECK_FAILED = "health_check_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class TripEvent:
    """Record of a circuit breaker trip."""

    reason: TripReason
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Optional[str] = None
    condition_name: Optional[str] = None
    auto_reset: bool = False
    reset_after_s: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "reason": self.reason.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "condition_name": self.condition_name,
            "auto_reset": self.auto_reset,
        }


@dataclass
class StateSnapshot:
    """Snapshot of system state when circuit breaker trips."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    trip_event: Optional[TripEvent] = None

    # Balance state
    total_balance: Decimal = Decimal("0")
    available_balance: Decimal = Decimal("0")

    # Position state
    open_positions: int = 0
    total_position_value: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Order state
    pending_orders: int = 0
    orders_cancelled: int = 0

    # P&L state
    daily_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    # System state
    api_errors_recent: int = 0
    slippage_events_recent: int = 0
    consecutive_failures: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trip_event": self.trip_event.to_dict() if self.trip_event else None,
            "balance": {
                "total": str(self.total_balance),
                "available": str(self.available_balance),
            },
            "positions": {
                "open": self.open_positions,
                "value": str(self.total_position_value),
                "unrealized_pnl": str(self.unrealized_pnl),
            },
            "orders": {
                "pending": self.pending_orders,
                "cancelled": self.orders_cancelled,
            },
            "pnl": {
                "daily": str(self.daily_pnl),
                "realized": str(self.realized_pnl),
            },
            "errors": {
                "api_errors": self.api_errors_recent,
                "slippage_events": self.slippage_events_recent,
                "consecutive_failures": self.consecutive_failures,
            },
        }


# Callback types
TripCallback = Callable[[TripEvent], Awaitable[None]]
ResetCallback = Callable[[], Awaitable[None]]
SnapshotProvider = Callable[[], Awaitable[StateSnapshot]]


class BreakerCondition(ABC):
    """Base class for circuit breaker conditions.

    Subclass this to create custom trip conditions.
    """

    def __init__(
        self,
        name: str,
        reason: TripReason,
        auto_reset: bool = False,
        reset_after_s: Optional[int] = None,
    ):
        """Initialize condition.

        Args:
            name: Condition name for logging
            reason: Trip reason
            auto_reset: Whether to auto-reset after cooldown
            reset_after_s: Seconds before auto-reset
        """
        self.name = name
        self.reason = reason
        self.auto_reset = auto_reset
        self.reset_after_s = reset_after_s

    @abstractmethod
    async def is_triggered(self) -> bool:
        """Check if condition is triggered.

        Returns:
            True if condition is triggered
        """
        pass

    @abstractmethod
    def get_details(self) -> str:
        """Get details about current state.

        Returns:
            Human-readable details
        """
        pass

    def create_trip_event(self) -> TripEvent:
        """Create trip event for this condition.

        Returns:
            TripEvent with condition details
        """
        return TripEvent(
            reason=self.reason,
            details=self.get_details(),
            condition_name=self.name,
            auto_reset=self.auto_reset,
            reset_after_s=self.reset_after_s,
        )


class CircuitBreaker:
    """Main circuit breaker coordinator.

    Monitors multiple conditions and trips when any is triggered.
    Handles trading halt, order cancellation, and recovery.

    Example:
        breaker = CircuitBreaker()
        breaker.add_condition(MaxDailyLoss(threshold=Decimal("500")))
        breaker.add_condition(MaxConsecutiveFailures(count=5))

        await breaker.start()

        # Check before trading
        if breaker.is_open:
            return  # Don't trade

        # Manual trip
        await breaker.trip(TripReason.MANUAL, "User requested halt")
    """

    def __init__(
        self,
        check_interval_s: float = 1.0,
        on_trip: Optional[TripCallback] = None,
        on_reset: Optional[ResetCallback] = None,
        snapshot_provider: Optional[SnapshotProvider] = None,
    ):
        """Initialize circuit breaker.

        Args:
            check_interval_s: Interval between condition checks
            on_trip: Callback when breaker trips
            on_reset: Callback when breaker resets
            snapshot_provider: Provides state snapshots
        """
        self.check_interval_s = check_interval_s
        self._on_trip = on_trip
        self._on_reset = on_reset
        self._snapshot_provider = snapshot_provider

        # State
        self._state = BreakerState.CLOSED
        self._conditions: list[BreakerCondition] = []
        self._trip_history: list[TripEvent] = []
        self._snapshots: list[StateSnapshot] = []

        # Current trip info
        self._current_trip: Optional[TripEvent] = None
        self._tripped_at: Optional[datetime] = None

        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reset_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> BreakerState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if breaker is closed (normal operation)."""
        return self._state == BreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if breaker is open (trading halted)."""
        return self._state == BreakerState.OPEN

    @property
    def current_trip(self) -> Optional[TripEvent]:
        """Get current trip event if open."""
        return self._current_trip

    def add_condition(self, condition: BreakerCondition) -> None:
        """Add a trip condition.

        Args:
            condition: Condition to add
        """
        self._conditions.append(condition)
        logger.debug(
            "condition_added",
            name=condition.name,
            reason=condition.reason.value,
        )

    def remove_condition(self, name: str) -> bool:
        """Remove a condition by name.

        Args:
            name: Condition name

        Returns:
            True if removed
        """
        for i, cond in enumerate(self._conditions):
            if cond.name == name:
                self._conditions.pop(i)
                return True
        return False

    async def start(self) -> None:
        """Start circuit breaker monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

        logger.info(
            "circuit_breaker_started",
            conditions=len(self._conditions),
            check_interval_s=self.check_interval_s,
        )

    async def stop(self) -> None:
        """Stop circuit breaker monitoring."""
        if not self._running:
            return

        self._running = False

        for task in [self._task, self._reset_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(
            "circuit_breaker_stopped",
            trips=len(self._trip_history),
        )

    async def trip(
        self,
        reason: TripReason,
        details: Optional[str] = None,
        auto_reset: bool = False,
        reset_after_s: Optional[int] = None,
    ) -> None:
        """Manually trip the circuit breaker.

        Args:
            reason: Trip reason
            details: Additional details
            auto_reset: Whether to auto-reset
            reset_after_s: Seconds before auto-reset
        """
        if self._state == BreakerState.OPEN:
            logger.warning("already_tripped", current_reason=self._current_trip.reason.value)
            return

        event = TripEvent(
            reason=reason,
            details=details,
            auto_reset=auto_reset,
            reset_after_s=reset_after_s,
        )

        await self._execute_trip(event)

    async def reset(self) -> bool:
        """Reset the circuit breaker to closed state.

        Returns:
            True if reset successful
        """
        if self._state == BreakerState.CLOSED:
            return True

        # Check if any conditions are still triggered
        for condition in self._conditions:
            try:
                if await condition.is_triggered():
                    logger.warning(
                        "reset_blocked",
                        condition=condition.name,
                    )
                    return False
            except Exception as e:
                logger.error(
                    "condition_check_error_on_reset",
                    condition=condition.name,
                    error=str(e),
                )

        # Reset
        self._state = BreakerState.CLOSED
        self._current_trip = None
        self._tripped_at = None

        if self._on_reset:
            try:
                await self._on_reset()
            except Exception as e:
                logger.error("reset_callback_error", error=str(e))

        logger.info("circuit_breaker_reset")
        return True

    async def force_reset(self) -> None:
        """Force reset without checking conditions."""
        self._state = BreakerState.CLOSED
        self._current_trip = None
        self._tripped_at = None

        if self._reset_task:
            self._reset_task.cancel()

        logger.warning("circuit_breaker_force_reset")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                if self._state == BreakerState.CLOSED:
                    await self._check_conditions()

                await asyncio.sleep(self.check_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitor_loop_error", error=str(e))
                await asyncio.sleep(self.check_interval_s)

    async def _check_conditions(self) -> None:
        """Check all conditions."""
        for condition in self._conditions:
            try:
                if await condition.is_triggered():
                    event = condition.create_trip_event()
                    await self._execute_trip(event)
                    return  # Only need to trip once
            except Exception as e:
                logger.error(
                    "condition_check_error",
                    condition=condition.name,
                    error=str(e),
                )

    async def _execute_trip(self, event: TripEvent) -> None:
        """Execute a circuit breaker trip.

        Args:
            event: Trip event
        """
        self._state = BreakerState.OPEN
        self._current_trip = event
        self._tripped_at = datetime.utcnow()
        self._trip_history.append(event)

        logger.critical(
            "CIRCUIT_BREAKER_TRIPPED",
            reason=event.reason.value,
            details=event.details,
            condition=event.condition_name,
        )

        # Take snapshot
        if self._snapshot_provider:
            try:
                snapshot = await self._snapshot_provider()
                snapshot.trip_event = event
                self._snapshots.append(snapshot)

                logger.info(
                    "state_snapshot_captured",
                    balance=str(snapshot.total_balance),
                    positions=snapshot.open_positions,
                )
            except Exception as e:
                logger.error("snapshot_error", error=str(e))

        # Notify callback
        if self._on_trip:
            try:
                await self._on_trip(event)
            except Exception as e:
                logger.error("trip_callback_error", error=str(e))

        # Schedule auto-reset if configured
        if event.auto_reset and event.reset_after_s:
            self._reset_task = asyncio.create_task(
                self._auto_reset(event.reset_after_s)
            )

    async def _auto_reset(self, delay_s: int) -> None:
        """Auto-reset after delay.

        Args:
            delay_s: Delay before reset
        """
        try:
            await asyncio.sleep(delay_s)

            if self._state == BreakerState.OPEN:
                logger.info("attempting_auto_reset", delay_s=delay_s)
                await self.reset()

        except asyncio.CancelledError:
            pass

    def get_trip_history(self, limit: int = 50) -> list[TripEvent]:
        """Get trip history.

        Args:
            limit: Maximum events to return

        Returns:
            List of trip events
        """
        return self._trip_history[-limit:]

    def get_snapshots(self, limit: int = 10) -> list[StateSnapshot]:
        """Get state snapshots.

        Args:
            limit: Maximum snapshots to return

        Returns:
            List of snapshots
        """
        return self._snapshots[-limit:]

    def get_status(self) -> dict:
        """Get current status.

        Returns:
            Status dictionary
        """
        return {
            "state": self._state.value,
            "current_trip": self._current_trip.to_dict() if self._current_trip else None,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "conditions": [c.name for c in self._conditions],
            "trip_count": len(self._trip_history),
        }


class CompositeCircuitBreaker:
    """Combines multiple circuit breakers.

    Useful for having separate breakers for different
    subsystems (e.g., per-target breakers).
    """

    def __init__(self):
        """Initialize composite breaker."""
        self._breakers: dict[str, CircuitBreaker] = {}

    def add_breaker(self, name: str, breaker: CircuitBreaker) -> None:
        """Add a named breaker.

        Args:
            name: Breaker name
            breaker: Circuit breaker
        """
        self._breakers[name] = breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a breaker by name.

        Args:
            name: Breaker name

        Returns:
            CircuitBreaker or None
        """
        return self._breakers.get(name)

    @property
    def any_open(self) -> bool:
        """Check if any breaker is open."""
        return any(b.is_open for b in self._breakers.values())

    @property
    def all_closed(self) -> bool:
        """Check if all breakers are closed."""
        return all(b.is_closed for b in self._breakers.values())

    async def start_all(self) -> None:
        """Start all breakers."""
        for breaker in self._breakers.values():
            await breaker.start()

    async def stop_all(self) -> None:
        """Stop all breakers."""
        for breaker in self._breakers.values():
            await breaker.stop()

    async def trip_all(self, reason: TripReason, details: str) -> None:
        """Trip all breakers.

        Args:
            reason: Trip reason
            details: Trip details
        """
        for name, breaker in self._breakers.items():
            if breaker.is_closed:
                await breaker.trip(reason, f"{details} (breaker: {name})")

    def get_status(self) -> dict:
        """Get status of all breakers.

        Returns:
            Status dictionary
        """
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
