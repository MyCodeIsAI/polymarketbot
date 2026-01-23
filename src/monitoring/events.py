"""Event types and event bus for position monitoring.

This module defines the events emitted when position changes are detected,
and provides an async event bus for publishing and subscribing to these events.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Coroutine, Optional
from uuid import uuid4

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Types of monitoring events."""

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_INCREASED = "position_increased"
    POSITION_DECREASED = "position_decreased"
    POSITION_CLOSED = "position_closed"

    # Trade events
    TRADE_DETECTED = "trade_detected"

    # System events
    MONITORING_STARTED = "monitoring_started"
    MONITORING_STOPPED = "monitoring_stopped"
    SYNC_COMPLETED = "sync_completed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class BaseEvent(ABC):
    """Base class for all events."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Return the event type."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for logging/serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeDetectedEvent(BaseEvent):
    """Emitted when a new trade is detected for a target wallet."""

    target_name: str
    target_wallet: str
    condition_id: str
    token_id: str
    outcome: str
    side: str  # BUY or SELL
    size: Decimal
    price: Decimal
    usd_value: Decimal
    trade_timestamp: datetime
    activity_id: Optional[str] = None

    @property
    def event_type(self) -> EventType:
        return EventType.TRADE_DETECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "target_name": self.target_name,
            "target_wallet": self.target_wallet,
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "usd_value": str(self.usd_value),
            "trade_timestamp": self.trade_timestamp.isoformat(),
        }


@dataclass
class PositionOpenedEvent(BaseEvent):
    """Emitted when a target opens a new position."""

    target_name: str
    target_wallet: str
    condition_id: str
    token_id: str
    outcome: str
    size: Decimal
    entry_price: Decimal
    usd_value: Decimal

    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_OPENED

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "target_name": self.target_name,
            "target_wallet": self.target_wallet,
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "usd_value": str(self.usd_value),
        }


@dataclass
class PositionIncreasedEvent(BaseEvent):
    """Emitted when a target increases an existing position."""

    target_name: str
    target_wallet: str
    condition_id: str
    token_id: str
    outcome: str
    previous_size: Decimal
    new_size: Decimal
    size_delta: Decimal
    price: Decimal
    usd_value: Decimal

    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_INCREASED

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "target_name": self.target_name,
            "target_wallet": self.target_wallet,
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "previous_size": str(self.previous_size),
            "new_size": str(self.new_size),
            "size_delta": str(self.size_delta),
            "price": str(self.price),
            "usd_value": str(self.usd_value),
        }


@dataclass
class PositionDecreasedEvent(BaseEvent):
    """Emitted when a target decreases an existing position."""

    target_name: str
    target_wallet: str
    condition_id: str
    token_id: str
    outcome: str
    previous_size: Decimal
    new_size: Decimal
    size_delta: Decimal
    price: Decimal
    usd_value: Decimal

    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_DECREASED

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "target_name": self.target_name,
            "target_wallet": self.target_wallet,
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "previous_size": str(self.previous_size),
            "new_size": str(self.new_size),
            "size_delta": str(self.size_delta),
            "price": str(self.price),
            "usd_value": str(self.usd_value),
        }


@dataclass
class PositionClosedEvent(BaseEvent):
    """Emitted when a target closes a position entirely."""

    target_name: str
    target_wallet: str
    condition_id: str
    token_id: str
    outcome: str
    closed_size: Decimal
    exit_price: Decimal
    usd_value: Decimal
    realized_pnl: Optional[Decimal] = None

    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_CLOSED

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "target_name": self.target_name,
            "target_wallet": self.target_wallet,
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "closed_size": str(self.closed_size),
            "exit_price": str(self.exit_price),
            "usd_value": str(self.usd_value),
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl else None,
        }


@dataclass
class MonitoringStartedEvent(BaseEvent):
    """Emitted when monitoring starts for a target."""

    target_name: str
    target_wallet: str
    position_count: int = 0

    @property
    def event_type(self) -> EventType:
        return EventType.MONITORING_STARTED


@dataclass
class MonitoringStoppedEvent(BaseEvent):
    """Emitted when monitoring stops for a target."""

    target_name: str
    target_wallet: str
    reason: str = "manual"

    @property
    def event_type(self) -> EventType:
        return EventType.MONITORING_STOPPED


@dataclass
class SyncCompletedEvent(BaseEvent):
    """Emitted when a full position sync completes."""

    target_name: str
    target_wallet: str
    positions_synced: int
    changes_detected: int

    @property
    def event_type(self) -> EventType:
        return EventType.SYNC_COMPLETED


@dataclass
class ErrorEvent(BaseEvent):
    """Emitted when an error occurs during monitoring."""

    target_name: Optional[str]
    error_type: str
    error_message: str
    recoverable: bool = True

    @property
    def event_type(self) -> EventType:
        return EventType.ERROR_OCCURRED


# Type alias for event handlers
EventHandler = Callable[[BaseEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Async event bus for publishing and subscribing to events.

    Supports multiple subscribers per event type, wildcard subscriptions,
    and async event handlers.

    Example:
        bus = EventBus()

        async def handle_trade(event: TradeDetectedEvent):
            print(f"Trade detected: {event.side} {event.size} @ {event.price}")

        bus.subscribe(EventType.TRADE_DETECTED, handle_trade)

        await bus.publish(TradeDetectedEvent(...))
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._wildcard_handlers: list[EventHandler] = []
        self._event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors": 0,
        }

    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler,
    ) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to, or None for all events
            handler: Async function to call when event is published
        """
        if event_type is None:
            self._wildcard_handlers.append(handler)
            logger.debug("wildcard_subscription_added")
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            logger.debug("subscription_added", event_type=event_type.value)

    def unsubscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler,
    ) -> None:
        """Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from, or None for wildcard
            handler: Handler function to remove
        """
        if event_type is None:
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to all subscribers.

        Events are queued and processed asynchronously.

        Args:
            event: Event to publish
        """
        await self._event_queue.put(event)
        self._stats["events_published"] += 1

        logger.debug(
            "event_published",
            event_type=event.event_type.value,
            event_id=event.event_id,
        )

    async def publish_immediate(self, event: BaseEvent) -> None:
        """Publish an event and wait for all handlers to complete.

        Use this when you need synchronous event processing.

        Args:
            event: Event to publish
        """
        await self._process_event(event)

    async def _process_event(self, event: BaseEvent) -> None:
        """Process a single event by calling all relevant handlers."""
        handlers = []

        # Get type-specific handlers
        if event.event_type in self._handlers:
            handlers.extend(self._handlers[event.event_type])

        # Add wildcard handlers
        handlers.extend(self._wildcard_handlers)

        if not handlers:
            return

        # Call all handlers concurrently
        tasks = [self._safe_call_handler(handler, event) for handler in handlers]
        await asyncio.gather(*tasks)

        self._stats["events_processed"] += 1

    async def _safe_call_handler(
        self,
        handler: EventHandler,
        event: BaseEvent,
    ) -> None:
        """Call a handler with error handling."""
        try:
            await handler(event)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(
                "event_handler_error",
                event_type=event.event_type.value,
                error=str(e),
            )

    async def _event_processor(self) -> None:
        """Background task that processes queued events."""
        while self._running:
            try:
                # Wait for event with timeout to allow graceful shutdown
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0,
                    )
                    await self._process_event(event)
                except asyncio.TimeoutError:
                    continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("event_processor_error", error=str(e))

    async def start(self) -> None:
        """Start the event processor background task."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._event_processor())
        logger.info("event_bus_started")

    async def stop(self) -> None:
        """Stop the event processor and process remaining events."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Process remaining events
        while not self._event_queue.empty():
            event = self._event_queue.get_nowait()
            await self._process_event(event)

        logger.info("event_bus_stopped", stats=self._stats)

    @property
    def stats(self) -> dict[str, int]:
        """Get event bus statistics."""
        return self._stats.copy()

    @property
    def queue_size(self) -> int:
        """Get current event queue size."""
        return self._event_queue.qsize()


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Creates the instance if it doesn't exist.

    Returns:
        The global EventBus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _event_bus
    _event_bus = None
