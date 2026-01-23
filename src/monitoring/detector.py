"""Position change detection from trade activity.

This module processes detected trades and determines the resulting
position changes, emitting appropriate events.
"""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ..config.models import TargetAccount
from ..utils.logging import get_logger
from .events import (
    EventBus,
    BaseEvent,
    TradeDetectedEvent,
    PositionOpenedEvent,
    PositionIncreasedEvent,
    PositionDecreasedEvent,
    PositionClosedEvent,
)
from .state import (
    PositionStateManager,
    TrackedPosition,
    PositionChange,
    ChangeType,
)

logger = get_logger(__name__)


@dataclass
class ChangeDetectorStats:
    """Statistics for the change detector."""

    trades_processed: int = 0
    positions_opened: int = 0
    positions_increased: int = 0
    positions_decreased: int = 0
    positions_closed: int = 0


class PositionChangeDetector:
    """Detects and classifies position changes from trade events.

    Listens to TradeDetectedEvent and emits appropriate position
    change events (opened, increased, decreased, closed).

    Example:
        detector = PositionChangeDetector(
            event_bus=event_bus,
            state_manager=state_manager,
            targets=config.enabled_targets,
        )
        await detector.start()
    """

    def __init__(
        self,
        event_bus: EventBus,
        state_manager: PositionStateManager,
        targets: list[TargetAccount],
    ):
        """Initialize the change detector.

        Args:
            event_bus: Event bus for publishing and subscribing
            state_manager: Position state manager
            targets: List of target accounts being monitored
        """
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.targets = {t.name: t for t in targets}
        self.stats = ChangeDetectorStats()

    async def start(self) -> None:
        """Start listening for trade events."""
        self.event_bus.subscribe(
            TradeDetectedEvent(
                target_name="",
                target_wallet="",
                condition_id="",
                token_id="",
                outcome="",
                side="",
                size=Decimal("0"),
                price=Decimal("0"),
                usd_value=Decimal("0"),
                trade_timestamp=None,
            ).event_type,
            self._handle_trade,
        )
        logger.info("change_detector_started")

    async def stop(self) -> None:
        """Stop the change detector."""
        # Note: We don't unsubscribe as the event bus may be shared
        logger.info("change_detector_stopped", stats=self.stats.__dict__)

    async def _handle_trade(self, event: BaseEvent) -> None:
        """Handle a trade detected event.

        Args:
            event: The trade event to process
        """
        if not isinstance(event, TradeDetectedEvent):
            return

        self.stats.trades_processed += 1

        # Get target configuration
        target = self.targets.get(event.target_name)
        if not target:
            logger.warning(
                "unknown_target_in_trade",
                target_name=event.target_name,
            )
            return

        # Get current position state
        current_position = await self.state_manager.get_position(
            event.target_name,
            event.token_id,
        )

        # Calculate new position size based on trade
        if current_position:
            old_size = current_position.size
        else:
            old_size = Decimal("0")

        # Determine new size based on trade side
        if event.side == "BUY":
            new_size = old_size + event.size
        elif event.side == "SELL":
            new_size = old_size - event.size
            if new_size < 0:
                new_size = Decimal("0")  # Can't go negative
        else:
            logger.warning("unknown_trade_side", side=event.side)
            new_size = old_size

        # Update state and get change
        change = await self.state_manager.update_position(
            target_name=event.target_name,
            target_wallet=event.target_wallet,
            token_id=event.token_id,
            condition_id=event.condition_id,
            outcome=event.outcome,
            size=new_size,
            average_price=event.price,  # Simplified: use trade price
            current_value=new_size * event.price,
        )

        # Emit appropriate position change event
        await self._emit_position_event(event, change, old_size, new_size)

    async def _emit_position_event(
        self,
        trade: TradeDetectedEvent,
        change: PositionChange,
        old_size: Decimal,
        new_size: Decimal,
    ) -> None:
        """Emit the appropriate position change event.

        Args:
            trade: The original trade event
            change: The detected change
            old_size: Previous position size
            new_size: New position size
        """
        size_delta = abs(new_size - old_size)
        usd_value = size_delta * trade.price

        if change.change_type == ChangeType.OPENED:
            self.stats.positions_opened += 1
            await self.event_bus.publish(PositionOpenedEvent(
                target_name=trade.target_name,
                target_wallet=trade.target_wallet,
                condition_id=trade.condition_id,
                token_id=trade.token_id,
                outcome=trade.outcome,
                size=new_size,
                entry_price=trade.price,
                usd_value=usd_value,
            ))

        elif change.change_type == ChangeType.INCREASED:
            self.stats.positions_increased += 1
            await self.event_bus.publish(PositionIncreasedEvent(
                target_name=trade.target_name,
                target_wallet=trade.target_wallet,
                condition_id=trade.condition_id,
                token_id=trade.token_id,
                outcome=trade.outcome,
                previous_size=old_size,
                new_size=new_size,
                size_delta=size_delta,
                price=trade.price,
                usd_value=usd_value,
            ))

        elif change.change_type == ChangeType.DECREASED:
            self.stats.positions_decreased += 1
            await self.event_bus.publish(PositionDecreasedEvent(
                target_name=trade.target_name,
                target_wallet=trade.target_wallet,
                condition_id=trade.condition_id,
                token_id=trade.token_id,
                outcome=trade.outcome,
                previous_size=old_size,
                new_size=new_size,
                size_delta=size_delta,
                price=trade.price,
                usd_value=usd_value,
            ))

        elif change.change_type == ChangeType.CLOSED:
            self.stats.positions_closed += 1
            await self.event_bus.publish(PositionClosedEvent(
                target_name=trade.target_name,
                target_wallet=trade.target_wallet,
                condition_id=trade.condition_id,
                token_id=trade.token_id,
                outcome=trade.outcome,
                closed_size=old_size,
                exit_price=trade.price,
                usd_value=usd_value,
            ))


class TradeAggregator:
    """Aggregates rapid trades into single position changes.

    When a target makes multiple trades in quick succession (e.g., filling
    a large order in parts), this aggregates them into a single event.
    """

    def __init__(
        self,
        aggregation_window_ms: int = 500,
    ):
        """Initialize the aggregator.

        Args:
            aggregation_window_ms: Time window to aggregate trades
        """
        self.aggregation_window_ms = aggregation_window_ms
        self._pending_trades: dict[str, list[TradeDetectedEvent]] = {}
        self._timers: dict[str, asyncio.Task] = {}

    def _get_key(self, trade: TradeDetectedEvent) -> str:
        """Get aggregation key for a trade."""
        return f"{trade.target_name}:{trade.token_id}:{trade.side}"

    async def add_trade(
        self,
        trade: TradeDetectedEvent,
        callback,
    ) -> None:
        """Add a trade for potential aggregation.

        Args:
            trade: The trade to add
            callback: Async function to call with aggregated trade
        """
        key = self._get_key(trade)

        if key not in self._pending_trades:
            self._pending_trades[key] = []

        self._pending_trades[key].append(trade)

        # Cancel existing timer
        if key in self._timers:
            self._timers[key].cancel()

        # Start new timer
        self._timers[key] = asyncio.create_task(
            self._flush_after_delay(key, callback)
        )

    async def _flush_after_delay(self, key: str, callback) -> None:
        """Flush trades after delay.

        Args:
            key: Aggregation key
            callback: Callback to invoke
        """
        await asyncio.sleep(self.aggregation_window_ms / 1000)
        await self._flush(key, callback)

    async def _flush(self, key: str, callback) -> None:
        """Flush pending trades for a key.

        Args:
            key: Aggregation key
            callback: Callback to invoke
        """
        if key not in self._pending_trades:
            return

        trades = self._pending_trades.pop(key)
        if key in self._timers:
            del self._timers[key]

        if not trades:
            return

        # Aggregate trades
        aggregated = self._aggregate_trades(trades)
        await callback(aggregated)

    def _aggregate_trades(
        self,
        trades: list[TradeDetectedEvent],
    ) -> TradeDetectedEvent:
        """Aggregate multiple trades into one.

        Args:
            trades: List of trades to aggregate

        Returns:
            Single aggregated trade event
        """
        if len(trades) == 1:
            return trades[0]

        # Use first trade as base
        base = trades[0]

        # Sum sizes and calculate weighted average price
        total_size = Decimal("0")
        weighted_price = Decimal("0")
        total_usd = Decimal("0")
        latest_timestamp = base.trade_timestamp

        for trade in trades:
            total_size += trade.size
            weighted_price += trade.price * trade.size
            total_usd += trade.usd_value
            if trade.trade_timestamp > latest_timestamp:
                latest_timestamp = trade.trade_timestamp

        avg_price = weighted_price / total_size if total_size > 0 else base.price

        return TradeDetectedEvent(
            target_name=base.target_name,
            target_wallet=base.target_wallet,
            condition_id=base.condition_id,
            token_id=base.token_id,
            outcome=base.outcome,
            side=base.side,
            size=total_size,
            price=avg_price,
            usd_value=total_usd,
            trade_timestamp=latest_timestamp,
            activity_id=f"aggregated_{len(trades)}_trades",
        )

    async def flush_all(self, callback) -> None:
        """Flush all pending trades immediately.

        Args:
            callback: Callback to invoke for each aggregation
        """
        keys = list(self._pending_trades.keys())
        for key in keys:
            # Cancel timer
            if key in self._timers:
                self._timers[key].cancel()
            await self._flush(key, callback)
