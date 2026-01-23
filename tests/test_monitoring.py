"""Tests for position monitoring components."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.monitoring.events import (
    EventBus,
    EventType,
    TradeDetectedEvent,
    PositionOpenedEvent,
    PositionClosedEvent,
    reset_event_bus,
)
from src.monitoring.state import (
    PositionStateManager,
    TrackedPosition,
    ChangeType,
    PositionStatus,
    reset_state_manager,
)
from src.monitoring.poller import ActivityPoller, PollerConfig
from src.monitoring.detector import PositionChangeDetector, TradeAggregator
from src.config.models import TargetAccount


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_bus():
    """Create a fresh event bus for each test."""
    reset_event_bus()
    bus = EventBus()
    yield bus
    # Cleanup
    asyncio.get_event_loop().run_until_complete(bus.stop())


@pytest.fixture
def state_manager():
    """Create a fresh state manager for each test."""
    reset_state_manager()
    return PositionStateManager()


@pytest.fixture
def target_account():
    """Create a test target account."""
    return TargetAccount(
        name="test_whale",
        wallet="0x1234567890123456789012345678901234567890",
        position_ratio=Decimal("0.01"),
        max_position_usd=Decimal("500"),
        slippage_tolerance=Decimal("0.05"),
    )


# =============================================================================
# Event Bus Tests
# =============================================================================


class TestEventBus:
    """Tests for EventBus functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.TRADE_DETECTED, handler)

        event = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.5"),
            usd_value=Decimal("50"),
            trade_timestamp=datetime.utcnow(),
        )

        await event_bus.publish_immediate(event)

        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.TRADE_DETECTED
        assert received_events[0].target_name == "test"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test subscribing to all events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe to all events
        event_bus.subscribe(None, handler)

        # Publish different event types
        trade_event = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.5"),
            usd_value=Decimal("50"),
            trade_timestamp=datetime.utcnow(),
        )

        position_event = PositionOpenedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            size=Decimal("100"),
            entry_price=Decimal("0.5"),
            usd_value=Decimal("50"),
        )

        await event_bus.publish_immediate(trade_event)
        await event_bus.publish_immediate(position_event)

        assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event type."""
        handler1_calls = []
        handler2_calls = []

        async def handler1(event):
            handler1_calls.append(event)

        async def handler2(event):
            handler2_calls.append(event)

        event_bus.subscribe(EventType.TRADE_DETECTED, handler1)
        event_bus.subscribe(EventType.TRADE_DETECTED, handler2)

        event = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.5"),
            usd_value=Decimal("50"),
            trade_timestamp=datetime.utcnow(),
        )

        await event_bus.publish_immediate(event)

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self, event_bus):
        """Test that one handler error doesn't affect others."""
        good_handler_calls = []

        async def bad_handler(event):
            raise Exception("Handler error")

        async def good_handler(event):
            good_handler_calls.append(event)

        event_bus.subscribe(EventType.TRADE_DETECTED, bad_handler)
        event_bus.subscribe(EventType.TRADE_DETECTED, good_handler)

        event = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.5"),
            usd_value=Decimal("50"),
            trade_timestamp=datetime.utcnow(),
        )

        # Should not raise, and good handler should still be called
        await event_bus.publish_immediate(event)
        assert len(good_handler_calls) == 1


# =============================================================================
# State Manager Tests
# =============================================================================


class TestPositionStateManager:
    """Tests for PositionStateManager functionality."""

    @pytest.mark.asyncio
    async def test_update_new_position(self, state_manager):
        """Test updating a new position."""
        change = await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
            current_value=Decimal("50"),
        )

        assert change.change_type == ChangeType.OPENED
        assert change.new_size == Decimal("100")
        assert change.old_size == Decimal("0")

    @pytest.mark.asyncio
    async def test_update_increase_position(self, state_manager):
        """Test increasing an existing position."""
        # Create initial position
        await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
        )

        # Increase position
        change = await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("150"),
            average_price=Decimal("0.55"),
        )

        assert change.change_type == ChangeType.INCREASED
        assert change.old_size == Decimal("100")
        assert change.new_size == Decimal("150")
        assert change.size_delta == Decimal("50")

    @pytest.mark.asyncio
    async def test_update_decrease_position(self, state_manager):
        """Test decreasing an existing position."""
        # Create initial position
        await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
        )

        # Decrease position
        change = await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("50"),
            average_price=Decimal("0.5"),
        )

        assert change.change_type == ChangeType.DECREASED
        assert change.old_size == Decimal("100")
        assert change.new_size == Decimal("50")

    @pytest.mark.asyncio
    async def test_close_position(self, state_manager):
        """Test closing a position."""
        # Create initial position
        await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
        )

        # Close position
        change = await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("0"),
            average_price=Decimal("0.6"),
        )

        assert change.change_type == ChangeType.CLOSED
        assert change.old_size == Decimal("100")
        assert change.new_size == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_all_positions(self, state_manager):
        """Test getting all positions for a target."""
        # Create multiple positions
        await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
        )

        await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token2",
            condition_id="cond2",
            outcome="No",
            size=Decimal("200"),
            average_price=Decimal("0.3"),
        )

        positions = await state_manager.get_all_positions("whale1")
        assert len(positions) == 2

    @pytest.mark.asyncio
    async def test_sync_positions_detects_closed(self, state_manager):
        """Test that sync detects closed positions."""
        # Create initial position
        await state_manager.update_position(
            target_name="whale1",
            target_wallet="0x123",
            token_id="token1",
            condition_id="cond1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
        )

        # Sync with empty positions (simulating closed position)
        changes = await state_manager.sync_positions(
            target_name="whale1",
            target_wallet="0x123",
            positions=[],  # No positions from API = all closed
        )

        # Should detect the closed position
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.CLOSED

    @pytest.mark.asyncio
    async def test_activity_timestamp_tracking(self, state_manager):
        """Test last activity timestamp tracking."""
        await state_manager.set_last_activity_timestamp("whale1", 1704067200)

        ts = await state_manager.get_last_activity_timestamp("whale1")
        assert ts == 1704067200

        # Non-existent target
        ts = await state_manager.get_last_activity_timestamp("unknown")
        assert ts is None


# =============================================================================
# Trade Aggregator Tests
# =============================================================================


class TestTradeAggregator:
    """Tests for TradeAggregator functionality."""

    @pytest.mark.asyncio
    async def test_single_trade_passthrough(self):
        """Test that single trades pass through unchanged."""
        aggregator = TradeAggregator(aggregation_window_ms=100)
        received = []

        async def callback(trade):
            received.append(trade)

        trade = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.5"),
            usd_value=Decimal("50"),
            trade_timestamp=datetime.utcnow(),
        )

        await aggregator.add_trade(trade, callback)
        await asyncio.sleep(0.15)  # Wait for aggregation window

        assert len(received) == 1
        assert received[0].size == Decimal("100")

    @pytest.mark.asyncio
    async def test_multiple_trades_aggregation(self):
        """Test that rapid trades are aggregated."""
        aggregator = TradeAggregator(aggregation_window_ms=200)
        received = []

        async def callback(trade):
            received.append(trade)

        base_time = datetime.utcnow()

        # Add multiple trades quickly
        for i in range(3):
            trade = TradeDetectedEvent(
                target_name="test",
                target_wallet="0x123",
                condition_id="cond1",
                token_id="token1",
                outcome="Yes",
                side="BUY",
                size=Decimal("100"),
                price=Decimal("0.5"),
                usd_value=Decimal("50"),
                trade_timestamp=base_time,
            )
            await aggregator.add_trade(trade, callback)

        await asyncio.sleep(0.3)  # Wait for aggregation window

        # Should receive one aggregated trade
        assert len(received) == 1
        assert received[0].size == Decimal("300")  # 100 * 3

    @pytest.mark.asyncio
    async def test_different_tokens_not_aggregated(self):
        """Test that trades for different tokens are not aggregated."""
        aggregator = TradeAggregator(aggregation_window_ms=100)
        received = []

        async def callback(trade):
            received.append(trade)

        base_time = datetime.utcnow()

        # Add trades for different tokens
        trade1 = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.5"),
            usd_value=Decimal("50"),
            trade_timestamp=base_time,
        )

        trade2 = TradeDetectedEvent(
            target_name="test",
            target_wallet="0x123",
            condition_id="cond2",
            token_id="token2",
            outcome="No",
            side="BUY",
            size=Decimal("200"),
            price=Decimal("0.3"),
            usd_value=Decimal("60"),
            trade_timestamp=base_time,
        )

        await aggregator.add_trade(trade1, callback)
        await aggregator.add_trade(trade2, callback)
        await asyncio.sleep(0.15)

        # Should receive two separate trades
        assert len(received) == 2


# =============================================================================
# Tracked Position Tests
# =============================================================================


class TestTrackedPosition:
    """Tests for TrackedPosition model."""

    def test_position_key(self):
        """Test position key generation."""
        position = TrackedPosition(
            target_name="whale1",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
            current_value=Decimal("50"),
        )

        assert position.position_key == "0x123:token1"

    def test_usd_value_calculation(self):
        """Test USD value calculation."""
        position = TrackedPosition(
            target_name="whale1",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
            current_value=Decimal("50"),
        )

        assert position.usd_value == Decimal("50")

    def test_position_update(self):
        """Test position update returns correct change."""
        position = TrackedPosition(
            target_name="whale1",
            target_wallet="0x123",
            condition_id="cond1",
            token_id="token1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.5"),
            current_value=Decimal("50"),
        )

        change = position.update(Decimal("150"))

        assert change.change_type == ChangeType.INCREASED
        assert change.old_size == Decimal("100")
        assert change.new_size == Decimal("150")
        assert position.size == Decimal("150")
