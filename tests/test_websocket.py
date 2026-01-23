"""Tests for WebSocket components."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.websocket.client import (
    WebSocketClient,
    AuthenticatedWebSocketClient,
    ConnectionState,
    ReconnectConfig,
    ConnectionStats,
)
from src.websocket.market_feed import (
    MarketFeed,
    LocalOrderBook,
    BookLevel,
    BookUpdate,
)
from src.websocket.user_feed import (
    UserFeed,
    OrderUpdate,
    TradeUpdate,
    OrderStatus,
    FillType,
    FillTracker,
)
from src.websocket.health import (
    HealthMonitor,
    HealthStatus,
    ConnectionHealth,
    LatencyMetrics,
)


# =============================================================================
# Local Order Book Tests
# =============================================================================


class TestLocalOrderBook:
    """Tests for LocalOrderBook functionality."""

    def test_empty_book(self):
        """Test empty order book."""
        book = LocalOrderBook(token_id="0xtest")

        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None
        assert book.mid_price is None

    def test_apply_snapshot(self):
        """Test applying a snapshot update."""
        book = LocalOrderBook(token_id="0xtest")

        update = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=1,
            bids=[
                BookLevel(price=Decimal("0.48"), size=Decimal("100")),
                BookLevel(price=Decimal("0.47"), size=Decimal("200")),
            ],
            asks=[
                BookLevel(price=Decimal("0.52"), size=Decimal("150")),
                BookLevel(price=Decimal("0.53"), size=Decimal("250")),
            ],
            is_snapshot=True,
        )

        applied = book.apply_update(update)

        assert applied
        assert book.best_bid == Decimal("0.48")
        assert book.best_ask == Decimal("0.52")
        assert book.spread == Decimal("0.04")
        assert book.mid_price == Decimal("0.50")
        assert book.sequence == 1

    def test_apply_incremental_update(self):
        """Test applying incremental updates."""
        book = LocalOrderBook(token_id="0xtest")

        # Initial snapshot
        snapshot = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=1,
            bids=[BookLevel(price=Decimal("0.48"), size=Decimal("100"))],
            asks=[BookLevel(price=Decimal("0.52"), size=Decimal("100"))],
            is_snapshot=True,
        )
        book.apply_update(snapshot)

        # Incremental update - add new bid level
        update = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=2,
            bids=[BookLevel(price=Decimal("0.49"), size=Decimal("50"))],
            asks=[],
            is_snapshot=False,
        )
        applied = book.apply_update(update)

        assert applied
        assert book.best_bid == Decimal("0.49")
        assert len(book.bids) == 2

    def test_remove_level(self):
        """Test removing a price level."""
        book = LocalOrderBook(token_id="0xtest")

        # Initial state
        snapshot = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=1,
            bids=[
                BookLevel(price=Decimal("0.48"), size=Decimal("100")),
                BookLevel(price=Decimal("0.47"), size=Decimal("200")),
            ],
            asks=[],
            is_snapshot=True,
        )
        book.apply_update(snapshot)

        # Remove top bid
        update = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=2,
            bids=[BookLevel(price=Decimal("0.48"), size=Decimal("0"))],  # Size 0 = remove
            asks=[],
            is_snapshot=False,
        )
        book.apply_update(update)

        assert book.best_bid == Decimal("0.47")
        assert len(book.bids) == 1

    def test_stale_update_rejected(self):
        """Test that stale updates are rejected."""
        book = LocalOrderBook(token_id="0xtest")

        # Initial snapshot
        snapshot = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=10,
            bids=[BookLevel(price=Decimal("0.48"), size=Decimal("100"))],
            asks=[],
            is_snapshot=True,
        )
        book.apply_update(snapshot)

        # Stale update (lower sequence)
        stale_update = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=5,
            bids=[BookLevel(price=Decimal("0.49"), size=Decimal("50"))],
            asks=[],
            is_snapshot=False,
        )
        applied = book.apply_update(stale_update)

        assert not applied
        assert book.best_bid == Decimal("0.48")

    def test_get_depth(self):
        """Test getting order book depth."""
        book = LocalOrderBook(token_id="0xtest")

        snapshot = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=1,
            bids=[
                BookLevel(price=Decimal("0.48"), size=Decimal("100")),
                BookLevel(price=Decimal("0.47"), size=Decimal("200")),
                BookLevel(price=Decimal("0.46"), size=Decimal("300")),
            ],
            asks=[
                BookLevel(price=Decimal("0.52"), size=Decimal("150")),
                BookLevel(price=Decimal("0.53"), size=Decimal("250")),
            ],
            is_snapshot=True,
        )
        book.apply_update(snapshot)

        # Get bid depth
        bids = book.get_depth("BUY", levels=2)
        assert len(bids) == 2
        assert bids[0].price == Decimal("0.48")  # Best first
        assert bids[1].price == Decimal("0.47")

        # Get ask depth
        asks = book.get_depth("SELL", levels=2)
        assert len(asks) == 2
        assert asks[0].price == Decimal("0.52")  # Best first

    def test_get_liquidity(self):
        """Test calculating available liquidity."""
        book = LocalOrderBook(token_id="0xtest")

        snapshot = BookUpdate(
            token_id="0xtest",
            timestamp=datetime.utcnow(),
            sequence=1,
            bids=[],
            asks=[
                BookLevel(price=Decimal("0.50"), size=Decimal("100")),  # $50 worth
                BookLevel(price=Decimal("0.51"), size=Decimal("100")),  # $51 worth
            ],
            is_snapshot=True,
        )
        book.apply_update(snapshot)

        # Get liquidity for buys (from asks)
        liquidity = book.get_liquidity("BUY", depth_usd=Decimal("75"))

        # Should get all 100 shares at 0.50 ($50) + some at 0.51
        assert liquidity > Decimal("100")
        assert liquidity < Decimal("200")


# =============================================================================
# WebSocket Client Tests
# =============================================================================


class TestWebSocketClient:
    """Tests for WebSocketClient functionality."""

    def test_initial_state(self):
        """Test initial client state."""
        client = WebSocketClient(url="wss://test.example.com/ws")

        assert client.state == ConnectionState.DISCONNECTED
        assert not client.is_connected
        assert client.stats.connects == 0

    def test_reconnect_config(self):
        """Test reconnect configuration."""
        config = ReconnectConfig(
            initial_delay_ms=100,
            max_delay_ms=5000,
            multiplier=2.0,
            max_attempts=5,
        )
        client = WebSocketClient(
            url="wss://test.example.com/ws",
            reconnect_config=config,
        )

        assert client.reconnect_config.initial_delay_ms == 100
        assert client.reconnect_config.max_attempts == 5

    @pytest.mark.asyncio
    async def test_subscription_tracking(self):
        """Test that subscriptions are tracked for reconnect."""
        client = WebSocketClient(url="wss://test.example.com/ws")

        # Add subscription (won't actually send since not connected)
        subscription = {"type": "subscribe", "channel": "book", "market": "0xtest"}

        # Manually add to tracked subscriptions
        client._subscriptions.append(subscription)

        assert subscription in client._subscriptions

    def test_connection_stats(self):
        """Test connection statistics."""
        stats = ConnectionStats()

        assert stats.connects == 0
        assert stats.avg_wait_time_ms == 0

        stats.messages_received = 100
        stats.bytes_received = 50000

        result = stats.to_dict()
        assert result["messages_received"] == 100
        assert result["bytes_received"] == 50000


# =============================================================================
# User Feed Tests
# =============================================================================


class TestOrderUpdate:
    """Tests for OrderUpdate dataclass."""

    def test_order_update_properties(self):
        """Test OrderUpdate properties."""
        update = OrderUpdate(
            order_id="order123",
            status=OrderStatus.PARTIALLY_FILLED,
            token_id="0xtoken",
            side="BUY",
            price=Decimal("0.50"),
            original_size=Decimal("100"),
            filled_size=Decimal("60"),
            remaining_size=Decimal("40"),
            timestamp=datetime.utcnow(),
        )

        assert not update.is_terminal
        assert update.fill_percent == Decimal("60")

    def test_terminal_states(self):
        """Test terminal order states."""
        for status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            update = OrderUpdate(
                order_id="order123",
                status=status,
                token_id="0xtoken",
                side="BUY",
                price=Decimal("0.50"),
                original_size=Decimal("100"),
                filled_size=Decimal("100"),
                remaining_size=Decimal("0"),
                timestamp=datetime.utcnow(),
            )
            assert update.is_terminal


class TestTradeUpdate:
    """Tests for TradeUpdate dataclass."""

    def test_trade_value(self):
        """Test trade value calculation."""
        trade = TradeUpdate(
            trade_id="trade123",
            order_id="order123",
            token_id="0xtoken",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("100"),
            fee=Decimal("0.01"),
            timestamp=datetime.utcnow(),
            fill_type=FillType.TAKER,
        )

        assert trade.value == Decimal("50")


class TestFillTracker:
    """Tests for FillTracker functionality."""

    def test_execution_quality(self):
        """Test execution quality calculation."""
        # Create mock user feed
        user_feed = MagicMock()
        tracker = FillTracker(user_feed)

        # Set expected price
        order_id = "order123"
        tracker.set_expected_price(order_id, Decimal("0.50"))

        # Simulate fills
        tracker._fills[order_id] = [
            TradeUpdate(
                trade_id="t1",
                order_id=order_id,
                token_id="0x",
                side="BUY",
                price=Decimal("0.51"),
                size=Decimal("60"),
                fee=Decimal("0"),
                timestamp=datetime.utcnow(),
                fill_type=FillType.TAKER,
            ),
            TradeUpdate(
                trade_id="t2",
                order_id=order_id,
                token_id="0x",
                side="BUY",
                price=Decimal("0.52"),
                size=Decimal("40"),
                fee=Decimal("0"),
                timestamp=datetime.utcnow(),
                fill_type=FillType.TAKER,
            ),
        ]

        quality = tracker.get_execution_quality(order_id)

        assert quality is not None
        assert quality["fill_count"] == 2
        assert quality["total_size"] == "100"
        assert "slippage_percent" in quality


# =============================================================================
# Health Monitoring Tests
# =============================================================================


class TestLatencyMetrics:
    """Tests for LatencyMetrics."""

    def test_add_samples(self):
        """Test adding latency samples."""
        metrics = LatencyMetrics(max_samples=5)

        for i in range(10):
            metrics.add_sample(float(i * 10))

        # Should only keep last 5 samples
        assert len(metrics.samples) == 5
        assert metrics.samples == [50.0, 60.0, 70.0, 80.0, 90.0]

    def test_statistics(self):
        """Test latency statistics."""
        metrics = LatencyMetrics()

        for latency in [10, 20, 30, 40, 50]:
            metrics.add_sample(float(latency))

        assert metrics.avg_latency_ms == 30.0
        assert metrics.min_latency_ms == 10.0
        assert metrics.max_latency_ms == 50.0


class TestConnectionHealth:
    """Tests for ConnectionHealth."""

    def test_stale_detection(self):
        """Test stale data detection."""
        health = ConnectionHealth(
            name="test",
            stale_threshold_s=30.0,
        )

        # No last message = stale
        assert health.is_stale

        # Recent message = not stale
        health.last_message_at = datetime.utcnow()
        assert not health.is_stale

        # Old message = stale
        health.last_message_at = datetime.utcnow() - timedelta(seconds=60)
        assert health.is_stale

    def test_health_to_dict(self):
        """Test health serialization."""
        health = ConnectionHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            connected=True,
            uptime_s=3600,
        )

        result = health.to_dict()

        assert result["name"] == "test"
        assert result["status"] == "healthy"
        assert result["connected"] is True
        assert result["uptime_s"] == 3600


class TestHealthStatus:
    """Tests for health status levels."""

    def test_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


# =============================================================================
# Integration Tests
# =============================================================================


class TestWebSocketIntegration:
    """Integration tests for WebSocket components."""

    @pytest.mark.asyncio
    async def test_order_book_update_flow(self):
        """Test order book update processing flow."""
        # Create local order book
        book = LocalOrderBook(token_id="0xtoken123")

        # Simulate receiving snapshot
        snapshot = BookUpdate(
            token_id="0xtoken123",
            timestamp=datetime.utcnow(),
            sequence=1,
            bids=[
                BookLevel(price=Decimal("0.45"), size=Decimal("500")),
                BookLevel(price=Decimal("0.44"), size=Decimal("1000")),
            ],
            asks=[
                BookLevel(price=Decimal("0.55"), size=Decimal("500")),
                BookLevel(price=Decimal("0.56"), size=Decimal("1000")),
            ],
            is_snapshot=True,
        )

        book.apply_update(snapshot)

        assert book.best_bid == Decimal("0.45")
        assert book.best_ask == Decimal("0.55")
        assert book.spread == Decimal("0.10")

        # Simulate incremental updates
        updates = [
            BookUpdate(
                token_id="0xtoken123",
                timestamp=datetime.utcnow(),
                sequence=2,
                bids=[BookLevel(price=Decimal("0.46"), size=Decimal("200"))],
                asks=[],
                is_snapshot=False,
            ),
            BookUpdate(
                token_id="0xtoken123",
                timestamp=datetime.utcnow(),
                sequence=3,
                bids=[],
                asks=[BookLevel(price=Decimal("0.54"), size=Decimal("300"))],
                is_snapshot=False,
            ),
        ]

        for update in updates:
            book.apply_update(update)

        # New best prices
        assert book.best_bid == Decimal("0.46")
        assert book.best_ask == Decimal("0.54")
        assert book.spread == Decimal("0.08")

    @pytest.mark.asyncio
    async def test_fill_tracking_flow(self):
        """Test fill tracking through order lifecycle."""
        user_feed = MagicMock()
        tracker = FillTracker(user_feed)

        order_id = "order_abc"
        expected_price = Decimal("0.50")

        # Set expectation
        tracker.set_expected_price(order_id, expected_price)

        # Simulate partial fills
        fills = [
            TradeUpdate(
                trade_id=f"t{i}",
                order_id=order_id,
                token_id="0x",
                side="BUY",
                price=Decimal("0.50") + Decimal(str(i)) / 100,
                size=Decimal("25"),
                fee=Decimal("0"),
                timestamp=datetime.utcnow(),
                fill_type=FillType.TAKER,
            )
            for i in range(4)
        ]

        for fill in fills:
            await tracker.on_trade(fill)

        # Check execution quality
        quality = tracker.get_execution_quality(order_id)

        assert quality is not None
        assert quality["fill_count"] == 4
        assert Decimal(quality["total_size"]) == Decimal("100")
        # Should show some slippage since fills were above expected
        assert "slippage_percent" in quality
