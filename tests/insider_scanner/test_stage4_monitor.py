"""Stage 4 Tests: Real-Time Monitoring Infrastructure.

Tests for:
- TradeEvent parsing
- WalletAccumulation tracking
- RealTimeMonitor state management
- Event processing pipeline
- Alert generation
- NewWalletMonitor

Run with: pytest tests/insider_scanner/test_stage4_monitor.py -v
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.insider_scanner.monitor import (
    RealTimeMonitor,
    NewWalletMonitor,
    TradeEvent,
    WalletAccumulation,
    MonitorState,
    MonitorStats,
)
from src.insider_scanner.scoring import InsiderScorer, ScoringResult


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# TradeEvent Tests
# =============================================================================

class TestTradeEvent:
    """Tests for TradeEvent parsing."""

    def test_from_websocket_basic(self):
        """Test parsing trade from WebSocket message."""
        data = {
            "id": "trade_123",
            "timestamp": "2024-01-15T10:30:00",
            "maker": "0xabc123",
            "market": "0xmarket123",
            "market_title": "Will X happen?",
            "asset_id": "0xtoken123",
            "side": "BUY",
            "size": "1000",
            "price": "0.45",
            "transaction_hash": "0xtx123",
        }

        trade = TradeEvent.from_websocket(data)

        assert trade.trade_id == "trade_123"
        assert trade.wallet_address == "0xabc123"
        assert trade.market_id == "0xmarket123"
        assert trade.side == "BUY"
        assert trade.size == Decimal("1000")
        assert trade.price == Decimal("0.45")
        assert trade.usd_value == Decimal("450")
        assert trade.tx_hash == "0xtx123"

    def test_from_websocket_missing_fields(self):
        """Test parsing with missing optional fields."""
        data = {
            "id": "trade_456",
            "taker": "0xdef456",  # taker instead of maker
            "condition_id": "0xmarket456",  # alternative field
            "size": "500",
            "price": "0.30",
        }

        trade = TradeEvent.from_websocket(data)

        assert trade.trade_id == "trade_456"
        assert trade.wallet_address == "0xdef456"
        assert trade.market_id == "0xmarket456"
        assert trade.side == "BUY"  # default
        assert trade.usd_value == Decimal("150")
        assert trade.tx_hash is None

    def test_from_websocket_sell(self):
        """Test parsing SELL trade."""
        data = {
            "id": "trade_789",
            "maker": "0xseller",
            "market": "0xmarket789",
            "side": "SELL",
            "size": "2000",
            "price": "0.70",
        }

        trade = TradeEvent.from_websocket(data)

        assert trade.side == "SELL"
        assert trade.usd_value == Decimal("1400")


# =============================================================================
# WalletAccumulation Tests
# =============================================================================

class TestWalletAccumulation:
    """Tests for wallet accumulation tracking."""

    def test_add_single_trade(self):
        """Test adding single trade to accumulation."""
        acc = WalletAccumulation(
            wallet_address="0xtest",
            market_id="0xmarket",
            market_title="Test Market",
        )

        trade = TradeEvent(
            trade_id="t1",
            timestamp=datetime.utcnow(),
            wallet_address="0xtest",
            market_id="0xmarket",
            market_title="Test Market",
            token_id="0xtoken",
            side="BUY",
            size=Decimal("1000"),
            price=Decimal("0.40"),
            usd_value=Decimal("10000"),
        )

        acc.add_trade(trade)

        assert acc.total_size == Decimal("1000")
        assert acc.total_usd == Decimal("10000")
        assert acc.entry_count == 1
        assert acc.avg_price == Decimal("0.40")
        assert acc.first_entry is not None

    def test_add_multiple_trades(self):
        """Test accumulating multiple trades."""
        acc = WalletAccumulation(
            wallet_address="0xtest",
            market_id="0xmarket",
            market_title="Test Market",
        )

        now = datetime.utcnow()

        trades = [
            TradeEvent(
                trade_id="t1",
                timestamp=now,
                wallet_address="0xtest",
                market_id="0xmarket",
                market_title="Test Market",
                token_id="0xtoken",
                side="BUY",
                size=Decimal("1000"),
                price=Decimal("0.40"),
                usd_value=Decimal("10000"),
            ),
            TradeEvent(
                trade_id="t2",
                timestamp=now + timedelta(hours=1),
                wallet_address="0xtest",
                market_id="0xmarket",
                market_title="Test Market",
                token_id="0xtoken",
                side="BUY",
                size=Decimal("500"),
                price=Decimal("0.50"),
                usd_value=Decimal("5000"),
            ),
        ]

        for trade in trades:
            acc.add_trade(trade)

        assert acc.total_size == Decimal("1500")
        assert acc.total_usd == Decimal("15000")
        assert acc.entry_count == 2
        assert acc.avg_price == Decimal("0.45")  # (0.40 + 0.50) / 2

    def test_sell_not_accumulated(self):
        """Test that SELL trades are not accumulated."""
        acc = WalletAccumulation(
            wallet_address="0xtest",
            market_id="0xmarket",
            market_title="Test Market",
        )

        trade = TradeEvent(
            trade_id="t1",
            timestamp=datetime.utcnow(),
            wallet_address="0xtest",
            market_id="0xmarket",
            market_title="Test Market",
            token_id="0xtoken",
            side="SELL",  # SELL should not accumulate
            size=Decimal("1000"),
            price=Decimal("0.60"),
            usd_value=Decimal("6000"),
        )

        acc.add_trade(trade)

        assert acc.total_size == Decimal("0")
        assert acc.entry_count == 0


# =============================================================================
# RealTimeMonitor Tests
# =============================================================================

class TestRealTimeMonitorState:
    """Tests for monitor state management."""

    def test_initial_state(self):
        """Test monitor starts in STOPPED state."""
        monitor = RealTimeMonitor()
        assert monitor.state == MonitorState.STOPPED
        assert not monitor.is_running

    def test_stats_initialization(self):
        """Test stats are initialized correctly."""
        monitor = RealTimeMonitor()
        assert monitor.stats.trades_processed == 0
        assert monitor.stats.alerts_generated == 0
        assert monitor.stats.started_at is None


class TestTradeProcessing:
    """Tests for trade processing pipeline."""

    @pytest.fixture
    def mock_scorer(self):
        """Create mock scorer."""
        scorer = MagicMock(spec=InsiderScorer)
        scorer.score_wallet.return_value = ScoringResult(
            score=30.0,
            confidence_low=25.0,
            confidence_high=35.0,
            priority="normal",
            dimensions={"account": 0, "trading": 10, "behavioral": 10, "contextual": 5, "cluster": 0},
            signals=[],
            signal_count=3,
            active_dimensions=3,
            downgraded=False,
            downgrade_reason=None,
        )
        return scorer

    def test_process_trade_accumulation(self, mock_scorer):
        """Test that trades are accumulated correctly."""
        async def _test():
            monitor = RealTimeMonitor(scorer=mock_scorer, min_position_usd=1000)

            trade = TradeEvent(
                trade_id="t1",
                timestamp=datetime.utcnow(),
                wallet_address="0xbuyer",
                market_id="0xmarket",
                market_title="Test Market",
                token_id="0xtoken",
                side="BUY",
                size=Decimal("500"),
                price=Decimal("0.40"),
                usd_value=Decimal("5000"),
            )

            await monitor._process_trade(trade)

            assert monitor.stats.trades_processed == 1
            accumulations = monitor.get_accumulations()
            assert len(accumulations) == 1
            assert accumulations[0].wallet_address == "0xbuyer"
            assert accumulations[0].total_usd == Decimal("5000")

        run_async(_test())

    def test_skip_small_trades(self, mock_scorer):
        """Test that small trades are skipped."""
        async def _test():
            monitor = RealTimeMonitor(scorer=mock_scorer, min_position_usd=5000)

            trade = TradeEvent(
                trade_id="t1",
                timestamp=datetime.utcnow(),
                wallet_address="0xsmall",
                market_id="0xmarket",
                market_title="Test Market",
                token_id="0xtoken",
                side="BUY",
                size=Decimal("100"),
                price=Decimal("0.40"),
                usd_value=Decimal("1000"),  # Below threshold
            )

            await monitor._process_trade(trade)

            accumulations = monitor.get_accumulations()
            assert len(accumulations) == 0

        run_async(_test())

    def test_skip_sell_trades(self, mock_scorer):
        """Test that SELL trades don't accumulate."""
        async def _test():
            monitor = RealTimeMonitor(scorer=mock_scorer, min_position_usd=1000)

            trade = TradeEvent(
                trade_id="t1",
                timestamp=datetime.utcnow(),
                wallet_address="0xseller",
                market_id="0xmarket",
                market_title="Test Market",
                token_id="0xtoken",
                side="SELL",
                size=Decimal("1000"),
                price=Decimal("0.60"),
                usd_value=Decimal("6000"),
            )

            await monitor._process_trade(trade)

            # Should process but not accumulate
            assert monitor.stats.trades_processed == 1
            accumulations = monitor.get_accumulations()
            assert len(accumulations) == 0

        run_async(_test())


class TestAlertGeneration:
    """Tests for alert generation."""

    def test_alert_on_high_score(self):
        """Test alert is generated when score exceeds threshold."""
        async def _test():
            alert_called = False
            alert_data = {}

            async def on_alert(wallet, result, accumulation):
                nonlocal alert_called, alert_data
                alert_called = True
                alert_data = {
                    "wallet": wallet,
                    "score": result.score,
                    "market": accumulation.market_id,
                }

            # Create scorer that returns high score
            mock_scorer = MagicMock(spec=InsiderScorer)
            mock_scorer.score_wallet.return_value = ScoringResult(
                score=70.0,  # Above default threshold of 55
                confidence_low=65.0,
                confidence_high=75.0,
                priority="high",
                dimensions={"account": 15, "trading": 25, "behavioral": 15, "contextual": 10, "cluster": 0},
                signals=[],
                signal_count=6,
                active_dimensions=4,
                downgraded=False,
                downgrade_reason=None,
            )

            monitor = RealTimeMonitor(
                scorer=mock_scorer,
                on_alert=on_alert,
                alert_threshold=55.0,
                min_position_usd=1000,
            )

            trade = TradeEvent(
                trade_id="t1",
                timestamp=datetime.utcnow(),
                wallet_address="0xsuspicious",
                market_id="0xmarket",
                market_title="Insider Market",
                token_id="0xtoken",
                side="BUY",
                size=Decimal("1000"),
                price=Decimal("0.10"),
                usd_value=Decimal("10000"),
            )

            await monitor._process_trade(trade)

            assert alert_called
            assert alert_data["wallet"] == "0xsuspicious"
            assert alert_data["score"] == 70.0
            assert monitor.stats.alerts_generated == 1

        run_async(_test())

    def test_no_duplicate_alerts(self):
        """Test same wallet doesn't get alerted twice."""
        async def _test():
            alert_count = 0

            async def on_alert(wallet, result, accumulation):
                nonlocal alert_count
                alert_count += 1

            mock_scorer = MagicMock(spec=InsiderScorer)
            mock_scorer.score_wallet.return_value = ScoringResult(
                score=70.0,
                confidence_low=65.0,
                confidence_high=75.0,
                priority="high",
                dimensions={},
                signals=[],
                signal_count=6,
                active_dimensions=4,
                downgraded=False,
                downgrade_reason=None,
            )

            monitor = RealTimeMonitor(
                scorer=mock_scorer,
                on_alert=on_alert,
                alert_threshold=55.0,
                min_position_usd=1000,
            )

            # Send two trades from same wallet+market
            for i in range(2):
                trade = TradeEvent(
                    trade_id=f"t{i}",
                    timestamp=datetime.utcnow(),
                    wallet_address="0xsame_wallet",
                    market_id="0xsame_market",
                    market_title="Same Market",
                    token_id="0xtoken",
                    side="BUY",
                    size=Decimal("500"),
                    price=Decimal("0.10"),
                    usd_value=Decimal("5000"),
                )
                await monitor._process_trade(trade)

            # Should only alert once
            assert alert_count == 1
            assert monitor.stats.alerts_generated == 1

        run_async(_test())

    def test_clear_alerts_allows_re_alert(self):
        """Test that clearing alerts allows re-alerting."""
        async def _test():
            alert_count = 0

            async def on_alert(wallet, result, accumulation):
                nonlocal alert_count
                alert_count += 1

            mock_scorer = MagicMock(spec=InsiderScorer)
            mock_scorer.score_wallet.return_value = ScoringResult(
                score=70.0,
                confidence_low=65.0,
                confidence_high=75.0,
                priority="high",
                dimensions={},
                signals=[],
                signal_count=6,
                active_dimensions=4,
                downgraded=False,
                downgrade_reason=None,
            )

            monitor = RealTimeMonitor(
                scorer=mock_scorer,
                on_alert=on_alert,
                alert_threshold=55.0,
                min_position_usd=1000,
            )

            trade = TradeEvent(
                trade_id="t1",
                timestamp=datetime.utcnow(),
                wallet_address="0xwallet",
                market_id="0xmarket",
                market_title="Market",
                token_id="0xtoken",
                side="BUY",
                size=Decimal("500"),
                price=Decimal("0.10"),
                usd_value=Decimal("5000"),
            )

            # First trade - should alert
            await monitor._process_trade(trade)
            assert alert_count == 1

            # Clear alerts
            monitor.clear_alerts()

            # Need new accumulation since we're same wallet+market
            monitor._accumulations.clear()

            # Second trade - should alert again
            trade.trade_id = "t2"
            await monitor._process_trade(trade)
            assert alert_count == 2

        run_async(_test())


class TestMonitorStats:
    """Tests for monitor statistics."""

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = MonitorStats(
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            trades_processed=100,
            wallets_evaluated=25,
            alerts_generated=3,
            websocket_uptime_s=3600.5,
            polling_cycles=10,
        )

        data = stats.to_dict()

        assert data["trades_processed"] == 100
        assert data["wallets_evaluated"] == 25
        assert data["alerts_generated"] == 3
        assert data["websocket_uptime_s"] == 3600.5
        assert data["polling_cycles"] == 10


# =============================================================================
# NewWalletMonitor Tests
# =============================================================================

class TestNewWalletMonitor:
    """Tests for new wallet detection."""

    def test_new_wallet_detection(self):
        """Test detecting new wallet with significant deposit."""
        async def _test():
            detected_wallets = []

            async def on_new_wallet(wallet, deposit):
                detected_wallets.append({"wallet": wallet, "deposit": deposit})

            monitor = NewWalletMonitor(
                on_new_wallet=on_new_wallet,
                min_deposit_usd=1000.0,
            )

            result = await monitor.check_wallet("0xnew_wallet", Decimal("5000"))

            assert result is True
            assert len(detected_wallets) == 1
            assert detected_wallets[0]["wallet"] == "0xnew_wallet"
            assert detected_wallets[0]["deposit"] == Decimal("5000")

        run_async(_test())

    def test_known_wallet_not_detected(self):
        """Test that known wallets are not re-detected."""
        async def _test():
            detected_count = 0

            async def on_new_wallet(wallet, deposit):
                nonlocal detected_count
                detected_count += 1

            monitor = NewWalletMonitor(
                on_new_wallet=on_new_wallet,
                min_deposit_usd=1000.0,
            )

            # First check - should detect
            await monitor.check_wallet("0xwallet", Decimal("5000"))
            # Second check - same wallet, should not detect
            await monitor.check_wallet("0xwallet", Decimal("10000"))

            assert detected_count == 1

        run_async(_test())

    def test_small_deposit_not_detected(self):
        """Test that small deposits don't trigger detection."""
        async def _test():
            detected_wallets = []

            async def on_new_wallet(wallet, deposit):
                detected_wallets.append(wallet)

            monitor = NewWalletMonitor(
                on_new_wallet=on_new_wallet,
                min_deposit_usd=5000.0,
            )

            result = await monitor.check_wallet("0xsmall_deposit", Decimal("1000"))

            assert result is False
            assert len(detected_wallets) == 0

        run_async(_test())

    def test_case_insensitive_wallet_tracking(self):
        """Test wallet addresses are tracked case-insensitively."""
        async def _test():
            detected_count = 0

            async def on_new_wallet(wallet, deposit):
                nonlocal detected_count
                detected_count += 1

            monitor = NewWalletMonitor(
                on_new_wallet=on_new_wallet,
                min_deposit_usd=1000.0,
            )

            await monitor.check_wallet("0xABC123", Decimal("5000"))
            await monitor.check_wallet("0xabc123", Decimal("5000"))  # Same wallet, different case

            assert detected_count == 1

        run_async(_test())


# =============================================================================
# Integration Tests
# =============================================================================

class TestStage4Integration:
    """Integration tests for Stage 4 completion."""

    def test_module_exports(self):
        """Test Stage 4 classes are exported from module."""
        from src.insider_scanner import (
            RealTimeMonitor,
            NewWalletMonitor,
            TradeEvent,
            WalletAccumulation,
            MonitorState,
            MonitorStats,
        )

        # Verify classes exist
        assert RealTimeMonitor is not None
        assert NewWalletMonitor is not None
        assert TradeEvent is not None
        assert WalletAccumulation is not None
        assert MonitorState is not None
        assert MonitorStats is not None

    def test_complete_event_pipeline(self):
        """Test complete event processing pipeline."""
        async def _test():
            alerts = []

            async def capture_alert(wallet, result, accumulation):
                alerts.append({
                    "wallet": wallet,
                    "score": result.score,
                    "priority": result.priority,
                    "market": accumulation.market_id,
                    "position_usd": float(accumulation.total_usd),
                })

            # Create real scorer
            scorer = InsiderScorer()

            monitor = RealTimeMonitor(
                scorer=scorer,
                on_alert=capture_alert,
                alert_threshold=40.0,  # Lower for testing
                min_position_usd=1000,
            )

            # Simulate suspicious trading pattern
            # Fresh wallet making large position at low odds
            now = datetime.utcnow()
            trades = [
                TradeEvent(
                    trade_id=f"t{i}",
                    timestamp=now + timedelta(hours=i),
                    wallet_address="0xsuspicious_trader",
                    market_id="0xhigh_risk_market",
                    market_title="Will Military Operation X Happen?",
                    token_id="0xyes_token",
                    side="BUY",
                    size=Decimal("10000"),
                    price=Decimal("0.10"),  # Low odds
                    usd_value=Decimal("10000"),
                )
                for i in range(3)  # 3 entries = split entry pattern
            ]

            for trade in trades:
                await monitor._process_trade(trade)

            # Verify accumulation
            accumulations = monitor.get_accumulations()
            assert len(accumulations) == 1
            assert accumulations[0].total_usd == Decimal("30000")
            assert accumulations[0].entry_count == 3

            # Verify stats
            assert monitor.stats.trades_processed == 3
            assert monitor.stats.wallets_evaluated >= 1

        run_async(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
