"""Tests for account management components."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.accounts.balance import (
    BalanceTracker,
    MultiAccountBalanceTracker,
    BalanceChange,
    BalanceChangeType,
)
from src.accounts.positions import (
    PositionManager,
    CopyPosition,
    PositionLot,
    PositionStatus,
)
from src.accounts.pnl import (
    PnLCalculator,
    TradeResult,
    PerformanceMetrics,
    PnLPeriod,
)
from src.accounts.reconciliation import (
    DriftDetector,
    DriftResult,
    DriftStatus,
)


# =============================================================================
# Balance Tracker Tests
# =============================================================================


class TestBalanceTracker:
    """Tests for BalanceTracker functionality."""

    def test_initial_balance(self):
        """Test initial balance state."""
        tracker = BalanceTracker(initial_balance=Decimal("1000"))

        assert tracker.total_balance == Decimal("1000")
        assert tracker.reserved_balance == Decimal("0")
        assert tracker.available_balance == Decimal("1000")

    def test_reserve_and_release(self):
        """Test balance reservation and release."""
        tracker = BalanceTracker(initial_balance=Decimal("1000"))

        # Reserve
        success = tracker.reserve(Decimal("300"), order_id="order1")

        assert success
        assert tracker.total_balance == Decimal("1000")
        assert tracker.reserved_balance == Decimal("300")
        assert tracker.available_balance == Decimal("700")

        # Release
        released = tracker.release("order1")

        assert released == Decimal("300")
        assert tracker.reserved_balance == Decimal("0")
        assert tracker.available_balance == Decimal("1000")

    def test_reserve_insufficient_balance(self):
        """Test reservation fails with insufficient balance."""
        tracker = BalanceTracker(initial_balance=Decimal("100"))

        success = tracker.reserve(Decimal("500"), order_id="order1")

        assert not success
        assert tracker.reserved_balance == Decimal("0")

    def test_record_buy(self):
        """Test recording a buy trade."""
        tracker = BalanceTracker(initial_balance=Decimal("1000"))

        tracker.record_buy(cost=Decimal("500"), order_id="order1")

        assert tracker.total_balance == Decimal("500")
        assert tracker.available_balance == Decimal("500")

    def test_record_sell(self):
        """Test recording a sell trade."""
        tracker = BalanceTracker(initial_balance=Decimal("500"))

        tracker.record_sell(proceeds=Decimal("600"), order_id="order1")

        assert tracker.total_balance == Decimal("1100")

    def test_sync_balance(self):
        """Test syncing with actual balance."""
        tracker = BalanceTracker(initial_balance=Decimal("1000"))

        # Simulate drift
        diff = tracker.sync_balance(Decimal("1050"), source="api")

        assert diff == Decimal("50")
        assert tracker.total_balance == Decimal("1050")

    def test_get_snapshot(self):
        """Test getting balance snapshot."""
        tracker = BalanceTracker(initial_balance=Decimal("1000"))
        tracker.reserve(Decimal("200"), order_id="order1")

        snapshot = tracker.get_snapshot()

        assert snapshot.total_balance == Decimal("1000")
        assert snapshot.reserved_balance == Decimal("200")
        assert snapshot.available_balance == Decimal("800")

    def test_history_tracking(self):
        """Test balance change history."""
        tracker = BalanceTracker(initial_balance=Decimal("1000"))

        tracker.reserve(Decimal("100"), order_id="order1")
        tracker.record_buy(cost=Decimal("100"), order_id="order1")

        history = tracker.get_history()

        assert len(history) == 2
        assert history[0].change_type == BalanceChangeType.RESERVE
        assert history[1].change_type == BalanceChangeType.TRADE_BUY


class TestMultiAccountBalanceTracker:
    """Tests for MultiAccountBalanceTracker."""

    def test_add_multiple_accounts(self):
        """Test adding multiple accounts."""
        multi = MultiAccountBalanceTracker()

        multi.add_account("whale1", Decimal("1000"))
        multi.add_account("whale2", Decimal("500"))

        assert multi.get_total_balance() == Decimal("1500")
        assert multi.get_total_available() == Decimal("1500")

    def test_get_tracker(self):
        """Test getting individual tracker."""
        multi = MultiAccountBalanceTracker()

        multi.add_account("whale1", Decimal("1000"))
        tracker = multi.get_tracker("whale1")

        assert tracker is not None
        assert tracker.total_balance == Decimal("1000")


# =============================================================================
# Position Manager Tests
# =============================================================================


class TestCopyPosition:
    """Tests for CopyPosition dataclass."""

    def test_add_shares(self):
        """Test adding shares to position."""
        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
        )

        position.add_shares(Decimal("100"), Decimal("0.50"))

        assert position.size == Decimal("100")
        assert position.average_price == Decimal("0.50")
        assert position.total_cost == Decimal("50")
        assert position.status == PositionStatus.OPEN

    def test_remove_shares_fifo(self):
        """Test FIFO accounting for sells."""
        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
        )

        # Buy at different prices
        position.add_shares(Decimal("50"), Decimal("0.40"))  # Cost: 20
        position.add_shares(Decimal("50"), Decimal("0.60"))  # Cost: 30

        assert position.size == Decimal("100")
        assert position.average_price == Decimal("0.50")  # (20+30)/100

        # Sell 60 shares at 0.70
        # FIFO: 50 @ 0.40 (cost 20) + 10 @ 0.60 (cost 6) = 26 cost basis
        # Proceeds: 60 * 0.70 = 42
        # Realized P&L: 42 - 26 = 16
        realized = position.remove_shares(Decimal("60"), Decimal("0.70"))

        assert realized == Decimal("16")
        assert position.size == Decimal("40")
        assert position.realized_pnl == Decimal("16")

    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
        )

        position.add_shares(Decimal("100"), Decimal("0.50"))
        position.update_market_price(Decimal("0.60"))

        # Cost: 50, Current value: 60
        assert position.unrealized_pnl == Decimal("10")
        assert position.unrealized_pnl_percent == Decimal("20")

    def test_close_position(self):
        """Test closing position on settlement."""
        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
        )

        position.add_shares(Decimal("100"), Decimal("0.50"))

        # Market settles at 1.0 (win)
        realized = position.close(settlement_price=Decimal("1.0"))

        # Cost: 50, Settlement: 100, P&L: 50
        assert realized == Decimal("50")
        assert position.size == Decimal("0")
        assert position.status == PositionStatus.CLOSED


class TestPositionManager:
    """Tests for PositionManager."""

    def test_record_buy_creates_position(self):
        """Test recording buy creates position."""
        manager = PositionManager("test_account")

        position = manager.record_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        assert position.size == Decimal("100")
        assert manager.get_position_count() == 1

    def test_record_sell_reduces_position(self):
        """Test recording sell reduces position."""
        manager = PositionManager("test_account")

        manager.record_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        position, realized = manager.record_sell(
            token_id="0xtoken",
            size=Decimal("50"),
            price=Decimal("0.60"),
        )

        assert position.size == Decimal("50")
        # Realized: 50 * (0.60 - 0.50) = 5
        assert realized == Decimal("5")

    def test_get_totals(self):
        """Test getting total values."""
        manager = PositionManager("test_account")

        manager.record_buy("0xtoken1", "0xcond", "Yes", Decimal("100"), Decimal("0.50"))
        manager.record_buy("0xtoken2", "0xcond", "No", Decimal("200"), Decimal("0.30"))

        assert manager.get_total_cost() == Decimal("110")  # 50 + 60
        assert manager.get_position_count() == 2

    def test_sync_from_api(self):
        """Test syncing positions from API."""
        manager = PositionManager("test_account")

        api_positions = [
            {
                "assetId": "0xtoken1",
                "conditionId": "0xcond",
                "outcome": "Yes",
                "size": "150",
                "avgPrice": "0.55",
            },
            {
                "assetId": "0xtoken2",
                "conditionId": "0xcond",
                "outcome": "No",
                "size": "100",
                "avgPrice": "0.40",
            },
        ]

        added, updated, removed = manager.sync_from_api(api_positions)

        assert added == 2
        assert updated == 0
        assert manager.get_position_count() == 2


# =============================================================================
# P&L Calculator Tests
# =============================================================================


class TestPnLCalculator:
    """Tests for PnLCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with position manager."""
        manager = PositionManager("test_account")
        return PnLCalculator(manager, initial_capital=Decimal("1000"))

    def test_record_trade(self, calculator):
        """Test recording a trade."""
        trade = TradeResult(
            trade_id="trade1",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.50"),
            cost=Decimal("50"),
        )

        calculator.record_trade(trade)

        assert len(calculator._trades) == 1

    def test_realized_pnl_tracking(self, calculator):
        """Test realized P&L tracking."""
        # Record winning trade
        win_trade = TradeResult(
            trade_id="trade1",
            token_id="0xtoken",
            outcome="Yes",
            side="SELL",
            size=Decimal("100"),
            price=Decimal("0.60"),
            cost=Decimal("60"),
            realized_pnl=Decimal("10"),
        )

        calculator.record_trade(win_trade)

        assert calculator.get_realized_pnl() == Decimal("10")
        assert calculator._winning_trades == 1

    def test_win_rate(self, calculator):
        """Test win rate calculation."""
        # Record 3 wins, 1 loss
        for i in range(3):
            calculator.record_trade(TradeResult(
                trade_id=f"win{i}",
                token_id="0x",
                outcome="Yes",
                side="SELL",
                size=Decimal("10"),
                price=Decimal("0.60"),
                cost=Decimal("6"),
                realized_pnl=Decimal("1"),
            ))

        calculator.record_trade(TradeResult(
            trade_id="loss1",
            token_id="0x",
            outcome="Yes",
            side="SELL",
            size=Decimal("10"),
            price=Decimal("0.40"),
            cost=Decimal("4"),
            realized_pnl=Decimal("-1"),
        ))

        assert calculator.get_win_rate() == Decimal("75")

    def test_performance_metrics(self, calculator):
        """Test performance metrics calculation."""
        # Add some trades
        calculator.record_trade(TradeResult(
            trade_id="trade1",
            token_id="0x",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.50"),
            cost=Decimal("50"),
        ))

        metrics = calculator.get_performance(PnLPeriod.ALL_TIME)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.trades_count == 1
        assert metrics.total_invested == Decimal("50")


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        trade = TradeResult(
            trade_id="trade1",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            size=Decimal("100"),
            price=Decimal("0.52"),
            cost=Decimal("52"),
            target_price=Decimal("0.50"),
        )

        # Slippage: (0.52 - 0.50) / 0.50 = 0.04 = 4%
        assert trade.slippage == Decimal("0.04")


# =============================================================================
# Drift Detection Tests
# =============================================================================


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_synced_position(self):
        """Test position within sync tolerance."""
        detector = DriftDetector(position_ratio=Decimal("0.01"))

        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("100"),
        )

        result = detector.check_drift(
            our_position=position,
            target_size=Decimal("10000"),  # Expected: 100 (1%)
            current_price=Decimal("0.50"),
        )

        assert result.status == DriftStatus.SYNCED
        assert result.drift_percent == Decimal("0")

    def test_minor_drift(self):
        """Test minor drift detection."""
        detector = DriftDetector(
            position_ratio=Decimal("0.01"),
            minor_threshold=Decimal("0.05"),
        )

        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("92"),  # 8% less than expected 100
        )

        result = detector.check_drift(
            our_position=position,
            target_size=Decimal("10000"),
            current_price=Decimal("0.50"),
        )

        assert result.status == DriftStatus.MINOR

    def test_significant_drift(self):
        """Test significant drift detection."""
        detector = DriftDetector(
            position_ratio=Decimal("0.01"),
            significant_threshold=Decimal("0.10"),
        )

        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("85"),  # 15% less than expected 100
        )

        result = detector.check_drift(
            our_position=position,
            target_size=Decimal("10000"),
            current_price=Decimal("0.50"),
        )

        assert result.status == DriftStatus.SIGNIFICANT
        assert result.recommended_action is not None

    def test_critical_drift(self):
        """Test critical drift detection."""
        detector = DriftDetector(
            position_ratio=Decimal("0.01"),
            critical_threshold=Decimal("0.20"),
        )

        position = CopyPosition(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("70"),  # 30% less than expected 100
        )

        result = detector.check_drift(
            our_position=position,
            target_size=Decimal("10000"),
            current_price=Decimal("0.50"),
        )

        assert result.status == DriftStatus.CRITICAL

    def test_no_position_drift(self):
        """Test detecting missing position."""
        detector = DriftDetector(position_ratio=Decimal("0.01"))

        result = detector.check_drift(
            our_position=None,
            target_size=Decimal("10000"),  # Target has position
            current_price=Decimal("0.50"),
        )

        assert result.status == DriftStatus.NO_POSITION
        assert "Buy" in result.recommended_action


# =============================================================================
# Integration Tests
# =============================================================================


class TestAccountIntegration:
    """Integration tests for account components."""

    def test_full_trade_lifecycle(self):
        """Test complete trade lifecycle from buy to sell."""
        # Setup
        balance = BalanceTracker(initial_balance=Decimal("1000"))
        positions = PositionManager("test_account")
        pnl = PnLCalculator(positions, initial_capital=Decimal("1000"))

        # Buy
        order_id = "order1"
        cost = Decimal("100")

        balance.reserve(cost, order_id)
        balance.release(order_id)
        balance.record_buy(cost, order_id)

        position = positions.record_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("200"),
            price=Decimal("0.50"),
            order_id=order_id,
        )

        pnl.record_trade(TradeResult(
            trade_id="buy1",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            size=Decimal("200"),
            price=Decimal("0.50"),
            cost=cost,
        ))

        # Verify state after buy
        assert balance.total_balance == Decimal("900")
        assert position.size == Decimal("200")

        # Sell at profit
        sell_order_id = "order2"
        sell_size = Decimal("200")
        sell_price = Decimal("0.60")
        proceeds = sell_size * sell_price  # 120

        _, realized = positions.record_sell(
            token_id="0xtoken",
            size=sell_size,
            price=sell_price,
            order_id=sell_order_id,
        )

        balance.record_sell(proceeds, sell_order_id)

        pnl.record_trade(TradeResult(
            trade_id="sell1",
            token_id="0xtoken",
            outcome="Yes",
            side="SELL",
            size=sell_size,
            price=sell_price,
            cost=proceeds,
            realized_pnl=realized,
        ))

        # Verify final state
        assert balance.total_balance == Decimal("1020")  # 900 + 120
        assert positions.get_position_count() == 0  # Position closed
        assert pnl.get_realized_pnl() == Decimal("20")  # 120 - 100
