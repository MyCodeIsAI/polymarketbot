"""P&L calculation and tracking for copy trading.

This module provides:
- Realized P&L tracking
- Unrealized P&L calculation
- P&L attribution by market/trade
- Performance metrics (ROI, win rate, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..utils.logging import get_logger
from .positions import CopyPosition, PositionManager

logger = get_logger(__name__)


class PnLPeriod(str, Enum):
    """Time periods for P&L reporting."""

    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    ALL_TIME = "all_time"


@dataclass
class TradeResult:
    """Result of a single trade for P&L tracking."""

    trade_id: str
    token_id: str
    outcome: str
    side: str  # BUY or SELL

    size: Decimal
    price: Decimal
    cost: Decimal  # Total cost for buy, or proceeds for sell

    realized_pnl: Optional[Decimal] = None  # Only for sells
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Copy trade metadata
    target_name: Optional[str] = None
    target_price: Optional[Decimal] = None  # Price target got

    @property
    def is_buy(self) -> bool:
        return self.side == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.side == "SELL"

    @property
    def slippage(self) -> Optional[Decimal]:
        """Calculate slippage vs target price."""
        if self.target_price is None:
            return None

        if self.is_buy:
            return (self.price - self.target_price) / self.target_price
        else:
            return (self.target_price - self.price) / self.target_price


@dataclass
class DailyPnL:
    """P&L summary for a single day."""

    date: datetime
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_volume: Decimal = Decimal("0")

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def win_rate(self) -> Decimal:
        if self.trades_count == 0:
            return Decimal("0")
        return Decimal(self.winning_trades) / Decimal(self.trades_count)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a period."""

    period: PnLPeriod
    start_date: datetime
    end_date: datetime

    # P&L
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Investment
    total_invested: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")

    # Trade statistics
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")

    # Volume
    total_volume: Decimal = Decimal("0")
    avg_trade_size: Decimal = Decimal("0")

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def roi(self) -> Decimal:
        """Return on investment percentage."""
        if self.total_invested <= 0:
            return Decimal("0")
        return (self.total_pnl / self.total_invested) * 100

    @property
    def win_rate(self) -> Decimal:
        """Win rate percentage."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return Decimal("0")
        return (Decimal(self.winning_trades) / Decimal(total)) * 100

    @property
    def profit_factor(self) -> Decimal:
        """Profit factor (gross profit / gross loss)."""
        if self.largest_loss == 0:
            return Decimal("999")
        # This is simplified - would need actual gross profit/loss
        return abs(self.largest_win / self.largest_loss) if self.largest_loss else Decimal("999")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "period": self.period.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_pnl": str(self.total_pnl),
            "total_invested": str(self.total_invested),
            "roi_percent": str(round(self.roi, 2)),
            "trades_count": self.trades_count,
            "win_rate_percent": str(round(self.win_rate, 2)),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
        }


class PnLCalculator:
    """Calculates P&L metrics for a copy trading account.

    Tracks:
    - Realized P&L from closed trades
    - Unrealized P&L from open positions
    - Performance metrics over time

    Example:
        calculator = PnLCalculator(position_manager)

        # Record a trade
        calculator.record_trade(trade_result)

        # Get performance
        metrics = calculator.get_performance(PnLPeriod.TODAY)
        print(f"ROI: {metrics.roi}%")
    """

    def __init__(
        self,
        position_manager: PositionManager,
        initial_capital: Decimal = Decimal("0"),
    ):
        """Initialize P&L calculator.

        Args:
            position_manager: Position manager for unrealized P&L
            initial_capital: Starting capital for ROI calculation
        """
        self.position_manager = position_manager
        self.initial_capital = initial_capital

        # Trade history
        self._trades: list[TradeResult] = []

        # Daily summaries
        self._daily_pnl: dict[str, DailyPnL] = {}

        # Running totals
        self._total_realized_pnl = Decimal("0")
        self._total_invested = Decimal("0")
        self._winning_trades = 0
        self._losing_trades = 0

    def record_trade(self, trade: TradeResult) -> None:
        """Record a trade result.

        Args:
            trade: Trade result to record
        """
        self._trades.append(trade)

        # Update running totals
        if trade.is_buy:
            self._total_invested += trade.cost

        if trade.realized_pnl is not None:
            self._total_realized_pnl += trade.realized_pnl

            if trade.realized_pnl > 0:
                self._winning_trades += 1
            elif trade.realized_pnl < 0:
                self._losing_trades += 1

        # Update daily summary
        date_key = trade.timestamp.strftime("%Y-%m-%d")
        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = DailyPnL(
                date=trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            )

        daily = self._daily_pnl[date_key]
        daily.trades_count += 1
        daily.total_volume += trade.cost

        if trade.realized_pnl is not None:
            daily.realized_pnl += trade.realized_pnl
            if trade.realized_pnl > 0:
                daily.winning_trades += 1
            elif trade.realized_pnl < 0:
                daily.losing_trades += 1

        logger.debug(
            "trade_recorded",
            trade_id=trade.trade_id,
            side=trade.side,
            realized_pnl=str(trade.realized_pnl) if trade.realized_pnl else None,
        )

    def get_realized_pnl(self) -> Decimal:
        """Get total realized P&L.

        Returns:
            Total realized P&L
        """
        return self._total_realized_pnl

    def get_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L from open positions.

        Returns:
            Total unrealized P&L
        """
        return self.position_manager.get_total_unrealized_pnl()

    def get_total_pnl(self) -> Decimal:
        """Get total P&L (realized + unrealized).

        Returns:
            Total P&L
        """
        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_roi(self) -> Decimal:
        """Get overall return on investment.

        Returns:
            ROI as percentage
        """
        capital = self.initial_capital if self.initial_capital > 0 else self._total_invested
        if capital <= 0:
            return Decimal("0")
        return (self.get_total_pnl() / capital) * 100

    def get_performance(self, period: PnLPeriod = PnLPeriod.ALL_TIME) -> PerformanceMetrics:
        """Get performance metrics for a period.

        Args:
            period: Time period for metrics

        Returns:
            PerformanceMetrics for the period
        """
        now = datetime.utcnow()

        # Determine date range
        if period == PnLPeriod.TODAY:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PnLPeriod.WEEK:
            start_date = now - timedelta(days=7)
        elif period == PnLPeriod.MONTH:
            start_date = now - timedelta(days=30)
        else:
            start_date = datetime.min

        # Filter trades in period
        period_trades = [
            t for t in self._trades
            if t.timestamp >= start_date
        ]

        # Calculate metrics
        realized_pnl = Decimal("0")
        total_invested = Decimal("0")
        total_volume = Decimal("0")
        winning = 0
        losing = 0
        largest_win = Decimal("0")
        largest_loss = Decimal("0")

        for trade in period_trades:
            total_volume += trade.cost

            if trade.is_buy:
                total_invested += trade.cost

            if trade.realized_pnl is not None:
                realized_pnl += trade.realized_pnl

                if trade.realized_pnl > 0:
                    winning += 1
                    if trade.realized_pnl > largest_win:
                        largest_win = trade.realized_pnl
                elif trade.realized_pnl < 0:
                    losing += 1
                    if trade.realized_pnl < largest_loss:
                        largest_loss = trade.realized_pnl

        # Get unrealized from current positions
        unrealized_pnl = self.get_unrealized_pnl()
        current_value = self.position_manager.get_total_value()

        return PerformanceMetrics(
            period=period,
            start_date=start_date,
            end_date=now,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_invested=total_invested,
            current_value=current_value,
            trades_count=len(period_trades),
            winning_trades=winning,
            losing_trades=losing,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_volume=total_volume,
            avg_trade_size=total_volume / len(period_trades) if period_trades else Decimal("0"),
        )

    def get_daily_summary(self, date: Optional[datetime] = None) -> Optional[DailyPnL]:
        """Get P&L summary for a specific day.

        Args:
            date: Date to get summary for (default: today)

        Returns:
            DailyPnL or None if no data
        """
        if date is None:
            date = datetime.utcnow()

        date_key = date.strftime("%Y-%m-%d")
        return self._daily_pnl.get(date_key)

    def get_daily_history(self, days: int = 30) -> list[DailyPnL]:
        """Get daily P&L history.

        Args:
            days: Number of days to include

        Returns:
            List of DailyPnL sorted by date
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_key = cutoff.strftime("%Y-%m-%d")

        result = []
        for date_key in sorted(self._daily_pnl.keys()):
            if date_key >= cutoff_key:
                result.append(self._daily_pnl[date_key])

        return result

    def get_trade_history(
        self,
        limit: int = 100,
        token_id: Optional[str] = None,
    ) -> list[TradeResult]:
        """Get trade history.

        Args:
            limit: Maximum trades to return
            token_id: Filter by token ID

        Returns:
            List of TradeResult
        """
        trades = self._trades

        if token_id:
            trades = [t for t in trades if t.token_id == token_id]

        return trades[-limit:]

    def get_win_rate(self) -> Decimal:
        """Get overall win rate.

        Returns:
            Win rate as percentage
        """
        total = self._winning_trades + self._losing_trades
        if total == 0:
            return Decimal("0")
        return (Decimal(self._winning_trades) / Decimal(total)) * 100

    def get_summary(self) -> dict:
        """Get P&L summary.

        Returns:
            Summary dictionary
        """
        return {
            "realized_pnl": str(self.get_realized_pnl()),
            "unrealized_pnl": str(self.get_unrealized_pnl()),
            "total_pnl": str(self.get_total_pnl()),
            "roi_percent": str(round(self.get_roi(), 2)),
            "win_rate_percent": str(round(self.get_win_rate(), 2)),
            "total_trades": len(self._trades),
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "total_invested": str(self._total_invested),
        }


class MarketPnL:
    """P&L breakdown by market/token.

    Tracks performance for each market separately for
    analysis and reporting.
    """

    def __init__(self):
        """Initialize market P&L tracker."""
        self._market_data: dict[str, dict] = {}

    def record_trade(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
        trade: TradeResult,
    ) -> None:
        """Record a trade for a market.

        Args:
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            trade: Trade result
        """
        if token_id not in self._market_data:
            self._market_data[token_id] = {
                "condition_id": condition_id,
                "outcome": outcome,
                "trades": [],
                "realized_pnl": Decimal("0"),
                "total_bought": Decimal("0"),
                "total_sold": Decimal("0"),
            }

        data = self._market_data[token_id]
        data["trades"].append(trade)

        if trade.is_buy:
            data["total_bought"] += trade.size
        else:
            data["total_sold"] += trade.size

        if trade.realized_pnl:
            data["realized_pnl"] += trade.realized_pnl

    def get_market_pnl(self, token_id: str) -> Optional[dict]:
        """Get P&L data for a specific market.

        Args:
            token_id: Token ID

        Returns:
            Market P&L data or None
        """
        return self._market_data.get(token_id)

    def get_top_performers(self, limit: int = 10) -> list[tuple[str, Decimal]]:
        """Get top performing markets by realized P&L.

        Args:
            limit: Number of markets to return

        Returns:
            List of (token_id, pnl) tuples sorted by P&L
        """
        market_pnls = [
            (tid, data["realized_pnl"])
            for tid, data in self._market_data.items()
        ]
        return sorted(market_pnls, key=lambda x: x[1], reverse=True)[:limit]

    def get_worst_performers(self, limit: int = 10) -> list[tuple[str, Decimal]]:
        """Get worst performing markets by realized P&L.

        Args:
            limit: Number of markets to return

        Returns:
            List of (token_id, pnl) tuples sorted by P&L ascending
        """
        market_pnls = [
            (tid, data["realized_pnl"])
            for tid, data in self._market_data.items()
        ]
        return sorted(market_pnls, key=lambda x: x[1])[:limit]
