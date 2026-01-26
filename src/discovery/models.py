"""Data structures for Account Discovery.

Defines models for:
- Discovered accounts and their scores
- Analysis snapshots with detailed metrics
- Scan history and audit trail
- Red flags and anomaly detection

Using dataclasses for in-memory state management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Any


class DiscoveryMode(str, Enum):
    """Discovery scanning modes."""

    # Cast wide net for any profitable accounts - recommended starting point
    WIDE_NET_PROFITABILITY = "wide_net_profitability"

    # Find consistent small-bet traders on niche markets
    NICHE_SPECIALIST = "niche_specialist"

    # Find accounts making many micro-bets at long odds
    MICRO_BET_HUNTER = "micro_bet_hunter"

    # Find suspicious patterns suggesting insider info
    INSIDER_DETECTION = "insider_detection"

    # Find accounts similar to a reference account
    SIMILAR_TO = "similar_to"

    # Scan holders of specific niche markets
    MARKET_HOLDER_SCAN = "market_holder_scan"

    # Custom criteria scan
    CUSTOM = "custom"


class ScanStatus(str, Enum):
    """Status of a discovery scan."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RedFlagType(str, Enum):
    """Types of red flags that can be detected."""

    # Fresh account with immediate large position
    FRESH_ACCOUNT_LARGE_BET = "fresh_account_large_bet"

    # Large bet on extremely unlikely event near expiry
    YOLO_BET_NEAR_EXPIRY = "yolo_bet_near_expiry"

    # Sudden position building before news
    PRE_NEWS_ACCUMULATION = "pre_news_accumulation"

    # Single large lucky win dominates P/L
    SINGLE_WIN_DEPENDENT = "single_win_dependent"

    # Win rate suspiciously high (>85%)
    SUSPICIOUS_WIN_RATE = "suspicious_win_rate"

    # Heavy concentration in single condition/bet
    POSITION_CONCENTRATION = "position_concentration"

    # Copying known insider patterns
    INSIDER_PATTERN_MATCH = "insider_pattern_match"

    # Account less than 30 days old
    FRESH_ACCOUNT = "fresh_account"

    # Mostly trades mainstream categories
    MAINSTREAM_HEAVY = "mainstream_heavy"

    # P/L curve shows high volatility / low Sharpe ratio
    VOLATILE_PL_CURVE = "volatile_pl_curve"

    # Large peak-to-trough drawdowns
    LARGE_DRAWDOWNS = "large_drawdowns"

    # Account with very large average position size (whale)
    WHALE_ACCOUNT = "whale_account"

    # Not enough trade history for reliable analysis
    INSUFFICIENT_HISTORY = "insufficient_history"


class RedFlagSeverity(str, Enum):
    """Severity levels for red flags."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RedFlag:
    """A detected red flag for an account."""

    type: RedFlagType
    severity: RedFlagSeverity
    description: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "description": self.description,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class PLCurveMetrics:
    """Metrics computed from P/L curve analysis."""

    total_realized_pnl: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int
    win_rate: float
    avg_win_size: Decimal
    avg_loss_size: Decimal
    profit_factor: float
    longest_win_streak: int
    longest_loss_streak: int
    current_streak: int
    largest_win_pct_of_total: float
    top_3_wins_pct_of_total: float
    avg_recovery_time_days: float
    # Win/loss counts
    win_count: int = 0
    loss_count: int = 0
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")

    # Copytrade viability metrics (added based on empirical analysis)
    # These help identify traders whose returns are copyable vs luck-dependent
    top_5_wins_pct_of_total: float = 0.0  # Extended concentration check
    top_10_wins_pct_of_total: float = 0.0  # Even more distribution check
    market_win_rate: float = 0.0  # % of unique markets that were profitable
    markets_profitable: int = 0  # Count of unique markets with profit
    markets_unprofitable: int = 0  # Count of unique markets with loss
    redemption_win_count: int = 0  # Wins from holding to resolution
    sell_win_count: int = 0  # Wins from selling early
    redemption_pct_of_profit: float = 0.0  # What % of profits came from resolutions
    # Simulated capture analysis (what % of profit at random capture rates)
    simulated_50pct_capture_median: float = 0.0  # Median P/L capturing 50% of trades
    simulated_25pct_capture_median: float = 0.0  # Median P/L capturing 25% of trades

    # High watermark drawdown (robust measure of being "down from peak")
    # Uses leaderboard realized P/L as proxy for peak, compared to unrealized losses
    off_high_watermark_pct: float = 0.0  # How far below peak (0.0 = at peak, 0.5 = 50% down)
    unrealized_pnl: float = 0.0  # Current total unrealized P/L from open positions


@dataclass
class TradingPatternMetrics:
    """Metrics about trading patterns and behavior."""

    avg_position_size_usd: Decimal
    median_position_size_usd: Decimal
    max_position_size_usd: Decimal
    position_size_std_dev: Decimal

    pct_trades_under_5c: float
    pct_trades_under_10c: float
    pct_trades_under_20c: float
    pct_trades_over_80c: float

    avg_entry_odds: Decimal
    median_entry_odds: Decimal

    total_trades: int
    trades_per_day_avg: float
    active_days: int
    account_age_days: int

    unique_markets_traded: int
    markets_per_trade_ratio: float

    category_breakdown: dict[str, float]
    niche_market_pct: float

    avg_hold_time_hours: float
    pct_trades_near_expiry: float

    # Activity recency and trade direction
    days_since_last_trade: int = 0
    trades_last_7d: int = 0  # Number of trades in the last 7 days
    trades_last_30d: int = 0  # Number of trades in the last 30 days
    buy_sell_ratio: float = 0.5  # Ratio of buys to total trades

    # Primary category classification
    primary_category: str = "Diversified"  # Crypto, Politics, Sports, Finance, Weather, Tech, Culture, Diversified
    category_concentration: float = 0.0  # How concentrated they are in their primary category (0.0-1.0)


@dataclass
class InsiderSignals:
    """Signals that may indicate insider trading."""

    account_age_days: int
    first_trade_date: Optional[datetime]

    large_bets_on_unlikely_events: list[dict]
    pre_resolution_accumulation: list[dict]
    fresh_account_large_positions: bool

    single_market_concentration: float
    largest_position_usd: Decimal
    largest_position_odds_at_entry: Decimal

    trades_in_final_24h: int
    large_trades_in_final_24h: list[dict]

    insider_probability_score: float
