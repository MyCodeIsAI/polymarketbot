"""Multi-Phase Account Analyzer for Discovery.

Implements efficient three-phase scanning:
- Phase 1: Quick filter (leaderboard data, 0 API calls)
- Phase 2: Light scan (recent activity, 1 API call)
- Phase 3: Deep analysis (full history, 2-3 API calls)

This reduces API calls from ~1500+ (naive) to ~300-500 for typical scans.
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Any, Callable
from collections import defaultdict
import statistics

from ..api.data import DataAPIClient, Activity, ActivityType, TradeSide, Position
from ..api.gamma import GammaAPIClient, Market
from ..utils.logging import get_logger

from .models import (
    DiscoveryMode,
    PLCurveMetrics,
    TradingPatternMetrics,
    InsiderSignals,
    RedFlag,
    RedFlagType,
)
from .scoring import ScoringEngine, ScoringResult, MODE_CONFIGS

logger = get_logger(__name__)


# =============================================================================
# Category Classification
# =============================================================================

CATEGORY_KEYWORDS = {
    "politics": ["president", "election", "trump", "biden", "congress", "senate", "governor", "vote", "primary", "democrat", "republican"],
    "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "ufc", "boxing", "tennis", "golf", "formula"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "token", "coin", "defi", "solana", "dogecoin"],
    "weather": ["temperature", "weather", "snow", "rain", "hurricane", "storm", "heat", "cold", "climate", "tornado", "flood"],
    "economics": ["fed", "rate", "inflation", "gdp", "unemployment", "economy", "recession", "cpi", "fomc", "interest", "jobs"],
    "finance": ["stock", "market", "nasdaq", "s&p", "dow", "price", "trading", "earnings", "ipo"],
    "tech": ["ai", "openai", "google", "apple", "microsoft", "tesla", "tech", "software", "chatgpt", "meta", "nvidia"],
    "culture": ["oscar", "grammy", "movie", "music", "celebrity", "entertainment", "award", "film"],
}

MAINSTREAM_CATEGORIES = {"politics", "sports"}
NICHE_CATEGORIES = {"weather", "economics", "tech", "finance"}

# Primary category classification for traders
# Threshold: ≥60% of trades in a category to be classified as that type
PRIMARY_CATEGORY_THRESHOLD = 0.60
SPECIALIST_THRESHOLD = 0.80  # ≥80% = true specialist

# Simplified trader categories (maps raw categories to display categories)
TRADER_CATEGORIES = {
    "crypto": "Crypto",
    "politics": "Politics",
    "sports": "Sports",
    "finance": "Finance",  # Includes economics
    "economics": "Finance",  # Map to Finance
    "weather": "Weather",
    "tech": "Tech",
    "culture": "Culture",
}
DIVERSIFIED_CATEGORY = "Diversified"


def classify_trader_category(category_breakdown: dict[str, float]) -> tuple[str, float]:
    """Classify a trader into their primary category based on trade distribution.

    Args:
        category_breakdown: Dict of category -> percentage (0.0 to 1.0)

    Returns:
        Tuple of (primary_category, concentration_pct)
        primary_category is one of: Crypto, Politics, Sports, Finance, Weather, Tech, Culture, Diversified
    """
    if not category_breakdown:
        return DIVERSIFIED_CATEGORY, 0.0

    # Combine economics into finance for classification
    combined = {}
    for cat, pct in category_breakdown.items():
        cat_lower = cat.lower()
        if cat_lower == "economics":
            combined["finance"] = combined.get("finance", 0.0) + pct
        else:
            combined[cat_lower] = combined.get(cat_lower, 0.0) + pct

    if not combined:
        return DIVERSIFIED_CATEGORY, 0.0

    # Find the dominant category
    top_category = max(combined, key=combined.get)
    top_pct = combined[top_category]

    # If below threshold, they're diversified
    if top_pct < PRIMARY_CATEGORY_THRESHOLD:
        return DIVERSIFIED_CATEGORY, top_pct

    # Map to display name
    display_name = TRADER_CATEGORIES.get(top_category, top_category.title())
    return display_name, top_pct


# =============================================================================
# Data Classes for Each Phase
# =============================================================================

@dataclass
class LeaderboardEntry:
    """Data from leaderboard API (Phase 1 input)."""
    wallet_address: str
    rank: int
    total_pnl: Decimal
    volume: Decimal
    num_trades: int
    position_count: int
    avg_position_size: Optional[Decimal] = None
    categories: list[str] = field(default_factory=list)


@dataclass
class QuickFilterResult:
    """Phase 1 result: Quick filter using only leaderboard data."""
    wallet_address: str
    passed: bool
    rejection_reason: Optional[str] = None

    # Metrics from leaderboard
    total_pnl: Decimal = Decimal("0")
    num_trades: int = 0
    estimated_avg_position: Decimal = Decimal("0")

    # Quick scores
    preliminary_score: float = 0.0
    priority_rank: int = 0  # For ordering phase 2


@dataclass
class LightScanResult:
    """Phase 2 result: Light scan with 1 API call."""
    wallet_address: str
    passed: bool
    rejection_reason: Optional[str] = None

    # Core metrics from recent activity
    recent_trade_count: int = 0
    account_age_days: int = 0
    is_active: bool = False
    last_trade_date: Optional[datetime] = None

    # Quick pattern indicators
    avg_position_size: Decimal = Decimal("0")
    pct_long_odds: float = 0.0  # % of trades at <20c
    mainstream_pct: float = 0.0
    niche_pct: float = 0.0

    # Scoring
    light_score: float = 0.0
    priority_rank: int = 0  # For ordering phase 3


@dataclass
class TradeRecord:
    """Enriched trade record for deep analysis."""
    timestamp: datetime
    market_id: str
    market_title: str
    token_id: str
    side: TradeSide
    size: Decimal
    price: Decimal
    usd_value: Decimal
    category: str
    is_resolved: bool = False
    resolution_outcome: Optional[bool] = None
    pnl: Decimal = Decimal("0")


@dataclass
class DeepAnalysisResult:
    """Phase 3 result: Full deep analysis."""
    wallet_address: str

    # Computed metrics
    pl_metrics: Optional[PLCurveMetrics]
    pattern_metrics: Optional[TradingPatternMetrics]
    insider_signals: Optional[InsiderSignals]

    # Scoring result
    scoring_result: Optional[ScoringResult] = None

    # Raw data for display
    trades: list[TradeRecord] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    pl_curve_data: list[dict] = field(default_factory=list)
    market_categories: dict[str, float] = field(default_factory=dict)
    recent_trades_sample: list[dict] = field(default_factory=list)

    # Category-specific metrics
    category_win_rates: dict[str, float] = field(default_factory=dict)

    # Authoritative P/L from leaderboard (set by service, not computed)
    total_pnl: Optional[Decimal] = None

    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "wallet_address": self.wallet_address,
            "composite_score": self.scoring_result.composite_score if self.scoring_result else 0,
            "passes_threshold": self.scoring_result.passes_threshold if self.scoring_result else False,
            "score_breakdown": self.scoring_result.to_dict() if self.scoring_result else {},
            "red_flags": self.scoring_result.red_flags if self.scoring_result else [],
            "red_flag_count": self.scoring_result.red_flag_count if self.scoring_result else 0,
            # Authoritative P/L from leaderboard (more accurate than computed pl_metrics)
            "total_pnl": float(self.total_pnl) if self.total_pnl else None,
            "pl_metrics": self._metrics_to_dict(self.pl_metrics),
            "pattern_metrics": self._metrics_to_dict(self.pattern_metrics),
            "insider_signals": self._metrics_to_dict(self.insider_signals),
            "market_categories": self.market_categories,
            "category_win_rates": self.category_win_rates,
            "recent_trades_sample": self.recent_trades_sample,
            "pl_curve_data": self.pl_curve_data,
            "error": self.error,
        }

    def _metrics_to_dict(self, metrics) -> Optional[dict]:
        if not metrics:
            return None
        result = {}
        for key, value in metrics.__dict__.items():
            if isinstance(value, Decimal):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


# =============================================================================
# Multi-Phase Analyzer
# =============================================================================

class AccountAnalyzer:
    """Multi-phase account analyzer for efficient discovery scanning."""

    def __init__(
        self,
        data_client: Optional[DataAPIClient] = None,
        gamma_client: Optional[GammaAPIClient] = None,
        mode: DiscoveryMode = DiscoveryMode.NICHE_SPECIALIST,
    ):
        """Initialize analyzer.

        Args:
            data_client: Data API client
            gamma_client: Gamma API client
            mode: Discovery mode for scoring configuration
        """
        self._data_client = data_client
        self._gamma_client = gamma_client
        self._owns_clients = data_client is None
        self._market_cache: dict[str, Market] = {}
        self._failed_condition_ids: set[str] = set()  # Track failed lookups to avoid retries

        # Scoring engine with mode configuration
        self.scoring_engine = ScoringEngine(mode)
        self.mode = mode

        # Statistics tracking
        self._stats = {
            "phase1_processed": 0,
            "phase1_passed": 0,
            "phase2_processed": 0,
            "phase2_passed": 0,
            "phase3_processed": 0,
            "phase3_passed": 0,
            "api_calls": 0,
        }

    async def __aenter__(self) -> "AccountAnalyzer":
        """Async context manager entry."""
        if self._owns_clients:
            self._data_client = DataAPIClient()
            self._gamma_client = GammaAPIClient()
            await self._data_client.__aenter__()
            await self._gamma_client.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._owns_clients:
            if self._data_client:
                await self._data_client.__aexit__(*args)
            if self._gamma_client:
                await self._gamma_client.__aexit__(*args)

    def set_mode(self, mode: DiscoveryMode) -> None:
        """Change discovery mode."""
        self.mode = mode
        self.scoring_engine.set_mode(mode)

    def get_stats(self) -> dict:
        """Get scanning statistics."""
        return {
            **self._stats,
            "phase1_pass_rate": self._stats["phase1_passed"] / max(1, self._stats["phase1_processed"]),
            "phase2_pass_rate": self._stats["phase2_passed"] / max(1, self._stats["phase2_processed"]),
            "phase3_pass_rate": self._stats["phase3_passed"] / max(1, self._stats["phase3_processed"]),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {key: 0 for key in self._stats}

    # =========================================================================
    # PHASE 1: Quick Filter (No API Calls)
    # =========================================================================

    def quick_filter(
        self,
        entry: LeaderboardEntry,
        min_trades: int = 50,
        max_trades: int = None,
        min_pnl: Decimal = Decimal("0"),
        max_avg_position: Decimal = Decimal("1000"),
    ) -> QuickFilterResult:
        """Phase 1: Quick filter using only leaderboard data.

        Args:
            entry: Leaderboard data
            min_trades: Minimum trades required
            max_trades: Maximum trades allowed (filter out HFT bots)
            min_pnl: Minimum total P/L required
            max_avg_position: Maximum average position size

        Returns:
            QuickFilterResult with pass/fail and priority ranking
        """
        self._stats["phase1_processed"] += 1

        # Get mode-specific hard filter thresholds
        config = self.scoring_engine.config

        # Use config thresholds if available
        if "min_trades" in config.hard_filters:
            min_trades = int(config.hard_filters["min_trades"].threshold)
        if "max_trades" in config.hard_filters:
            max_trades = int(config.hard_filters["max_trades"].threshold)
        if "must_be_profitable" in config.hard_filters:
            min_pnl = Decimal(str(config.hard_filters["must_be_profitable"].threshold))
        if "max_avg_position" in config.hard_filters:
            max_avg_position = Decimal(str(config.hard_filters["max_avg_position"].threshold))

        # Calculate estimated avg position
        estimated_avg = Decimal("0")
        if entry.num_trades > 0 and entry.volume > 0:
            estimated_avg = entry.volume / Decimal(str(entry.num_trades))

        # Hard filter checks
        # Note: Leaderboard API often returns 0 for trade counts, so we only
        # apply this filter if we actually have trade data. Trade count will
        # be verified in Phase 2 when we fetch detailed account data.
        if entry.num_trades > 0 and entry.num_trades < min_trades:
            return QuickFilterResult(
                wallet_address=entry.wallet_address,
                passed=False,
                rejection_reason=f"Insufficient trades: {entry.num_trades} < {min_trades}",
                total_pnl=entry.total_pnl,
                num_trades=entry.num_trades,
                estimated_avg_position=estimated_avg,
            )

        # HFT filter: reject accounts with too many trades (likely bots)
        if max_trades is not None and entry.num_trades > 0 and entry.num_trades > max_trades:
            return QuickFilterResult(
                wallet_address=entry.wallet_address,
                passed=False,
                rejection_reason=f"HFT filter: {entry.num_trades:,} > {max_trades:,} trades",
                total_pnl=entry.total_pnl,
                num_trades=entry.num_trades,
                estimated_avg_position=estimated_avg,
            )

        if entry.total_pnl < min_pnl:
            return QuickFilterResult(
                wallet_address=entry.wallet_address,
                passed=False,
                rejection_reason=f"Not profitable: ${entry.total_pnl:.2f}",
                total_pnl=entry.total_pnl,
                num_trades=entry.num_trades,
                estimated_avg_position=estimated_avg,
            )

        # Mode-specific checks
        if self.mode in [DiscoveryMode.MICRO_BET_HUNTER, DiscoveryMode.NICHE_SPECIALIST]:
            if estimated_avg > max_avg_position:
                return QuickFilterResult(
                    wallet_address=entry.wallet_address,
                    passed=False,
                    rejection_reason=f"Position too large: ${estimated_avg:.0f} > ${max_avg_position:.0f}",
                    total_pnl=entry.total_pnl,
                    num_trades=entry.num_trades,
                    estimated_avg_position=estimated_avg,
                )

        # Calculate preliminary score for priority ranking
        score = 50.0

        # Volume bonus (more trading = more data)
        if entry.num_trades >= 200:
            score += 15
        elif entry.num_trades >= 100:
            score += 10

        # P/L bonus (proven profitability)
        pnl_float = float(entry.total_pnl)
        if pnl_float >= 10000:
            score += 20
        elif pnl_float >= 5000:
            score += 15
        elif pnl_float >= 1000:
            score += 10
        elif pnl_float >= 500:
            score += 5

        # Position size preference (mode-specific)
        avg_float = float(estimated_avg)
        if self.mode == DiscoveryMode.MICRO_BET_HUNTER:
            # Prefer smaller positions
            if avg_float < 20:
                score += 15
            elif avg_float < 50:
                score += 10
            elif avg_float > 200:
                score -= 10
        elif self.mode == DiscoveryMode.INSIDER_DETECTION:
            # Prefer larger positions (more suspicious)
            if avg_float > 1000:
                score += 20
            elif avg_float > 500:
                score += 15

        # Niche category bonus (if available from leaderboard)
        for cat in entry.categories:
            if cat in NICHE_CATEGORIES:
                score += 5
            if cat in MAINSTREAM_CATEGORIES:
                score -= 3

        self._stats["phase1_passed"] += 1

        return QuickFilterResult(
            wallet_address=entry.wallet_address,
            passed=True,
            total_pnl=entry.total_pnl,
            num_trades=entry.num_trades,
            estimated_avg_position=estimated_avg,
            preliminary_score=max(0, min(100, score)),
        )

    def batch_quick_filter(
        self,
        entries: list[LeaderboardEntry],
    ) -> tuple[list[QuickFilterResult], list[QuickFilterResult]]:
        """Batch quick filter with priority ranking.

        Returns:
            Tuple of (passed_results, rejected_results), passed sorted by priority
        """
        passed = []
        rejected = []

        for entry in entries:
            result = self.quick_filter(entry)
            if result.passed:
                passed.append(result)
            else:
                rejected.append(result)

        # Sort passed by preliminary score (highest first)
        passed.sort(key=lambda x: x.preliminary_score, reverse=True)

        # Assign priority ranks
        for i, result in enumerate(passed):
            result.priority_rank = i + 1

        return passed, rejected

    # =========================================================================
    # PHASE 2: Light Scan (1 API Call)
    # =========================================================================

    async def light_scan(
        self,
        wallet_address: str,
        recent_days: int = 14,
        min_recent_trades: int = 5,
    ) -> LightScanResult:
        """Phase 2: Light scan with 1 API call for recent activity.

        Fetches only recent trades to quickly assess:
        - Is the account still active?
        - Basic trading patterns
        - Category preferences

        Args:
            wallet_address: Wallet to scan
            recent_days: Days to look back
            min_recent_trades: Minimum recent trades to pass

        Returns:
            LightScanResult with quick metrics
        """
        self._stats["phase2_processed"] += 1
        wallet = wallet_address.lower()

        try:
            # Fetch recent activity only (1 API call)
            since_timestamp = int((datetime.utcnow() - timedelta(days=recent_days)).timestamp())

            activities = await self._data_client.get_activity(
                user=wallet,
                activity_type=ActivityType.TRADE,
                start_timestamp=since_timestamp,
                limit=200,  # Limit for efficiency
            )
            self._stats["api_calls"] += 1

            if not activities:
                # Wide net mode: don't reject just because no recent trades
                # Profitable accounts may not trade frequently
                if self.mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
                    # Pass them through to Phase 3 which will do full analysis
                    self._stats["phase2_passed"] += 1
                    return LightScanResult(
                        wallet_address=wallet,
                        passed=True,  # Let Phase 3 evaluate with full history
                        rejection_reason=None,
                        recent_trade_count=0,
                        is_active=False,
                        light_score=40,  # Neutral-ish score
                    )
                else:
                    return LightScanResult(
                        wallet_address=wallet,
                        passed=False,
                        rejection_reason=f"No trades in last {recent_days} days",
                    )

            # Quick metrics
            recent_count = len(activities)
            last_trade = max(a.timestamp for a in activities)
            days_since_last = (datetime.utcnow() - last_trade).days

            # Position sizes
            sizes = [float(a.usd_value) for a in activities if a.usd_value > 0]
            avg_size = Decimal(str(statistics.mean(sizes))) if sizes else Decimal("0")

            # Odds analysis
            prices = [float(a.price) for a in activities if a.price and a.price > 0]
            pct_long_odds = len([p for p in prices if p < 0.20]) / len(prices) if prices else 0.0

            # Category analysis (quick - just from recent activity)
            # Skip for wide_net mode - we don't need categories and it's expensive
            if self.mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
                mainstream_pct = 0.5  # Neutral
                niche_pct = 0.5
            else:
                category_counts = defaultdict(int)
                for activity in activities:
                    cat = await self._quick_categorize(activity.condition_id)
                    category_counts[cat] += 1

                total_cats = sum(category_counts.values())
                mainstream_pct = sum(category_counts.get(c, 0) for c in MAINSTREAM_CATEGORIES) / max(1, total_cats)
                niche_pct = sum(category_counts.get(c, 0) for c in NICHE_CATEGORIES) / max(1, total_cats)

            # Estimate account age (would need historical data for precise)
            account_age = recent_days if recent_count >= min_recent_trades else 0

            # Light scoring
            score = 50.0

            # Activity bonus
            if days_since_last < 3:
                score += 15
            elif days_since_last < 7:
                score += 10
            elif days_since_last > 14:
                score -= 10

            # Volume bonus
            if recent_count >= 50:
                score += 15
            elif recent_count >= 20:
                score += 10

            # Mode-specific scoring
            if self.mode == DiscoveryMode.MICRO_BET_HUNTER:
                if pct_long_odds > 0.5:
                    score += 20
                elif pct_long_odds > 0.3:
                    score += 10
                if float(avg_size) < 30:
                    score += 15
                elif float(avg_size) > 100:
                    score -= 10

            elif self.mode == DiscoveryMode.NICHE_SPECIALIST:
                if niche_pct > 0.5:
                    score += 25
                elif niche_pct > 0.3:
                    score += 15
                if mainstream_pct > 0.4:
                    score -= 20

            elif self.mode == DiscoveryMode.INSIDER_DETECTION:
                # For insider, we want unusual patterns
                if pct_long_odds > 0.3 and float(avg_size) > 200:
                    score += 30  # Large bets on unlikely events
                if recent_count < 10:
                    score += 15  # Fresh activity

            # Hard filter: activity check
            # Wide net mode: VERY relaxed - we want profitable accounts regardless of recent activity
            if self.mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
                # For wide net, pass accounts through to Phase 3 which will do full analysis
                # Many profitable traders made money historically and may not be active now
                # Only reject if there's NO activity at all (already handled at line 526)
                is_active = True  # Let Phase 3 evaluate with full history
            else:
                is_active = days_since_last < 14 and recent_count >= min_recent_trades

            # Mode-specific hard filters
            config = self.scoring_engine.config

            passed = is_active
            rejection_reason = None

            if not is_active:
                rejection_reason = f"Inactive: last trade {days_since_last} days ago, only {recent_count} recent trades"
                passed = False

            # Note: Mainstream check removed from Phase 2 - it's too strict based on
            # recent trades only. Full category analysis happens in Phase 3 deep
            # analysis with complete trade history for accurate mainstream %.

            if passed:
                self._stats["phase2_passed"] += 1

            return LightScanResult(
                wallet_address=wallet,
                passed=passed,
                rejection_reason=rejection_reason,
                recent_trade_count=recent_count,
                account_age_days=account_age,
                is_active=is_active,
                last_trade_date=last_trade,
                avg_position_size=avg_size,
                pct_long_odds=pct_long_odds,
                mainstream_pct=mainstream_pct,
                niche_pct=niche_pct,
                light_score=max(0, min(100, score)),
            )

        except Exception as e:
            logger.error("light_scan_failed", wallet=wallet[:10], error=str(e))
            return LightScanResult(
                wallet_address=wallet,
                passed=False,
                rejection_reason=f"Error: {str(e)}",
            )

    async def batch_light_scan(
        self,
        wallets: list[str],
        max_concurrent: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[list[LightScanResult], list[LightScanResult]]:
        """Batch light scan with concurrency control.

        Args:
            wallets: List of wallet addresses
            max_concurrent: Max concurrent API calls
            progress_callback: Optional callback(completed, total)

        Returns:
            Tuple of (passed_results, rejected_results)
        """
        passed = []
        rejected = []
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        total = len(wallets)

        async def scan_one(wallet: str):
            nonlocal completed
            async with semaphore:
                result = await self.light_scan(wallet)
                completed += 1
                if progress_callback:
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback(completed, total)
                    else:
                        progress_callback(completed, total)
                return result

        results = await asyncio.gather(*[scan_one(w) for w in wallets])

        for result in results:
            if result.passed:
                passed.append(result)
            else:
                rejected.append(result)

        # Sort by light score
        passed.sort(key=lambda x: x.light_score, reverse=True)
        for i, result in enumerate(passed):
            result.priority_rank = i + 1

        return passed, rejected

    # =========================================================================
    # PHASE 3: Deep Analysis (2-3 API Calls)
    # =========================================================================

    async def deep_analysis(
        self,
        wallet_address: str,
        lookback_days: int = 90,
        include_insider_signals: bool = True,
        max_trades: int = 500,
    ) -> DeepAnalysisResult:
        """Phase 3: Full deep analysis of an account.

        Fetches trade history and computes all metrics.

        Args:
            wallet_address: Wallet to analyze
            lookback_days: Days of history
            include_insider_signals: Whether to compute insider signals
            max_trades: Maximum trades to fetch per account

        Returns:
            DeepAnalysisResult with complete metrics and scoring
        """
        self._stats["phase3_processed"] += 1
        wallet = wallet_address.lower()

        try:
            # Fetch trade history (limited to max_trades)
            trades = await self._fetch_trade_history(wallet, lookback_days, max_trades)

            if not trades:
                return DeepAnalysisResult(
                    wallet_address=wallet,
                    pl_metrics=None,
                    pattern_metrics=None,
                    insider_signals=None,
                    error="No trade history found",
                )

            # Fetch current positions (1 API call)
            positions = await self._fetch_positions(wallet)

            # Calculate total unrealized P/L from open positions
            # This is critical for the off_high_watermark calculation
            total_unrealized_pnl = sum(
                float(getattr(p, 'unrealized_pnl', 0) or 0)
                for p in positions
            )

            # Compute all metrics
            pl_metrics = self._compute_pl_metrics(trades)
            # Store unrealized P/L in metrics for off_high_watermark calculation
            pl_metrics.unrealized_pnl = total_unrealized_pnl
            pattern_metrics = self._compute_pattern_metrics(trades)
            market_categories = self._compute_category_breakdown(trades)

            # Category-specific win rates
            category_win_rates = self._compute_category_win_rates(trades)

            # Insider signals
            insider_signals = None
            if include_insider_signals:
                insider_signals = self._compute_insider_signals(trades, positions)

            # Run through scoring engine
            # DEBUG: Log scoring config for first wallet analyzed
            if self._stats["phase3_processed"] <= 1:
                sc = self.scoring_engine.config
                logger.info("deep_analysis_debug_scoring_config",
                            wallet=wallet[:10],
                            min_composite_score=sc.min_composite_score,
                            total_trades=pattern_metrics.total_trades if pattern_metrics else 0)

            scoring_result = self.scoring_engine.score_account(
                pl_metrics=pl_metrics,
                pattern_metrics=pattern_metrics,
                insider_signals=insider_signals,
                category_metrics={"win_rates": category_win_rates},
            )

            # Track pass/fail
            if scoring_result.passes_threshold:
                self._stats["phase3_passed"] += 1

            # Generate output data
            pl_curve_data = self._generate_pl_curve(trades)
            recent_sample = self._get_recent_trades_sample(trades, limit=20)

            return DeepAnalysisResult(
                wallet_address=wallet,
                pl_metrics=pl_metrics,
                pattern_metrics=pattern_metrics,
                insider_signals=insider_signals,
                scoring_result=scoring_result,
                trades=[],  # Don't store trades to save memory - use recent_sample instead
                positions=[p.__dict__ if hasattr(p, '__dict__') else p for p in positions],
                pl_curve_data=pl_curve_data,
                market_categories=market_categories,
                recent_trades_sample=recent_sample,
                category_win_rates=category_win_rates,
            )

        except Exception as e:
            logger.error("deep_analysis_failed", wallet=wallet[:10], error=str(e))
            return DeepAnalysisResult(
                wallet_address=wallet,
                pl_metrics=None,
                pattern_metrics=None,
                insider_signals=None,
                error=str(e),
            )

    async def batch_deep_analysis(
        self,
        wallets: list[str],
        lookback_days: int = 90,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_trades: int = 500,
        enable_categorization: bool = False,
    ) -> list[DeepAnalysisResult]:
        """Batch deep analysis with limited concurrency.

        Args:
            wallets: Wallets to analyze
            lookback_days: Days of history
            max_concurrent: Max concurrent analyses
            progress_callback: Callback(completed, total, current_wallet)
            max_trades: Maximum trades to fetch per account
            enable_categorization: Force category detection even in wide_net mode

        Returns:
            List of DeepAnalysisResult
        """
        self._enable_categorization = enable_categorization
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        total = len(wallets)

        # DEBUG: Log scoring config at start of batch
        sc = self.scoring_engine.config
        logger.info("batch_debug_config",
                    min_composite_score=sc.min_composite_score,
                    min_trades_threshold=sc.hard_filters.get('min_trades', {}).threshold if hasattr(sc.hard_filters.get('min_trades', {}), 'threshold') else 'N/A')

        # DEBUG: Track pass/fail counts
        debug_pass_count = [0]
        debug_fail_count = [0]
        debug_no_score_count = [0]

        async def analyze_one(wallet: str):
            nonlocal completed
            async with semaphore:
                if progress_callback:
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback(completed, total, wallet[:10])
                    else:
                        progress_callback(completed, total, wallet[:10])
                result = await self.deep_analysis(wallet, lookback_days, max_trades=max_trades)
                completed += 1

                # DEBUG: Track scoring results
                if result.scoring_result:
                    if result.scoring_result.passes_threshold:
                        debug_pass_count[0] += 1
                    else:
                        debug_fail_count[0] += 1
                        # Log first few failures
                        if debug_fail_count[0] <= 3:
                            logger.warning("batch_debug_fail",
                                           wallet=wallet[:10],
                                           composite=result.scoring_result.composite_score,
                                           hard_passed=result.scoring_result.hard_filter_passed,
                                           failures=result.scoring_result.hard_filter_failures)
                else:
                    debug_no_score_count[0] += 1
                    if debug_no_score_count[0] <= 3:
                        logger.warning("batch_debug_no_score",
                                       wallet=wallet[:10],
                                       error=result.error)

                return result

        results = await asyncio.gather(*[analyze_one(w) for w in wallets])

        # DEBUG: Log final counts
        logger.info("batch_debug_final",
                    passed=debug_pass_count[0],
                    failed=debug_fail_count[0],
                    no_score=debug_no_score_count[0])

        # Sort by composite score
        results.sort(
            key=lambda x: x.scoring_result.composite_score if x.scoring_result else 0,
            reverse=True
        )

        return results

    # =========================================================================
    # Combined Multi-Phase Pipeline
    # =========================================================================

    async def run_multi_phase_scan(
        self,
        leaderboard_entries: list[LeaderboardEntry],
        max_phase2: int = 100,
        max_phase3: int = 25,
        lookback_days: int = 90,
        progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
    ) -> list[DeepAnalysisResult]:
        """Run the full multi-phase scanning pipeline.

        Args:
            leaderboard_entries: Initial candidate list
            max_phase2: Max candidates for phase 2
            max_phase3: Max candidates for phase 3
            lookback_days: Days of history for deep analysis
            progress_callback: Callback(phase, completed, passed, total)

        Returns:
            List of DeepAnalysisResult (passed candidates)
        """
        self.reset_stats()

        # PHASE 1: Quick Filter
        logger.info("phase1_starting", total=len(leaderboard_entries))
        if progress_callback:
            progress_callback("phase1", 0, 0, len(leaderboard_entries))

        phase1_passed, phase1_rejected = self.batch_quick_filter(leaderboard_entries)

        if progress_callback:
            progress_callback("phase1", len(leaderboard_entries), len(phase1_passed), len(leaderboard_entries))

        logger.info(
            "phase1_complete",
            passed=len(phase1_passed),
            rejected=len(phase1_rejected),
        )

        if not phase1_passed:
            return []

        # PHASE 2: Light Scan (top N from phase 1)
        phase2_candidates = [r.wallet_address for r in phase1_passed[:max_phase2]]

        logger.info("phase2_starting", candidates=len(phase2_candidates))
        if progress_callback:
            progress_callback("phase2", 0, 0, len(phase2_candidates))

        def phase2_progress(completed: int, total: int):
            if progress_callback:
                progress_callback("phase2", completed, 0, total)

        phase2_passed, phase2_rejected = await self.batch_light_scan(
            phase2_candidates,
            max_concurrent=5,
            progress_callback=phase2_progress,
        )

        if progress_callback:
            progress_callback("phase2", len(phase2_candidates), len(phase2_passed), len(phase2_candidates))

        logger.info(
            "phase2_complete",
            passed=len(phase2_passed),
            rejected=len(phase2_rejected),
        )

        if not phase2_passed:
            return []

        # PHASE 3: Deep Analysis (top N from phase 2)
        phase3_candidates = [r.wallet_address for r in phase2_passed[:max_phase3]]

        logger.info("phase3_starting", candidates=len(phase3_candidates))
        if progress_callback:
            progress_callback("phase3", 0, 0, len(phase3_candidates))

        def phase3_progress(completed: int, total: int, wallet: str):
            if progress_callback:
                progress_callback("phase3", completed, 0, total)

        results = await self.batch_deep_analysis(
            phase3_candidates,
            lookback_days=lookback_days,
            max_concurrent=3,
            progress_callback=phase3_progress,
        )

        # Filter to passing results
        passed_results = [r for r in results if r.scoring_result and r.scoring_result.passes_threshold]

        if progress_callback:
            progress_callback("phase3", len(phase3_candidates), len(passed_results), len(phase3_candidates))

        logger.info(
            "phase3_complete",
            analyzed=len(results),
            passed=len(passed_results),
        )

        return passed_results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _quick_categorize(self, condition_id: str) -> str:
        """Quick categorization using cache or market title."""
        if condition_id in self._market_cache:
            return self._categorize_market(self._market_cache[condition_id].question)

        # Skip if we've already failed to look up this condition
        if condition_id in self._failed_condition_ids:
            return "other"

        # Try to get from gamma (but don't fail if not available)
        try:
            market = await self._gamma_client.get_market(condition_id)
            if market:
                self._market_cache[condition_id] = market
                return self._categorize_market(market.question)
            else:
                # Market not found, cache the failure
                self._failed_condition_ids.add(condition_id)
        except Exception:
            # API error, cache the failure to avoid retries
            self._failed_condition_ids.add(condition_id)

        return "other"

    def _categorize_market(self, title: str) -> str:
        """Categorize a market based on title keywords.

        Prioritizes niche categories (economics, finance, tech, weather) over
        mainstream (politics, sports) when a market matches multiple categories.
        """
        title_lower = title.lower()

        # Find all matching categories
        matched_categories = []
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in title_lower for keyword in keywords):
                matched_categories.append(category)

        if not matched_categories:
            return "other"

        # Prefer niche categories over mainstream
        for cat in matched_categories:
            if cat in NICHE_CATEGORIES:
                return cat

        # Fall back to first mainstream match
        return matched_categories[0]

    async def _fetch_trade_history(
        self,
        wallet: str,
        lookback_days: int,
        max_trades: int = 500,
    ) -> list[TradeRecord]:
        """Fetch and enrich trade history with pagination.

        CRITICAL: Fetches ALL activity types (TRADE, REDEEM, MERGE) to ensure
        accurate P/L calculation:
        - TRADE: Buy/sell transactions
        - REDEEM: Winning positions that paid out $1/share at resolution
        - MERGE: YES+NO pairs combined for $1/share

        Missing REDEEM trades causes massive P/L underreporting since many
        profitable traders hold positions to resolution rather than selling.

        Args:
            wallet: Wallet address
            lookback_days: Days of history to fetch
            max_trades: Maximum trades to fetch (default 500)
        """
        since_timestamp = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp())

        # Paginate up to max_trades limit
        # NOTE: No activity_type filter - we need TRADE, REDEEM, and MERGE for accurate P/L
        all_activities = []
        offset = 0
        page_size = min(1000, max_trades)
        max_pages = (max_trades // page_size) + 1

        for _ in range(max_pages):
            if len(all_activities) >= max_trades:
                break
            activities = await self._data_client.get_activity(
                user=wallet,
                activity_type=None,  # Fetch ALL types - critical for accurate P/L!
                start_timestamp=since_timestamp,
                limit=page_size,
                offset=offset,
            )
            self._stats["api_calls"] += 1

            if not activities:
                break  # No more results

            all_activities.extend(activities)

            if len(all_activities) >= max_trades:
                all_activities = all_activities[:max_trades]
                break  # Hit max_trades limit

            if len(activities) < page_size:
                break  # Last page (partial)

            offset += page_size

        trades = []

        # Wide net mode: skip expensive market categorization unless explicitly enabled
        skip_categorization = (
            self.mode == DiscoveryMode.WIDE_NET_PROFITABILITY
            and not getattr(self, '_enable_categorization', False)
        )

        for activity in all_activities:
            # Only process activities that affect positions/P&L
            # TRADE = buy/sell, REDEEM = resolution payout, MERGE = combine YES+NO
            if activity.type not in [ActivityType.TRADE, ActivityType.REDEEM, ActivityType.MERGE]:
                continue

            if skip_categorization:
                # Skip API calls - just use placeholder data
                market_title = f"Market {activity.condition_id[:8]}..."
                category = "other"
            else:
                market_title = await self._get_market_title(activity.condition_id)
                category = self._categorize_market(market_title)

            # Determine side and price based on activity type
            if activity.type == ActivityType.REDEEM:
                # REDEEM = winning position paid out at $1/share
                side = TradeSide.SELL  # Treat as exit
                price = Decimal("1.0")  # Resolution payout is always $1
                usd_value = activity.size  # Size * $1
                is_resolved = True
                resolution_outcome = True  # This side won
            elif activity.type == ActivityType.MERGE:
                # MERGE = combining YES+NO for $1/share
                side = TradeSide.SELL  # Treat as exit
                price = Decimal("1.0")  # Merge payout is $1
                usd_value = activity.size  # Size * $1
                is_resolved = False  # Not a resolution, just a merge
                resolution_outcome = None
            else:
                # Regular TRADE
                side = activity.side or TradeSide.BUY
                price = activity.price
                usd_value = activity.usd_value
                is_resolved = False
                resolution_outcome = None

            trades.append(TradeRecord(
                timestamp=activity.timestamp,
                market_id=activity.condition_id,
                market_title=market_title,
                token_id=activity.token_id,
                side=side,
                size=activity.size,
                price=price,
                usd_value=usd_value,
                category=category,
                is_resolved=is_resolved,
                resolution_outcome=resolution_outcome,
            ))

        trades.sort(key=lambda t: t.timestamp)
        return trades

    async def _fetch_positions(self, wallet: str) -> list[Position]:
        """Fetch ALL current positions with pagination.

        Some accounts have 1000+ positions, so we need to paginate
        to get accurate unrealized P/L for watermark calculations.
        """
        all_positions: list[Position] = []
        offset = 0
        batch_size = 100
        max_positions = 2000  # Safety limit

        while len(all_positions) < max_positions:
            self._stats["api_calls"] += 1
            batch = await self._data_client.get_positions(
                user=wallet,
                limit=batch_size,
                offset=offset
            )

            if not batch:
                break

            all_positions.extend(batch)

            if len(batch) < batch_size:
                # No more positions
                break

            offset += batch_size

        return all_positions

    async def _get_market_title(self, condition_id: str) -> str:
        """Get market title, using cache."""
        if condition_id in self._market_cache:
            return self._market_cache[condition_id].question

        # Skip if we've already failed to look up this condition
        if condition_id in self._failed_condition_ids:
            return f"Market {condition_id[:8]}..."

        try:
            market = await self._gamma_client.get_market(condition_id)
            if market:
                self._market_cache[condition_id] = market
                return market.question
            else:
                self._failed_condition_ids.add(condition_id)
        except Exception:
            self._failed_condition_ids.add(condition_id)

        return "Unknown Market"

    def _compute_pl_metrics(self, trades: list[TradeRecord]) -> PLCurveMetrics:
        """Compute P/L curve metrics from trade history."""
        if not trades:
            return self._empty_pl_metrics()

        daily_pnl: dict[str, Decimal] = defaultdict(Decimal)
        cumulative_pnl = Decimal("0")
        pnl_series: list[float] = []
        wins: list[Decimal] = []
        losses: list[Decimal] = []

        positions: dict[str, dict] = {}

        for trade in trades:
            date_key = trade.timestamp.strftime("%Y-%m-%d")

            if trade.side == TradeSide.BUY:
                if trade.token_id not in positions:
                    positions[trade.token_id] = {"size": Decimal("0"), "cost": Decimal("0")}
                positions[trade.token_id]["size"] += trade.size
                positions[trade.token_id]["cost"] += trade.usd_value
            else:
                if trade.token_id in positions:
                    pos = positions[trade.token_id]
                    if pos["size"] > 0:
                        avg_cost = pos["cost"] / pos["size"]
                        realized = trade.usd_value - (avg_cost * min(trade.size, pos["size"]))
                        daily_pnl[date_key] += realized
                        cumulative_pnl += realized

                        if realized > 0:
                            wins.append(realized)
                        elif realized < 0:
                            losses.append(abs(realized))

                        pos["size"] -= trade.size
                        if pos["size"] <= 0:
                            del positions[trade.token_id]

            pnl_series.append(float(cumulative_pnl))

        # Calculate metrics
        if len(pnl_series) < 2:
            returns = [0.0]
        else:
            returns = [pnl_series[i] - pnl_series[i-1] for i in range(1, len(pnl_series))]

        # Sharpe ratio
        if returns and len(returns) > 1:
            std = statistics.stdev(returns)
            sharpe = statistics.mean(returns) / std if std > 0 else 0.0
        else:
            sharpe = 0.0

        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and len(negative_returns) > 1:
            down_std = statistics.stdev(negative_returns)
            sortino = statistics.mean(returns) / down_std if down_std > 0 else sharpe
        else:
            sortino = sharpe

        # Drawdown
        peak = Decimal("0")
        max_drawdown = Decimal("0")
        drawdowns = []

        for pnl in pnl_series:
            pnl_dec = Decimal(str(pnl))
            if pnl_dec > peak:
                peak = pnl_dec
            elif peak > 0:
                dd = (peak - pnl_dec) / peak
                # Cap drawdown at 100% - can't lose more than 100% from peak
                dd = min(dd, Decimal("1.0"))
                if dd > max_drawdown:
                    max_drawdown = dd
                drawdowns.append(float(dd))

        avg_drawdown = statistics.mean(drawdowns) if drawdowns else 0.0

        # Calmar ratio
        calmar = (float(cumulative_pnl) / float(max_drawdown)) if max_drawdown > 0 else 0.0

        # Win/loss stats
        total_wins = sum(wins)
        total_losses = sum(losses)
        win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.0
        avg_win = total_wins / len(wins) if wins else Decimal("0")
        avg_loss = total_losses / len(losses) if losses else Decimal("0")
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else float("inf")

        # Concentration (extended for copytrade viability analysis)
        sorted_wins = sorted(wins, reverse=True)
        largest_win_pct = float(sorted_wins[0] / cumulative_pnl) if sorted_wins and cumulative_pnl > 0 else 0.0
        top_3_pct = float(sum(sorted_wins[:3]) / cumulative_pnl) if len(sorted_wins) >= 3 and cumulative_pnl > 0 else largest_win_pct
        top_5_pct = float(sum(sorted_wins[:5]) / cumulative_pnl) if len(sorted_wins) >= 5 and cumulative_pnl > 0 else top_3_pct
        top_10_pct = float(sum(sorted_wins[:10]) / cumulative_pnl) if len(sorted_wins) >= 10 and cumulative_pnl > 0 else top_5_pct

        # Market-level P/L analysis for copytrade viability
        # Track P/L per market to compute market win rate
        market_pnl: dict[str, Decimal] = defaultdict(Decimal)
        redemption_profit = Decimal("0")
        sell_profit = Decimal("0")
        redemption_wins = 0
        sell_wins = 0

        # Recompute per-market P/L
        market_positions: dict[str, dict] = {}
        for trade in trades:
            market_id = trade.market_id
            if trade.side == TradeSide.BUY:
                if market_id not in market_positions:
                    market_positions[market_id] = {"size": Decimal("0"), "cost": Decimal("0")}
                market_positions[market_id]["size"] += trade.size
                market_positions[market_id]["cost"] += trade.usd_value
            else:  # SELL (includes REDEEM and MERGE)
                if market_id in market_positions:
                    pos = market_positions[market_id]
                    if pos["size"] > 0:
                        avg_cost = pos["cost"] / pos["size"]
                        realized = trade.usd_value - (avg_cost * min(trade.size, pos["size"]))
                        market_pnl[market_id] += realized
                        # Track if this was a redemption (resolution) vs regular sell
                        if trade.is_resolved:
                            if realized > 0:
                                redemption_profit += realized
                                redemption_wins += 1
                        else:
                            if realized > 0:
                                sell_profit += realized
                                sell_wins += 1
                        pos["size"] -= trade.size
                        if pos["size"] <= 0:
                            del market_positions[market_id]

        # Calculate market-level win rate
        markets_profitable = sum(1 for pnl in market_pnl.values() if pnl > 0)
        markets_unprofitable = sum(1 for pnl in market_pnl.values() if pnl < 0)
        market_win_rate = markets_profitable / (markets_profitable + markets_unprofitable) if (markets_profitable + markets_unprofitable) > 0 else 0.0

        # Redemption percentage of profit
        redemption_pct = float(redemption_profit / total_wins) if total_wins > 0 else 0.0

        # Simulated capture rate analysis
        # This estimates what % of profit you'd get if you randomly captured X% of trades
        import random
        market_pnl_list = list(market_pnl.values())
        sim_50_results = []
        sim_25_results = []

        if len(market_pnl_list) >= 4:
            for _ in range(100):  # 100 simulations
                # 50% capture
                captured_50 = random.sample(market_pnl_list, len(market_pnl_list) // 2)
                sim_50_results.append(float(sum(captured_50)))
                # 25% capture
                captured_25 = random.sample(market_pnl_list, len(market_pnl_list) // 4)
                sim_25_results.append(float(sum(captured_25)))

            sim_50_results.sort()
            sim_25_results.sort()
            sim_50_median = sim_50_results[len(sim_50_results) // 2]
            sim_25_median = sim_25_results[len(sim_25_results) // 2]
        else:
            sim_50_median = float(cumulative_pnl) * 0.5
            sim_25_median = float(cumulative_pnl) * 0.25

        return PLCurveMetrics(
            total_realized_pnl=cumulative_pnl,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=float(max_drawdown),
            avg_drawdown_pct=avg_drawdown,
            max_drawdown_duration_days=0,
            win_rate=win_rate,
            avg_win_size=avg_win,
            avg_loss_size=avg_loss,
            profit_factor=min(profit_factor, 999),
            longest_win_streak=0,
            longest_loss_streak=0,
            current_streak=0,
            largest_win_pct_of_total=largest_win_pct,
            top_3_wins_pct_of_total=top_3_pct,
            avg_recovery_time_days=0.0,
            # Win/loss counts and totals
            win_count=len(wins),
            loss_count=len(losses),
            gross_profit=total_wins,
            gross_loss=total_losses,
            # Copytrade viability metrics
            top_5_wins_pct_of_total=top_5_pct,
            top_10_wins_pct_of_total=top_10_pct,
            market_win_rate=market_win_rate,
            markets_profitable=markets_profitable,
            markets_unprofitable=markets_unprofitable,
            redemption_win_count=redemption_wins,
            sell_win_count=sell_wins,
            redemption_pct_of_profit=redemption_pct,
            simulated_50pct_capture_median=sim_50_median,
            simulated_25pct_capture_median=sim_25_median,
        )

    def _compute_pattern_metrics(self, trades: list[TradeRecord]) -> TradingPatternMetrics:
        """Compute trading pattern metrics."""
        if not trades:
            return self._empty_pattern_metrics()

        sizes = [float(t.usd_value) for t in trades]
        avg_size = Decimal(str(statistics.mean(sizes)))
        median_size = Decimal(str(statistics.median(sizes)))
        max_size = Decimal(str(max(sizes)))
        size_std = Decimal(str(statistics.stdev(sizes))) if len(sizes) > 1 else Decimal("0")

        prices = [float(t.price) for t in trades if t.price > 0]
        pct_under_5c = len([p for p in prices if p < 0.05]) / len(prices) if prices else 0.0
        pct_under_10c = len([p for p in prices if p < 0.10]) / len(prices) if prices else 0.0
        pct_under_20c = len([p for p in prices if p < 0.20]) / len(prices) if prices else 0.0
        pct_over_80c = len([p for p in prices if p > 0.80]) / len(prices) if prices else 0.0
        avg_odds = Decimal(str(statistics.mean(prices))) if prices else Decimal("0.5")
        median_odds = Decimal(str(statistics.median(prices))) if prices else Decimal("0.5")

        total_trades = len(trades)
        # Use min/max to correctly identify first/last trade regardless of API sort order
        timestamps = [t.timestamp for t in trades]
        first_trade = min(timestamps)
        last_trade = max(timestamps)
        days_span = max(1, (last_trade - first_trade).days)
        trades_per_day = total_trades / days_span
        account_age = (datetime.utcnow() - first_trade).days

        unique_markets = len(set(t.market_id for t in trades))
        markets_per_trade = unique_markets / total_trades if total_trades > 0 else 0.0

        category_counts: dict[str, int] = defaultdict(int)
        for t in trades:
            category_counts[t.category] += 1

        category_breakdown = {
            cat: count / total_trades
            for cat, count in category_counts.items()
        } if total_trades > 0 else {}

        niche_count = sum(count for cat, count in category_counts.items() if cat in NICHE_CATEGORIES)
        niche_pct = niche_count / total_trades if total_trades > 0 else 0.0

        # Days since last trade (activity recency)
        days_since_last = (datetime.utcnow() - last_trade).days

        # Buy/sell ratio
        buy_count = sum(1 for t in trades if t.side == TradeSide.BUY)
        buy_ratio = buy_count / total_trades if total_trades > 0 else 0.5

        # Primary category classification
        primary_cat, cat_concentration = classify_trader_category(category_breakdown)

        return TradingPatternMetrics(
            avg_position_size_usd=avg_size,
            median_position_size_usd=median_size,
            max_position_size_usd=max_size,
            position_size_std_dev=size_std,
            pct_trades_under_5c=pct_under_5c,
            pct_trades_under_10c=pct_under_10c,
            pct_trades_under_20c=pct_under_20c,
            pct_trades_over_80c=pct_over_80c,
            avg_entry_odds=avg_odds,
            median_entry_odds=median_odds,
            total_trades=total_trades,
            trades_per_day_avg=trades_per_day,
            active_days=days_span,
            account_age_days=account_age,
            unique_markets_traded=unique_markets,
            markets_per_trade_ratio=markets_per_trade,
            category_breakdown=category_breakdown,
            niche_market_pct=niche_pct,
            avg_hold_time_hours=0.0,
            pct_trades_near_expiry=0.0,
            days_since_last_trade=days_since_last,
            buy_sell_ratio=buy_ratio,
            primary_category=primary_cat,
            category_concentration=cat_concentration,
        )

    def _compute_category_win_rates(self, trades: list[TradeRecord]) -> dict[str, float]:
        """Compute win rates by category."""
        category_trades: dict[str, list[Decimal]] = defaultdict(list)
        positions: dict[str, dict] = {}

        for trade in trades:
            if trade.side == TradeSide.BUY:
                if trade.token_id not in positions:
                    positions[trade.token_id] = {
                        "size": Decimal("0"),
                        "cost": Decimal("0"),
                        "category": trade.category,
                    }
                positions[trade.token_id]["size"] += trade.size
                positions[trade.token_id]["cost"] += trade.usd_value
            else:
                if trade.token_id in positions:
                    pos = positions[trade.token_id]
                    if pos["size"] > 0:
                        avg_cost = pos["cost"] / pos["size"]
                        realized = trade.usd_value - (avg_cost * min(trade.size, pos["size"]))
                        category_trades[pos["category"]].append(realized)
                        pos["size"] -= trade.size
                        if pos["size"] <= 0:
                            del positions[trade.token_id]

        win_rates = {}
        for cat, pnls in category_trades.items():
            wins = len([p for p in pnls if p > 0])
            total = len(pnls)
            win_rates[cat] = wins / total if total > 0 else 0.0

        return win_rates

    def _compute_insider_signals(
        self,
        trades: list[TradeRecord],
        positions: list,
    ) -> InsiderSignals:
        """Compute insider detection signals."""
        if not trades:
            return self._empty_insider_signals()

        # Use min to correctly identify first trade regardless of API sort order
        first_trade = min(t.timestamp for t in trades)
        account_age = (datetime.utcnow() - first_trade).days

        large_unlikely_bets = []
        for t in trades:
            if float(t.usd_value) > 1000 and float(t.price) < 0.10:
                large_unlikely_bets.append({
                    "market": t.market_title[:50],
                    "size_usd": str(t.usd_value),
                    "odds": str(t.price),
                    "date": t.timestamp.isoformat(),
                })

        fresh_large = False
        if account_age < 30:
            max_size = max(float(t.usd_value) for t in trades)
            if max_size > 5000:
                fresh_large = True

        market_values: dict[str, Decimal] = defaultdict(Decimal)
        for t in trades:
            if t.side == TradeSide.BUY:
                market_values[t.market_id] += t.usd_value

        total_value = sum(market_values.values())
        max_concentration = 0.0
        if total_value > 0:
            for value in market_values.values():
                conc = float(value / total_value)
                if conc > max_concentration:
                    max_concentration = conc

        sizes = [(float(t.usd_value), float(t.price)) for t in trades]
        largest_size = max(s[0] for s in sizes) if sizes else 0
        largest_odds = next((s[1] for s in sizes if s[0] == largest_size), 0.5)

        insider_score = 0.0
        if account_age < 14 and largest_size > 10000:
            insider_score += 0.4
        if large_unlikely_bets:
            insider_score += min(0.3, len(large_unlikely_bets) * 0.1)
        if max_concentration > 0.8:
            insider_score += 0.2

        return InsiderSignals(
            account_age_days=account_age,
            first_trade_date=first_trade,
            large_bets_on_unlikely_events=large_unlikely_bets[:10],
            pre_resolution_accumulation=[],
            fresh_account_large_positions=fresh_large,
            single_market_concentration=max_concentration,
            largest_position_usd=Decimal(str(largest_size)),
            largest_position_odds_at_entry=Decimal(str(largest_odds)),
            trades_in_final_24h=0,
            large_trades_in_final_24h=[],
            insider_probability_score=min(1.0, insider_score),
        )

    def _compute_category_breakdown(self, trades: list[TradeRecord]) -> dict[str, float]:
        """Compute category breakdown from trades."""
        if not trades:
            return {}

        category_counts: dict[str, int] = defaultdict(int)
        for t in trades:
            category_counts[t.category] += 1

        total = len(trades)
        return {cat: count / total for cat, count in category_counts.items()}

    def _generate_pl_curve(self, trades: list[TradeRecord]) -> list[dict]:
        """Generate P/L curve data points for charting."""
        if not trades:
            return []

        cumulative = Decimal("0")
        curve_data = []
        positions: dict[str, Decimal] = defaultdict(Decimal)

        for trade in trades:
            if trade.side == TradeSide.BUY:
                positions[trade.token_id] += trade.usd_value
            else:
                if trade.token_id in positions:
                    cost_basis = positions[trade.token_id]
                    cumulative += trade.usd_value - cost_basis
                    positions[trade.token_id] = Decimal("0")

            curve_data.append({
                "date": trade.timestamp.isoformat(),
                "cumulative_pnl": str(cumulative),
            })

        return curve_data

    def _get_recent_trades_sample(
        self,
        trades: list[TradeRecord],
        limit: int = 20,
    ) -> list[dict]:
        """Get recent trades sample for display."""
        recent = sorted(trades, key=lambda t: t.timestamp, reverse=True)[:limit]

        return [
            {
                "date": t.timestamp.isoformat(),
                "market": t.market_title[:60],
                "side": t.side.value,
                "size_usd": str(t.usd_value),
                "odds": str(t.price),
                "category": t.category,
            }
            for t in recent
        ]

    def _empty_pl_metrics(self) -> PLCurveMetrics:
        """Return empty P/L metrics."""
        return PLCurveMetrics(
            total_realized_pnl=Decimal("0"),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            avg_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            win_rate=0.0,
            avg_win_size=Decimal("0"),
            avg_loss_size=Decimal("0"),
            profit_factor=0.0,
            longest_win_streak=0,
            longest_loss_streak=0,
            current_streak=0,
            largest_win_pct_of_total=0.0,
            top_3_wins_pct_of_total=0.0,
            avg_recovery_time_days=0.0,
            win_count=0,
            loss_count=0,
            gross_profit=Decimal("0"),
            gross_loss=Decimal("0"),
        )

    def _empty_pattern_metrics(self) -> TradingPatternMetrics:
        """Return empty pattern metrics."""
        return TradingPatternMetrics(
            avg_position_size_usd=Decimal("0"),
            median_position_size_usd=Decimal("0"),
            max_position_size_usd=Decimal("0"),
            position_size_std_dev=Decimal("0"),
            pct_trades_under_5c=0.0,
            pct_trades_under_10c=0.0,
            pct_trades_under_20c=0.0,
            pct_trades_over_80c=0.0,
            avg_entry_odds=Decimal("0.5"),
            median_entry_odds=Decimal("0.5"),
            total_trades=0,
            trades_per_day_avg=0.0,
            active_days=0,
            account_age_days=0,
            unique_markets_traded=0,
            markets_per_trade_ratio=0.0,
            category_breakdown={},
            niche_market_pct=0.0,
            avg_hold_time_hours=0.0,
            pct_trades_near_expiry=0.0,
            days_since_last_trade=999,
            buy_sell_ratio=0.5,
            primary_category="Diversified",
            category_concentration=0.0,
        )

    def _empty_insider_signals(self) -> InsiderSignals:
        """Return empty insider signals."""
        return InsiderSignals(
            account_age_days=0,
            first_trade_date=None,
            large_bets_on_unlikely_events=[],
            pre_resolution_accumulation=[],
            fresh_account_large_positions=False,
            single_market_concentration=0.0,
            largest_position_usd=Decimal("0"),
            largest_position_odds_at_entry=Decimal("0.5"),
            trades_in_final_24h=0,
            large_trades_in_final_24h=[],
            insider_probability_score=0.0,
        )
