"""Deep Scoring Engine for Account Discovery.

Implements a multi-layered scoring system with:
- Hard filters (absolute disqualifiers)
- Soft signals (weighted scoring)
- Mode-specific compound criteria
- UI-editable thresholds

Based on research from PolyTrack, Unusual Whales, Nansen, and academic studies.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Any
from enum import Enum
import math

from .models import (
    DiscoveryMode,
    RedFlag,
    RedFlagType,
    PLCurveMetrics,
    TradingPatternMetrics,
    InsiderSignals,
)


class FilterResult(str, Enum):
    """Result of applying a filter."""
    PASS = "pass"
    SOFT_FAIL = "soft_fail"  # Contributes negative score but doesn't reject
    HARD_FAIL = "hard_fail"  # Absolute rejection


@dataclass
class FilterConfig:
    """Configuration for a single filter criterion."""
    name: str
    description: str
    enabled: bool = True
    is_hard_filter: bool = False  # Hard = auto-reject, Soft = score impact

    # Threshold values (UI-editable)
    threshold: float = 0.0
    threshold_min: float = 0.0  # UI slider min
    threshold_max: float = 100.0  # UI slider max

    # Scoring impact (for soft filters)
    weight: float = 1.0  # How much this affects score
    pass_bonus: float = 0.0  # Bonus points if passed
    fail_penalty: float = 0.0  # Penalty if failed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "is_hard_filter": self.is_hard_filter,
            "threshold": self.threshold,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "weight": self.weight,
            "pass_bonus": self.pass_bonus,
            "fail_penalty": self.fail_penalty,
        }


@dataclass
class ModeConfig:
    """Complete configuration for a discovery mode."""
    mode: DiscoveryMode
    name: str
    description: str

    # Hard filters - absolute disqualifiers
    hard_filters: dict[str, FilterConfig] = field(default_factory=dict)

    # Soft filters - contribute to score
    soft_filters: dict[str, FilterConfig] = field(default_factory=dict)

    # Scoring weights for different components
    pl_consistency_weight: float = 0.25
    pattern_match_weight: float = 0.25
    specialization_weight: float = 0.25
    risk_weight: float = 0.25

    # Final thresholds
    min_composite_score: float = 50.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "name": self.name,
            "description": self.description,
            "hard_filters": {k: v.to_dict() for k, v in self.hard_filters.items()},
            "soft_filters": {k: v.to_dict() for k, v in self.soft_filters.items()},
            "pl_consistency_weight": self.pl_consistency_weight,
            "pattern_match_weight": self.pattern_match_weight,
            "specialization_weight": self.specialization_weight,
            "risk_weight": self.risk_weight,
            "min_composite_score": self.min_composite_score,
        }


def create_micro_bet_hunter_config() -> ModeConfig:
    """Create configuration for Micro-Bet Hunter mode.

    Finds accounts making many small bets at long odds with demonstrated edge.
    """
    return ModeConfig(
        mode=DiscoveryMode.MICRO_BET_HUNTER,
        name="Micro-Bet Hunter",
        description="Find accounts with proven edge making small bets at long odds",

        hard_filters={
            # Absolute requirements
            "min_trades": FilterConfig(
                name="Minimum Trades",
                description="Statistical significance requires sufficient sample size",
                is_hard_filter=True,
                threshold=100,
                threshold_min=50,
                threshold_max=500,
            ),
            "min_account_age": FilterConfig(
                name="Minimum Account Age (days)",
                description="Exclude flash-in-pan accounts",
                is_hard_filter=True,
                threshold=45,
                threshold_min=14,
                threshold_max=180,
            ),
            "must_be_profitable": FilterConfig(
                name="Must Be Profitable",
                description="Total P/L must be positive",
                is_hard_filter=True,
                threshold=0,  # P/L > 0
            ),
            "max_single_win_pct": FilterConfig(
                name="Max Single Win % of P/L",
                description="Reject if one lucky win dominates returns",
                is_hard_filter=True,
                threshold=50,  # No single win > 50% of total P/L
                threshold_min=30,
                threshold_max=70,
            ),
            "max_avg_position": FilterConfig(
                name="Max Average Position ($)",
                description="Must be small-bet focused, not whale",
                is_hard_filter=True,
                threshold=100,
                threshold_min=25,
                threshold_max=500,
            ),
        },

        soft_filters={
            # Position sizing (scored, not hard reject)
            "ideal_avg_position": FilterConfig(
                name="Ideal Avg Position ($)",
                description="Target average position size",
                threshold=15,
                threshold_min=2,
                threshold_max=50,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=5,
            ),
            "position_consistency": FilterConfig(
                name="Position Size Consistency",
                description="Std dev / mean of position sizes (lower = more consistent)",
                threshold=1.5,  # Std dev should be < 1.5x mean
                threshold_min=0.5,
                threshold_max=3.0,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=5,
            ),

            # Odds preferences
            "pct_under_10c": FilterConfig(
                name="% Trades Under 10c Odds",
                description="Percentage of trades at long odds (<10 cents)",
                threshold=50,  # Want >50% at long odds
                threshold_min=20,
                threshold_max=80,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),
            "pct_under_5c": FilterConfig(
                name="% Trades Under 5c Odds",
                description="Percentage at extreme long odds (<5 cents)",
                threshold=20,
                threshold_min=5,
                threshold_max=50,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=0,  # Not penalized for missing this
            ),

            # Edge verification
            "win_rate_at_long_odds": FilterConfig(
                name="Win Rate at <10c Odds (%)",
                description="Must beat expected ~5-8% to show edge",
                threshold=8,  # Win rate > 8% at <10c odds
                threshold_min=5,
                threshold_max=20,
                weight=2.5,
                pass_bonus=25,
                fail_penalty=15,
            ),
            "profit_factor": FilterConfig(
                name="Profit Factor",
                description="Gross profits / gross losses (>1.5 = good)",
                threshold=1.5,
                threshold_min=1.0,
                threshold_max=5.0,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),

            # Consistency
            "monthly_consistency": FilterConfig(
                name="Monthly Profit Consistency (%)",
                description="% of months profitable",
                threshold=60,
                threshold_min=40,
                threshold_max=90,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=10,
            ),
            "max_drawdown": FilterConfig(
                name="Max Drawdown (%)",
                description="Maximum peak-to-trough decline",
                threshold=30,  # Max 30% drawdown
                threshold_min=10,
                threshold_max=50,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=15,
            ),

            # Diversification (within category)
            "unique_markets": FilterConfig(
                name="Unique Markets Traded",
                description="Number of different markets (not concentrated)",
                threshold=30,
                threshold_min=10,
                threshold_max=100,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=5,
            ),
            "max_single_market_pct": FilterConfig(
                name="Max Single Market % of Trades",
                description="No one exact bet should dominate",
                threshold=15,  # No single condition_id > 15%
                threshold_min=5,
                threshold_max=30,
                weight=1.0,
                pass_bonus=5,
                fail_penalty=10,
            ),

            # Activity
            "trades_per_week": FilterConfig(
                name="Trades Per Week (avg)",
                description="Active trading frequency",
                threshold=5,
                threshold_min=1,
                threshold_max=30,
                weight=0.5,
                pass_bonus=5,
                fail_penalty=0,
            ),
            "recent_activity": FilterConfig(
                name="Days Since Last Trade",
                description="Must be recently active",
                threshold=14,  # Active within 14 days
                threshold_min=7,
                threshold_max=60,
                weight=0.5,
                pass_bonus=5,
                fail_penalty=10,
            ),
        },

        pl_consistency_weight=0.20,
        pattern_match_weight=0.35,  # High weight - pattern is key
        specialization_weight=0.20,
        risk_weight=0.25,
        min_composite_score=55,
    )


def create_niche_specialist_config() -> ModeConfig:
    """Create configuration for Niche Specialist mode.

    Finds accounts with demonstrated expertise in specific market categories.
    """
    return ModeConfig(
        mode=DiscoveryMode.NICHE_SPECIALIST,
        name="Niche Specialist",
        description="Find accounts with proven edge in specific categories (weather, economics, etc.)",

        hard_filters={
            "min_trades": FilterConfig(
                name="Minimum Trades",
                description="Need enough data to verify specialization",
                is_hard_filter=True,
                threshold=10,  # Very low - niche traders have less volume
                threshold_min=5,
                threshold_max=300,
            ),
            "min_account_age": FilterConfig(
                name="Minimum Account Age (days)",
                description="Account must have some history",
                is_hard_filter=True,
                threshold=7,  # Lowered significantly - accounts may have sparse recent activity
                threshold_min=1,
                threshold_max=180,
            ),
            "must_be_profitable": FilterConfig(
                name="Minimum P/L (allow losses)",
                description="P/L threshold (negative = allow losses)",
                is_hard_filter=True,
                threshold=-100000,  # Allow up to $100k losses - we rank by score, not P/L
                threshold_min=-500000,
                threshold_max=10000,
            ),
            "max_mainstream_pct": FilterConfig(
                name="Max Mainstream Category %",
                description="Politics + Sports should not dominate",
                is_hard_filter=True,
                threshold=100,  # Disabled - most Polymarket activity is politics anyway
                threshold_min=10,
                threshold_max=100,
            ),
            # Note: max_single_win_pct moved to soft_filters
        },

        soft_filters={
            # Win concentration (soft penalty, not hard reject)
            "max_single_win_pct": FilterConfig(
                name="Max Single Win % of P/L",
                description="Expertise = consistent wins, not one lucky bet",
                threshold=50,  # Penalty starts at 50%
                threshold_min=20,
                threshold_max=100,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=15,
            ),
            # Specialization depth
            "category_concentration": FilterConfig(
                name="Primary Category Concentration %",
                description="% of trades in top 1-2 categories",
                threshold=60,  # Want >60% in specialty
                threshold_min=40,
                threshold_max=90,
                weight=2.5,
                pass_bonus=25,
                fail_penalty=10,
            ),
            "specialty_win_rate": FilterConfig(
                name="Specialty Category Win Rate %",
                description="Win rate in primary category",
                threshold=58,  # Must beat 55% baseline
                threshold_min=50,
                threshold_max=80,
                weight=2.5,
                pass_bonus=25,
                fail_penalty=15,
            ),
            "specialty_vs_overall": FilterConfig(
                name="Specialty Win Rate vs Overall",
                description="Category win rate should exceed overall (% points)",
                threshold=3,  # At least 3% better in specialty
                threshold_min=0,
                threshold_max=15,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=5,
            ),

            # Niche market focus
            "niche_category_pct": FilterConfig(
                name="Niche Categories %",
                description="% in weather, economics, tech, finance",
                threshold=50,
                threshold_min=30,
                threshold_max=90,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),

            # Expertise indicators
            "early_entry_pct": FilterConfig(
                name="Early Market Entry %",
                description="% of trades in first 24h of market",
                threshold=30,  # Good specialists enter early
                threshold_min=10,
                threshold_max=60,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=0,
            ),
            "contrarian_win_rate": FilterConfig(
                name="Contrarian Win Rate %",
                description="Win rate when betting against market",
                threshold=55,
                threshold_min=45,
                threshold_max=75,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=0,
            ),

            # Consistency
            "profit_factor": FilterConfig(
                name="Profit Factor",
                description="Gross profits / gross losses",
                threshold=1.8,
                threshold_min=1.2,
                threshold_max=5.0,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=10,
            ),
            "sharpe_ratio": FilterConfig(
                name="Sharpe Ratio (approx)",
                description="Risk-adjusted returns",
                threshold=1.0,
                threshold_min=0.3,
                threshold_max=3.0,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=10,
            ),
            "monthly_consistency": FilterConfig(
                name="Monthly Profit Consistency %",
                description="% of months profitable",
                threshold=65,
                threshold_min=50,
                threshold_max=90,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=10,
            ),

            # Volume and activity
            "min_category_trades": FilterConfig(
                name="Trades in Primary Category",
                description="Enough trades to verify specialty edge",
                threshold=40,
                threshold_min=20,
                threshold_max=150,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=15,
            ),
        },

        pl_consistency_weight=0.25,
        pattern_match_weight=0.20,
        specialization_weight=0.35,  # High weight - specialization is key
        risk_weight=0.20,
        min_composite_score=25,  # Very low - most Polymarket traders are politics-focused
    )


def create_insider_detection_config() -> ModeConfig:
    """Create configuration for Insider Detection mode.

    Finds accounts with suspicious patterns suggesting privileged information.
    Uses ADDITIVE scoring - more signals = higher suspicion score.
    """
    return ModeConfig(
        mode=DiscoveryMode.INSIDER_DETECTION,
        name="Insider Detection",
        description="Find accounts with suspicious patterns (fresh accounts, large bets on unlikely events)",

        # Note: Hard filters are INVERTED for insider mode
        # We WANT new accounts, large positions, etc.
        hard_filters={
            "max_account_age": FilterConfig(
                name="Max Account Age (days)",
                description="Focus on newer accounts (more suspicious)",
                is_hard_filter=True,
                threshold=180,  # Only accounts < 6 months
                threshold_min=30,
                threshold_max=365,
            ),
            "min_position_size": FilterConfig(
                name="Min Largest Position ($)",
                description="Must have at least one notable position",
                is_hard_filter=True,
                threshold=500,
                threshold_min=100,
                threshold_max=10000,
            ),
        },

        soft_filters={
            # Account age signals (younger = more suspicious)
            "very_fresh_account": FilterConfig(
                name="Very Fresh Account (<14 days)",
                description="Brand new account (highly suspicious if large bets)",
                threshold=14,
                weight=1.0,
                pass_bonus=25,  # Bonus if account IS very fresh
                fail_penalty=0,
            ),
            "fresh_account": FilterConfig(
                name="Fresh Account (<30 days)",
                description="New account",
                threshold=30,
                weight=1.0,
                pass_bonus=15,
                fail_penalty=0,
            ),

            # Position size signals
            "large_first_bet": FilterConfig(
                name="Large First Bet ($)",
                description="First trade size (large = suspicious)",
                threshold=1000,
                threshold_min=100,
                threshold_max=10000,
                weight=1.5,
                pass_bonus=20,
                fail_penalty=0,
            ),
            "max_position_size": FilterConfig(
                name="Max Position Size ($)",
                description="Largest single position",
                threshold=5000,
                threshold_min=1000,
                threshold_max=50000,
                weight=1.5,
                pass_bonus=20,
                fail_penalty=0,
            ),

            # Odds + timing signals
            "bets_on_unlikely_events": FilterConfig(
                name="Large Bets on <10% Odds",
                description="Number of $1k+ bets at <10c odds",
                threshold=1,  # Even 1 is notable
                threshold_min=0,
                threshold_max=10,
                weight=2.0,
                pass_bonus=30,
                fail_penalty=0,
            ),
            "near_expiry_bets": FilterConfig(
                name="Large Bets Within 24h of Resolution",
                description="Betting big right before market closes",
                threshold=1,
                threshold_min=0,
                threshold_max=10,
                weight=2.5,
                pass_bonus=35,
                fail_penalty=0,
            ),
            "near_expiry_window_hours": FilterConfig(
                name="Near-Expiry Window (hours)",
                description="Define 'near expiry' - can be 24-72h",
                threshold=48,  # 48 hours before resolution
                threshold_min=12,
                threshold_max=72,
                weight=0,  # Config only, doesn't add to score
            ),

            # Concentration signals
            "single_market_concentration": FilterConfig(
                name="Single Market Concentration %",
                description="% of portfolio in one market (high = suspicious)",
                threshold=70,
                threshold_min=50,
                threshold_max=100,
                weight=1.5,
                pass_bonus=20,
                fail_penalty=0,
            ),

            # Win pattern signals
            "implausible_win_rate": FilterConfig(
                name="Implausible Win Rate %",
                description="Win rate that's too good (>80% suspicious)",
                threshold=80,
                threshold_min=70,
                threshold_max=100,
                weight=2.0,
                pass_bonus=25,
                fail_penalty=0,
            ),
            "perfect_timing_count": FilterConfig(
                name="Perfect Timing Events",
                description="Times where position was opened right before favorable news",
                threshold=2,
                threshold_min=1,
                threshold_max=10,
                weight=2.5,
                pass_bonus=35,
                fail_penalty=0,
            ),

            # Cluster signals
            "cluster_funding_match": FilterConfig(
                name="Cluster Funding Match",
                description="Funded from same source as other suspicious wallets",
                threshold=1,  # Boolean essentially
                weight=2.0,
                pass_bonus=30,
                fail_penalty=0,
            ),
            "synchronized_trading": FilterConfig(
                name="Synchronized Trading Pattern",
                description="Trades within minutes of related wallets",
                threshold=1,
                weight=2.0,
                pass_bonus=30,
                fail_penalty=0,
            ),
        },

        # For insider mode, weights are different
        pl_consistency_weight=0.10,  # Don't care much about consistency
        pattern_match_weight=0.20,
        specialization_weight=0.10,
        risk_weight=0.60,  # "Risk" here means suspicion signals
        min_composite_score=40,  # Lower threshold to catch more
    )


def create_wide_net_profitability_config() -> ModeConfig:
    """Create configuration for Wide Net Profitability mode.

    Designed to cast the widest possible net to find any consistently
    profitable accounts. Minimal hard filters, focus on profitability metrics.

    Philosophy: "0.04% of accounts make 70% of profits" - collect broadly, sort later.
    """
    return ModeConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        name="Wide Net (Profitability Focus)",
        description="Cast wide net for any profitable accounts, minimal filtering, sort by score later",

        hard_filters={
            # Only the most basic sanity checks
            "min_trades": FilterConfig(
                name="Minimum Trades",
                description="Need minimal sample for any signal",
                is_hard_filter=True,
                threshold=5,  # Very low - just need some activity
                threshold_min=1,
                threshold_max=100,
            ),
            "min_total_pnl": FilterConfig(
                name="Minimum Total P/L ($)",
                description="Must have made at least some profit",
                is_hard_filter=False,  # Disabled: leaderboard already filters by P/L
                enabled=False,  # Computed P/L from 90-day history is unreliable
                threshold=100,
                threshold_min=-10000,
                threshold_max=10000,
            ),
        },

        soft_filters={
            # Pure profitability metrics - no category gatekeeping
            "total_pnl": FilterConfig(
                name="Total P/L ($)",
                description="Higher total profit = better",
                threshold=5000,
                threshold_min=100,
                threshold_max=100000,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=0,  # No penalty for lower profit
            ),
            "profit_factor": FilterConfig(
                name="Profit Factor",
                description="Gross profits / gross losses (>1 = profitable)",
                threshold=1.5,
                threshold_min=1.0,
                threshold_max=5.0,
                weight=2.5,
                pass_bonus=25,
                fail_penalty=5,
            ),
            "win_rate": FilterConfig(
                name="Win Rate %",
                description="Percentage of winning trades",
                threshold=55,
                threshold_min=45,
                threshold_max=80,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=5,
            ),
            "sharpe_ratio": FilterConfig(
                name="Sharpe Ratio",
                description="Risk-adjusted returns (>1 = good)",
                threshold=1.0,
                threshold_min=0.3,
                threshold_max=3.0,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=5,
            ),
            "sortino_ratio": FilterConfig(
                name="Sortino Ratio",
                description="Downside risk-adjusted returns",
                threshold=1.2,
                threshold_min=0.5,
                threshold_max=4.0,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=0,
            ),

            # Consistency metrics
            "max_drawdown": FilterConfig(
                name="Max Drawdown %",
                description="Maximum peak-to-trough decline (lower = better)",
                threshold=40,
                threshold_min=10,
                threshold_max=60,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=10,
            ),
            "largest_win_pct": FilterConfig(
                name="Largest Win % of Total P/L",
                description="Single trade concentration (lower = more consistent)",
                threshold=40,
                threshold_min=20,
                threshold_max=80,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=5,
            ),

            # Activity metrics (informational, low weight)
            "num_trades": FilterConfig(
                name="Number of Trades",
                description="More trades = more signal",
                threshold=100,
                threshold_min=10,
                threshold_max=1000,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=0,
            ),
            "account_age_days": FilterConfig(
                name="Account Age (days)",
                description="Older accounts have more history",
                threshold=90,
                threshold_min=7,
                threshold_max=365,
                weight=0.5,
                pass_bonus=5,
                fail_penalty=0,
            ),

            # ============================================================
            # COPYTRADE VIABILITY METRICS
            # These help identify traders whose returns are actually copyable
            # vs those who got lucky on a few trades
            # ============================================================
            "market_win_rate": FilterConfig(
                name="Market Win Rate %",
                description="% of unique markets with profit (higher = more consistent, more copyable)",
                threshold=50,  # Want > 50% of markets to be profitable
                threshold_min=30,
                threshold_max=80,
                weight=2.5,  # High weight - key copyability indicator
                pass_bonus=25,
                fail_penalty=15,
            ),
            "top_3_concentration": FilterConfig(
                name="Top 3 Trades Concentration %",
                description="% of profit from top 3 trades (lower = more distributed, more copyable)",
                threshold=50,  # Want < 50% - if higher, too dependent on catching specific trades
                threshold_min=20,
                threshold_max=90,
                weight=2.5,  # High weight - critical luck detector
                pass_bonus=25,
                fail_penalty=20,
            ),
            "simulated_50pct_capture": FilterConfig(
                name="50% Capture Simulation ($)",
                description="Median P/L if you only capture 50% of trades (positive = copyable)",
                threshold=100,  # Want at least $100 profit at 50% capture
                threshold_min=0,
                threshold_max=10000,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),
            "redemption_pct": FilterConfig(
                name="Resolution Profit %",
                description="% of profits from holding to resolution vs selling early (informational)",
                threshold=30,
                threshold_min=0,
                threshold_max=100,
                weight=0.5,  # Low weight - just informational
                pass_bonus=5,
                fail_penalty=0,
            ),
        },

        # Weights focus on profitability, not specialization
        pl_consistency_weight=0.40,  # High - this is what we care about
        pattern_match_weight=0.25,
        specialization_weight=0.10,  # Low - don't penalize mainstream
        risk_weight=0.25,
        min_composite_score=20,  # Very low threshold - collect broadly, sort later
    )


def create_similar_to_config() -> ModeConfig:
    """Create configuration for Similar-To mode.

    Finds accounts with trading patterns similar to a reference account.
    """
    return ModeConfig(
        mode=DiscoveryMode.SIMILAR_TO,
        name="Similar To Reference",
        description="Find accounts with similar trading patterns to a reference wallet",

        hard_filters={
            "min_trades": FilterConfig(
                name="Minimum Trades",
                description="Need enough data for pattern matching",
                is_hard_filter=True,
                threshold=50,
                threshold_min=20,
                threshold_max=200,
            ),
            "must_be_profitable": FilterConfig(
                name="Must Be Profitable",
                description="Only copy profitable traders",
                is_hard_filter=True,
                threshold=0,
            ),
            "min_account_age": FilterConfig(
                name="Minimum Account Age (days)",
                description="Established accounts only",
                is_hard_filter=True,
                threshold=30,
                threshold_min=14,
                threshold_max=90,
            ),
        },

        soft_filters={
            # Pattern similarity
            "position_size_correlation": FilterConfig(
                name="Position Size Correlation",
                description="How similar are position sizes (0-100)",
                threshold=60,
                threshold_min=30,
                threshold_max=95,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),
            "odds_preference_correlation": FilterConfig(
                name="Odds Preference Correlation",
                description="Similar odds/probability preferences (0-100)",
                threshold=60,
                threshold_min=30,
                threshold_max=95,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),
            "category_overlap": FilterConfig(
                name="Category Overlap %",
                description="Shared market categories",
                threshold=50,
                threshold_min=20,
                threshold_max=90,
                weight=2.0,
                pass_bonus=20,
                fail_penalty=10,
            ),
            "market_overlap": FilterConfig(
                name="Market Overlap %",
                description="Shared specific markets",
                threshold=20,
                threshold_min=5,
                threshold_max=50,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=5,
            ),

            # Performance similarity
            "win_rate_delta": FilterConfig(
                name="Win Rate Delta (% points)",
                description="Difference in win rate from reference",
                threshold=10,  # Within 10% of reference
                threshold_min=5,
                threshold_max=20,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=5,
            ),
            "profit_factor_similarity": FilterConfig(
                name="Profit Factor Similarity",
                description="Similar risk-adjusted returns",
                threshold=0.5,  # Within 0.5 of reference
                threshold_min=0.2,
                threshold_max=1.0,
                weight=1.5,
                pass_bonus=15,
                fail_penalty=5,
            ),

            # Behavioral similarity
            "trade_frequency_ratio": FilterConfig(
                name="Trade Frequency Ratio",
                description="Trades/day ratio to reference (1.0 = same)",
                threshold=2.0,  # Within 2x of reference
                threshold_min=1.2,
                threshold_max=5.0,
                weight=1.0,
                pass_bonus=10,
                fail_penalty=5,
            ),
        },

        pl_consistency_weight=0.25,
        pattern_match_weight=0.35,  # Pattern matching is key
        specialization_weight=0.25,
        risk_weight=0.15,
        min_composite_score=55,
    )


# Registry of all mode configurations
MODE_CONFIGS: dict[DiscoveryMode, ModeConfig] = {
    DiscoveryMode.WIDE_NET_PROFITABILITY: create_wide_net_profitability_config(),
    DiscoveryMode.MICRO_BET_HUNTER: create_micro_bet_hunter_config(),
    DiscoveryMode.NICHE_SPECIALIST: create_niche_specialist_config(),
    DiscoveryMode.INSIDER_DETECTION: create_insider_detection_config(),
    DiscoveryMode.SIMILAR_TO: create_similar_to_config(),
}


@dataclass
class ScoringResult:
    """Complete result from scoring an account."""

    # Overall
    composite_score: float
    passes_threshold: bool

    # Filter results
    hard_filter_passed: bool
    hard_filter_failures: list[str]

    # Component scores (0-100)
    pl_consistency_score: float
    pattern_match_score: float
    specialization_score: float
    risk_score: float  # For normal modes, lower is better. For insider, higher = more suspicious

    # Detailed breakdown
    filter_results: dict[str, dict]  # filter_name -> {passed, value, threshold, impact}

    # Red flags
    red_flags: list[dict]
    red_flag_count: int

    # Confidence
    confidence_level: str  # "high", "medium", "low" based on data quality
    data_quality_score: float  # 0-100

    def to_dict(self) -> dict:
        return {
            "composite_score": round(self.composite_score, 2),
            "passes_threshold": self.passes_threshold,
            "hard_filter_passed": self.hard_filter_passed,
            "hard_filter_failures": self.hard_filter_failures,
            "breakdown": {
                "pl_consistency_score": round(self.pl_consistency_score, 2),
                "pattern_match_score": round(self.pattern_match_score, 2),
                "specialization_score": round(self.specialization_score, 2),
                "risk_score": round(self.risk_score, 2),
            },
            "filter_results": self.filter_results,
            "red_flags": self.red_flags,
            "red_flag_count": self.red_flag_count,
            "confidence_level": self.confidence_level,
            "data_quality_score": round(self.data_quality_score, 2),
        }


class ScoringEngine:
    """Multi-layered scoring engine for account evaluation."""

    def __init__(self, mode: DiscoveryMode = DiscoveryMode.NICHE_SPECIALIST):
        """Initialize with a discovery mode."""
        self.mode = mode
        self.config = MODE_CONFIGS.get(mode, create_niche_specialist_config())

    def set_mode(self, mode: DiscoveryMode) -> None:
        """Change discovery mode."""
        self.mode = mode
        self.config = MODE_CONFIGS.get(mode, create_niche_specialist_config())

    def update_config(self, updates: dict) -> None:
        """Update configuration from UI.

        Args:
            updates: Dict with filter updates like:
                {"hard_filters": {"min_trades": {"threshold": 150}}}
        """
        if "hard_filters" in updates:
            for name, values in updates["hard_filters"].items():
                if name in self.config.hard_filters:
                    for key, value in values.items():
                        setattr(self.config.hard_filters[name], key, value)

        if "soft_filters" in updates:
            for name, values in updates["soft_filters"].items():
                if name in self.config.soft_filters:
                    for key, value in values.items():
                        setattr(self.config.soft_filters[name], key, value)

        # Update weights
        for weight_key in ["pl_consistency_weight", "pattern_match_weight",
                          "specialization_weight", "risk_weight", "min_composite_score"]:
            if weight_key in updates:
                setattr(self.config, weight_key, updates[weight_key])

    def get_config(self) -> dict:
        """Get current configuration for UI."""
        return self.config.to_dict()

    def score_account(
        self,
        pl_metrics: Optional[PLCurveMetrics],
        pattern_metrics: Optional[TradingPatternMetrics],
        insider_signals: Optional[InsiderSignals] = None,
        category_metrics: Optional[dict] = None,
    ) -> ScoringResult:
        """Score an account using the multi-layered system.

        Args:
            pl_metrics: P/L curve metrics
            pattern_metrics: Trading pattern metrics
            insider_signals: Insider detection signals (for insider mode)
            category_metrics: Category-specific metrics

        Returns:
            ScoringResult with complete breakdown
        """
        red_flags: list[dict] = []
        filter_results: dict[str, dict] = {}

        # LAYER 1: Hard Filters
        hard_passed, hard_failures = self._apply_hard_filters(
            pl_metrics, pattern_metrics, insider_signals, filter_results
        )

        if not hard_passed:
            return ScoringResult(
                composite_score=0,
                passes_threshold=False,
                hard_filter_passed=False,
                hard_filter_failures=hard_failures,
                pl_consistency_score=0,
                pattern_match_score=0,
                specialization_score=0,
                risk_score=0,
                filter_results=filter_results,
                red_flags=[],
                red_flag_count=0,
                confidence_level="low",
                data_quality_score=0,
            )

        # LAYER 2: Calculate component scores
        pl_score = self._calculate_pl_score(pl_metrics, filter_results, red_flags)
        pattern_score = self._calculate_pattern_score(pattern_metrics, filter_results, red_flags)
        spec_score = self._calculate_specialization_score(pattern_metrics, category_metrics, filter_results, red_flags)
        risk_score = self._calculate_risk_score(pl_metrics, pattern_metrics, insider_signals, filter_results, red_flags)

        # LAYER 3: Data quality assessment
        data_quality = self._assess_data_quality(pl_metrics, pattern_metrics)
        confidence = "high" if data_quality > 70 else "medium" if data_quality > 40 else "low"

        # LAYER 4: Composite score
        if self.mode == DiscoveryMode.INSIDER_DETECTION:
            # Insider mode: higher risk score = MORE interesting
            composite = (
                pl_score * self.config.pl_consistency_weight +
                pattern_score * self.config.pattern_match_weight +
                spec_score * self.config.specialization_weight +
                risk_score * self.config.risk_weight  # Risk IS the score
            )
        else:
            # Normal modes: lower risk is better
            composite = (
                pl_score * self.config.pl_consistency_weight +
                pattern_score * self.config.pattern_match_weight +
                spec_score * self.config.specialization_weight +
                (100 - risk_score) * self.config.risk_weight
            )

        # Apply confidence adjustment (less aggressive for wide net mode)
        if self.mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
            # Wide net mode: minimal confidence penalty - we want to collect broadly
            if confidence == "low":
                composite *= 0.95  # Only 5% penalty instead of 20%
            elif confidence == "medium":
                composite *= 0.98  # Only 2% penalty instead of 10%
        else:
            # Other modes: standard confidence adjustment
            if confidence == "low":
                composite *= 0.8
            elif confidence == "medium":
                composite *= 0.9

        composite = max(0, min(100, composite))

        return ScoringResult(
            composite_score=composite,
            passes_threshold=composite >= self.config.min_composite_score,
            hard_filter_passed=True,
            hard_filter_failures=[],
            pl_consistency_score=pl_score,
            pattern_match_score=pattern_score,
            specialization_score=spec_score,
            risk_score=risk_score,
            filter_results=filter_results,
            red_flags=red_flags,
            red_flag_count=len(red_flags),
            confidence_level=confidence,
            data_quality_score=data_quality,
        )

    def _apply_hard_filters(
        self,
        pl_metrics: Optional[PLCurveMetrics],
        pattern_metrics: Optional[TradingPatternMetrics],
        insider_signals: Optional[InsiderSignals],
        filter_results: dict,
    ) -> tuple[bool, list[str]]:
        """Apply hard filters that auto-reject."""
        failures = []

        for name, fconfig in self.config.hard_filters.items():
            if not fconfig.enabled:
                continue

            value = self._get_metric_value(name, pl_metrics, pattern_metrics, insider_signals)
            passed = self._evaluate_filter(name, value, fconfig)

            filter_results[name] = {
                "passed": passed,
                "value": value,
                "threshold": fconfig.threshold,
                "is_hard": True,
            }

            if not passed:
                failures.append(f"{fconfig.name}: {value} vs threshold {fconfig.threshold}")

        return len(failures) == 0, failures

    def _calculate_pl_score(
        self,
        pl_metrics: Optional[PLCurveMetrics],
        filter_results: dict,
        red_flags: list,
    ) -> float:
        """Calculate P/L consistency score.

        Key indicators of systematic trading:
        - High Sharpe/Sortino ratio (risk-adjusted returns)
        - Good profit factor (wins > losses)
        - Low drawdowns (risk management)
        - Distributed wins (not luck-dependent)
        - Consistent win/loss sizes (disciplined exits)
        """
        if not pl_metrics:
            return 30  # Neutral

        score = 50  # Start neutral

        # Sharpe ratio - risk-adjusted returns
        if pl_metrics.sharpe_ratio >= 1.5:
            score += 20
        elif pl_metrics.sharpe_ratio >= 1.0:
            score += 15
        elif pl_metrics.sharpe_ratio >= 0.5:
            score += 8
        elif pl_metrics.sharpe_ratio < 0.3:
            score -= 10
            red_flags.append({
                "type": RedFlagType.VOLATILE_PL_CURVE.value,
                "severity": "medium",
                "description": f"Low Sharpe ratio: {pl_metrics.sharpe_ratio:.2f}",
            })

        # Sortino ratio bonus - rewards downside protection (systematic traders manage losses)
        if pl_metrics.sortino_ratio >= 2.0:
            score += 10
        elif pl_metrics.sortino_ratio >= 1.5:
            score += 5

        # Profit factor - core edge indicator
        if pl_metrics.profit_factor >= 3.0:
            score += 20
        elif pl_metrics.profit_factor >= 2.0:
            score += 15
        elif pl_metrics.profit_factor >= 1.5:
            score += 10
        elif pl_metrics.profit_factor < 1.0:
            score -= 20

        # Drawdown - risk management indicator
        if pl_metrics.max_drawdown_pct > 0.4:
            score -= 20
            red_flags.append({
                "type": RedFlagType.LARGE_DRAWDOWNS.value,
                "severity": "high",
                "description": f"Max drawdown: {pl_metrics.max_drawdown_pct*100:.1f}%",
            })
        elif pl_metrics.max_drawdown_pct > 0.25:
            score -= 10
        elif pl_metrics.max_drawdown_pct < 0.10:
            score += 15  # Very low drawdown = excellent risk management
        elif pl_metrics.max_drawdown_pct < 0.15:
            score += 10

        # TOP 3 WINS CONCENTRATION - Critical luck detector
        # If top 3 trades = most of the profit, it's likely luck not skill
        if pl_metrics.top_3_wins_pct_of_total > 0.85:
            score -= 30
            red_flags.append({
                "type": RedFlagType.SINGLE_WIN_DEPENDENT.value,
                "severity": "critical",
                "description": f"Top 3 trades = {pl_metrics.top_3_wins_pct_of_total*100:.0f}% of total P/L (luck indicator)",
            })
        elif pl_metrics.top_3_wins_pct_of_total > 0.70:
            score -= 20
            red_flags.append({
                "type": RedFlagType.SINGLE_WIN_DEPENDENT.value,
                "severity": "high",
                "description": f"Top 3 trades = {pl_metrics.top_3_wins_pct_of_total*100:.0f}% of total P/L",
            })
        elif pl_metrics.top_3_wins_pct_of_total > 0.50:
            score -= 10
        elif pl_metrics.top_3_wins_pct_of_total < 0.30:
            score += 10  # Well-distributed profits = systematic

        # Single win dependency (still check, but less harsh since top 3 is checked)
        if pl_metrics.largest_win_pct_of_total > 0.5:
            score -= 15  # Reduced from 25 since top_3 check covers this
            if pl_metrics.top_3_wins_pct_of_total <= 0.70:  # Only add if not already flagged
                red_flags.append({
                    "type": RedFlagType.SINGLE_WIN_DEPENDENT.value,
                    "severity": "high",
                    "description": f"{pl_metrics.largest_win_pct_of_total*100:.0f}% of P/L from one trade",
                })
        elif pl_metrics.largest_win_pct_of_total > 0.35:
            score -= 8

        # WIN/LOSS SIZE RATIO - Disciplined exit indicator
        # Systematic traders often have defined risk/reward ratios
        avg_win = float(pl_metrics.avg_win_size) if pl_metrics.avg_win_size else 0
        avg_loss = float(pl_metrics.avg_loss_size) if pl_metrics.avg_loss_size else 1
        if avg_loss > 0 and avg_win > 0:
            win_loss_ratio = avg_win / avg_loss
            if win_loss_ratio >= 3.0:
                score += 10  # Great risk/reward
            elif win_loss_ratio >= 2.0:
                score += 5
            elif win_loss_ratio < 0.5:
                score -= 10  # Wins smaller than losses = poor discipline
                red_flags.append({
                    "type": RedFlagType.VOLATILE_PL_CURVE.value,
                    "severity": "medium",
                    "description": f"Avg win (${avg_win:.0f}) smaller than avg loss (${avg_loss:.0f})",
                })

        # WIN COUNT CHECK - Need enough winning trades for statistical significance
        win_count = pl_metrics.win_count if hasattr(pl_metrics, 'win_count') else 0
        if win_count >= 20:
            score += 5  # Good sample size
        elif win_count < 5:
            score -= 10  # Too few wins to judge

        # EQUITY CURVE SMOOTHNESS - Systematic traders have steady growth
        # Low average drawdown = smooth equity curve
        if pl_metrics.avg_drawdown_pct < 0.05:
            score += 10  # Very smooth curve
        elif pl_metrics.avg_drawdown_pct < 0.10:
            score += 5
        elif pl_metrics.avg_drawdown_pct > 0.25:
            score -= 10  # Choppy equity curve

        # Recovery time consistency - systematic traders recover quickly and consistently
        if pl_metrics.avg_recovery_time_days > 0:
            if pl_metrics.avg_recovery_time_days < 7:
                score += 5  # Quick recovery from drawdowns
            elif pl_metrics.avg_recovery_time_days > 30:
                score -= 5  # Slow recovery = struggling to return to form

        # WIN/LOSS STREAK BALANCE - Extreme streaks suggest luck/variance
        # Systematic traders have moderate, balanced streaks
        max_streak = max(pl_metrics.longest_win_streak, pl_metrics.longest_loss_streak)
        if max_streak > 15:
            score -= 10  # Very long streak suggests variance, not skill
        elif pl_metrics.longest_win_streak > 10 and pl_metrics.longest_loss_streak < 3:
            score -= 5  # Suspicious - very asymmetric streaks

        # ============================================================
        # COPYTRADE VIABILITY METRICS
        # These indicate how well returns would transfer to a copytrader
        # ============================================================

        # MARKET WIN RATE - More important than trade win rate for copytrading
        # If they profit on most markets, you're more likely to profit even if you miss some
        if pl_metrics.market_win_rate >= 0.60:
            score += 15  # Very consistent - profits on most markets
        elif pl_metrics.market_win_rate >= 0.50:
            score += 10
        elif pl_metrics.market_win_rate >= 0.40:
            score += 5
        elif pl_metrics.market_win_rate < 0.25:
            score -= 15  # Poor consistency - profits on few markets
            red_flags.append({
                "type": RedFlagType.SINGLE_WIN_DEPENDENT.value,
                "severity": "high",
                "description": f"Only {pl_metrics.market_win_rate*100:.0f}% of markets profitable - likely luck",
            })

        # SIMULATED CAPTURE - Would copytrading be profitable?
        # If 50% capture simulation is still profitable, returns are copyable
        if pl_metrics.simulated_50pct_capture_median > 0:
            if pl_metrics.simulated_50pct_capture_median >= 1000:
                score += 10  # Very copyable
            elif pl_metrics.simulated_50pct_capture_median >= 100:
                score += 5
        else:
            score -= 10  # 50% capture would likely lose money
            red_flags.append({
                "type": RedFlagType.SINGLE_WIN_DEPENDENT.value,
                "severity": "medium",
                "description": f"50% capture simulation shows ${pl_metrics.simulated_50pct_capture_median:.0f} - returns not copyable",
            })

        return max(0, min(100, score))

    def _calculate_pattern_score(
        self,
        pattern_metrics: Optional[TradingPatternMetrics],
        filter_results: dict,
        red_flags: list,
    ) -> float:
        """Calculate pattern match score using soft filters.

        Key indicators of systematic trading:
        - Consistent position sizing (low coefficient of variation)
        - Regular trading frequency (active over time, not bursty)
        - Market diversification (not over-concentrated, not over-spread)
        - Disciplined entry odds preferences
        """
        if not pattern_metrics:
            return 30

        score = 50

        # Apply soft filters
        for name, fconfig in self.config.soft_filters.items():
            if not fconfig.enabled:
                continue

            # Skip non-pattern filters
            if name not in ["ideal_avg_position", "position_consistency", "pct_under_10c",
                           "pct_under_5c", "win_rate_at_long_odds", "unique_markets",
                           "max_single_market_pct", "trades_per_week"]:
                continue

            value = self._get_metric_value(name, None, pattern_metrics, None)
            passed = self._evaluate_filter(name, value, fconfig)

            filter_results[name] = {
                "passed": passed,
                "value": value,
                "threshold": fconfig.threshold,
                "is_hard": False,
                "impact": fconfig.pass_bonus if passed else -fconfig.fail_penalty,
            }

            if passed:
                score += fconfig.pass_bonus * fconfig.weight
            else:
                score -= fconfig.fail_penalty * fconfig.weight

        # POSITION SIZE CONSISTENCY - Key systematic trader indicator
        # Coefficient of variation (CV) = std_dev / mean
        # Lower CV = more consistent position sizing = more systematic
        avg_pos = float(pattern_metrics.avg_position_size_usd)
        pos_std_dev = float(pattern_metrics.position_size_std_dev)
        if avg_pos > 0:
            coefficient_of_variation = pos_std_dev / avg_pos
            if coefficient_of_variation < 0.5:
                score += 15  # Very consistent sizing - strong systematic signal
            elif coefficient_of_variation < 1.0:
                score += 8  # Reasonably consistent
            elif coefficient_of_variation > 2.5:
                score -= 10  # Highly erratic sizing - gambler behavior
                red_flags.append({
                    "type": RedFlagType.VOLATILE_PL_CURVE.value,
                    "severity": "low",
                    "description": f"Erratic position sizing (CV: {coefficient_of_variation:.1f})",
                })
            elif coefficient_of_variation > 1.5:
                score -= 5

        # TRADING FREQUENCY CONSISTENCY
        # Systematic traders trade regularly, not in bursts
        if pattern_metrics.active_days > 0 and pattern_metrics.account_age_days > 0:
            activity_ratio = pattern_metrics.active_days / pattern_metrics.account_age_days
            if activity_ratio >= 0.5:
                score += 10  # Active more than half the days = regular trader
            elif activity_ratio >= 0.3:
                score += 5
            elif activity_ratio < 0.1:
                score -= 10  # Very sporadic - not systematic
                red_flags.append({
                    "type": RedFlagType.INSUFFICIENT_HISTORY.value,
                    "severity": "low",
                    "description": f"Sporadic trading ({activity_ratio*100:.0f}% of days active)",
                })

        # MARKET DIVERSIFICATION SWEET SPOT
        # Too few markets = concentrated risk, too many = unfocused
        num_markets = pattern_metrics.unique_markets_traded
        if 10 <= num_markets <= 100:
            score += 5  # Good diversification
        elif num_markets < 5:
            score -= 10  # Over-concentrated
        elif num_markets > 200:
            score -= 5  # Scatter-shot approach, less systematic

        # MEDIAN VS AVG POSITION SIZE - Detect outlier bets
        # If median << avg, there are large outlier bets (gambler behavior)
        median_pos = float(pattern_metrics.median_position_size_usd)
        if avg_pos > 0 and median_pos > 0:
            median_to_avg_ratio = median_pos / avg_pos
            if median_to_avg_ratio >= 0.7:
                score += 5  # Median close to avg = no big outliers
            elif median_to_avg_ratio < 0.3:
                score -= 10  # Big gap = occasional yolo bets
                red_flags.append({
                    "type": RedFlagType.POSITION_CONCENTRATION.value,
                    "severity": "low",
                    "description": f"Position outliers detected (median/avg: {median_to_avg_ratio:.2f})",
                })

        # Whale check (keep original logic)
        if avg_pos > 500:
            score -= 15
            red_flags.append({
                "type": RedFlagType.WHALE_ACCOUNT.value,
                "severity": "medium",
                "description": f"Large avg position: ${avg_pos:.0f}",
            })

        # Insufficient data check (keep but adjust threshold)
        if pattern_metrics.total_trades < 20:
            score -= 25  # Severe penalty - can't judge systematic with few trades
            red_flags.append({
                "type": RedFlagType.INSUFFICIENT_HISTORY.value,
                "severity": "high",
                "description": f"Only {pattern_metrics.total_trades} trades",
            })
        elif pattern_metrics.total_trades < 50:
            score -= 10
            red_flags.append({
                "type": RedFlagType.INSUFFICIENT_HISTORY.value,
                "severity": "medium",
                "description": f"Only {pattern_metrics.total_trades} trades",
            })

        # RECENCY CHECK - Prefer active accounts
        days_since_last = getattr(pattern_metrics, 'days_since_last_trade', 0)
        if days_since_last <= 7:
            score += 5  # Active recently
        elif days_since_last > 60:
            score -= 10  # Dormant account
        elif days_since_last > 30:
            score -= 5

        return max(0, min(100, score))

    def _calculate_specialization_score(
        self,
        pattern_metrics: Optional[TradingPatternMetrics],
        category_metrics: Optional[dict],
        filter_results: dict,
        red_flags: list,
    ) -> float:
        """Calculate specialization score.

        For WIDE_NET_PROFITABILITY mode: Returns neutral score (no category gatekeeping)
        For other modes: Rewards niche focus, penalizes mainstream-heavy accounts
        """
        if not pattern_metrics:
            return 50  # Neutral

        # Wide Net mode: Skip all category-based scoring - we don't care
        if self.mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
            return 50  # Always neutral - no bonus or penalty for categories

        score = 50
        categories = pattern_metrics.category_breakdown

        # Niche focus (only for specialist modes)
        niche_pct = pattern_metrics.niche_market_pct
        if niche_pct >= 0.7:
            score += 25
        elif niche_pct >= 0.5:
            score += 15
        elif niche_pct >= 0.3:
            score += 5
        elif niche_pct < 0.2:
            score -= 10

        # Mainstream penalty (only for specialist modes)
        mainstream_pct = categories.get("politics", 0) + categories.get("sports", 0)
        if mainstream_pct > 0.4:
            score -= 20
            red_flags.append({
                "type": RedFlagType.MAINSTREAM_HEAVY.value,
                "severity": "low",
                "description": f"{mainstream_pct*100:.0f}% in politics/sports",
            })
        elif mainstream_pct > 0.25:
            score -= 10

        # Category specialization
        if category_metrics:
            # Check if specialty category outperforms
            primary_cat = max(categories, key=categories.get) if categories else None
            if primary_cat and primary_cat in category_metrics:
                cat_win_rate = category_metrics[primary_cat].get("win_rate", 0)
                if cat_win_rate > 0.58:
                    score += 20

        return max(0, min(100, score))

    def _calculate_risk_score(
        self,
        pl_metrics: Optional[PLCurveMetrics],
        pattern_metrics: Optional[TradingPatternMetrics],
        insider_signals: Optional[InsiderSignals],
        filter_results: dict,
        red_flags: list,
    ) -> float:
        """Calculate risk score.

        For normal modes: lower is better (less risky)
        For insider mode: higher means more suspicious signals
        """
        if self.mode == DiscoveryMode.INSIDER_DETECTION:
            return self._calculate_insider_suspicion_score(
                pl_metrics, pattern_metrics, insider_signals, filter_results, red_flags
            )

        # Normal risk calculation
        score = 0  # Start at 0 risk

        if pl_metrics:
            # Drawdown risk
            if pl_metrics.max_drawdown_pct > 0.4:
                score += 30
            elif pl_metrics.max_drawdown_pct > 0.25:
                score += 15

            # Volatility risk
            if pl_metrics.sharpe_ratio < 0.5:
                score += 20

        if pattern_metrics:
            # Concentration risk
            max_market_pct = max(pattern_metrics.category_breakdown.values()) if pattern_metrics.category_breakdown else 0
            if max_market_pct > 0.8:
                score += 15

        return max(0, min(100, score))

    def _calculate_insider_suspicion_score(
        self,
        pl_metrics: Optional[PLCurveMetrics],
        pattern_metrics: Optional[TradingPatternMetrics],
        insider_signals: Optional[InsiderSignals],
        filter_results: dict,
        red_flags: list,
    ) -> float:
        """Calculate suspicion score for insider detection mode."""
        score = 0

        if not insider_signals:
            return score

        # Fresh account signals
        if insider_signals.account_age_days < 14:
            score += 25
            red_flags.append({
                "type": RedFlagType.FRESH_ACCOUNT_LARGE_BET.value,
                "severity": "critical",
                "description": f"Account only {insider_signals.account_age_days} days old",
            })
        elif insider_signals.account_age_days < 30:
            score += 15

        # Fresh account + large position
        if insider_signals.fresh_account_large_positions:
            score += 30

        # Large bets on unlikely events
        if insider_signals.large_bets_on_unlikely_events:
            num_bets = len(insider_signals.large_bets_on_unlikely_events)
            score += min(40, num_bets * 15)
            red_flags.append({
                "type": RedFlagType.YOLO_BET_NEAR_EXPIRY.value,
                "severity": "high",
                "description": f"{num_bets} large bets on <10% odds events",
            })

        # Concentration
        if insider_signals.single_market_concentration > 0.8:
            score += 20
        elif insider_signals.single_market_concentration > 0.6:
            score += 10

        # Probability score from signals
        score += insider_signals.insider_probability_score * 30

        return max(0, min(100, score))

    def _get_metric_value(
        self,
        filter_name: str,
        pl_metrics: Optional[PLCurveMetrics],
        pattern_metrics: Optional[TradingPatternMetrics],
        insider_signals: Optional[InsiderSignals],
    ) -> float:
        """Get the metric value for a filter."""
        # Pattern metrics
        if pattern_metrics:
            # Compute mainstream % from category breakdown
            mainstream_pct = sum(
                pattern_metrics.category_breakdown.get(cat, 0)
                for cat in ["politics", "sports"]
            ) * 100  # Convert to percentage

            mapping = {
                "min_trades": pattern_metrics.total_trades,
                "num_trades": pattern_metrics.total_trades,
                "min_account_age": pattern_metrics.account_age_days,
                "account_age_days": pattern_metrics.account_age_days,
                "max_avg_position": float(pattern_metrics.avg_position_size_usd),
                "ideal_avg_position": float(pattern_metrics.avg_position_size_usd),
                "pct_under_10c": pattern_metrics.pct_trades_under_10c * 100,
                "pct_under_5c": pattern_metrics.pct_trades_under_5c * 100,
                "unique_markets": pattern_metrics.unique_markets_traded,
                "trades_per_week": pattern_metrics.trades_per_day_avg * 7,
                "max_mainstream_pct": mainstream_pct,
                "niche_category_pct": pattern_metrics.niche_market_pct * 100,
            }
            if filter_name in mapping:
                return mapping[filter_name]

        # P/L metrics
        if pl_metrics:
            mapping = {
                "must_be_profitable": float(pl_metrics.total_realized_pnl),
                "min_total_pnl": float(pl_metrics.total_realized_pnl),
                "total_pnl": float(pl_metrics.total_realized_pnl),
                "max_single_win_pct": pl_metrics.largest_win_pct_of_total * 100,
                "largest_win_pct": pl_metrics.largest_win_pct_of_total * 100,
                "profit_factor": pl_metrics.profit_factor,
                "sharpe_ratio": pl_metrics.sharpe_ratio,
                "sortino_ratio": pl_metrics.sortino_ratio,
                "max_drawdown": pl_metrics.max_drawdown_pct * 100,
                "win_rate": pl_metrics.win_rate * 100,
                "win_rate_at_long_odds": pl_metrics.win_rate * 100,  # Approximation
                # Copytrade viability metrics
                "market_win_rate": pl_metrics.market_win_rate * 100,
                "top_3_concentration": pl_metrics.top_3_wins_pct_of_total * 100,
                "top_5_concentration": pl_metrics.top_5_wins_pct_of_total * 100,
                "top_10_concentration": pl_metrics.top_10_wins_pct_of_total * 100,
                "simulated_50pct_capture": pl_metrics.simulated_50pct_capture_median,
                "simulated_25pct_capture": pl_metrics.simulated_25pct_capture_median,
                "redemption_pct": pl_metrics.redemption_pct_of_profit * 100,
            }
            if filter_name in mapping:
                return mapping[filter_name]

        # Insider signals
        if insider_signals:
            mapping = {
                "max_account_age": insider_signals.account_age_days,
                "min_position_size": float(insider_signals.largest_position_usd),
                "very_fresh_account": insider_signals.account_age_days,
                "fresh_account": insider_signals.account_age_days,
                "large_first_bet": float(insider_signals.largest_position_usd),  # Approximation
                "max_position_size": float(insider_signals.largest_position_usd),
                "single_market_concentration": insider_signals.single_market_concentration * 100,
                "bets_on_unlikely_events": len(insider_signals.large_bets_on_unlikely_events),
            }
            if filter_name in mapping:
                return mapping[filter_name]

        return 0

    def _evaluate_filter(
        self,
        filter_name: str,
        value: float,
        fconfig: FilterConfig,
    ) -> bool:
        """Evaluate if a filter passes."""
        # Most filters: value >= threshold is good
        # Some filters: value <= threshold is good (e.g., max_drawdown)

        less_is_better = filter_name in [
            "max_avg_position", "max_single_win_pct", "max_drawdown",
            "max_mainstream_pct", "max_single_market_pct", "max_account_age",
            "win_rate_delta", "recent_activity", "very_fresh_account", "fresh_account",
            "largest_win_pct",  # Lower concentration = better
            # Copytrade viability: lower concentration = more copyable
            "top_3_concentration", "top_5_concentration", "top_10_concentration",
        ]

        if less_is_better:
            return value <= fconfig.threshold
        else:
            return value >= fconfig.threshold

    def _assess_data_quality(
        self,
        pl_metrics: Optional[PLCurveMetrics],
        pattern_metrics: Optional[TradingPatternMetrics],
    ) -> float:
        """Assess data quality for confidence scoring."""
        score = 50  # Base

        if pattern_metrics:
            # More trades = higher confidence
            if pattern_metrics.total_trades >= 200:
                score += 25
            elif pattern_metrics.total_trades >= 100:
                score += 15
            elif pattern_metrics.total_trades >= 50:
                score += 5

            # Longer history = higher confidence
            if pattern_metrics.account_age_days >= 180:
                score += 15
            elif pattern_metrics.account_age_days >= 90:
                score += 10
            elif pattern_metrics.account_age_days >= 30:
                score += 5

        if pl_metrics:
            # Lower volatility = more reliable metrics
            if pl_metrics.sharpe_ratio > 1.0:
                score += 10

        return max(0, min(100, score))

    @staticmethod
    def get_all_modes() -> list[dict]:
        """Get all available modes with their configs."""
        return [
            {
                "mode": mode.value,
                "name": config.name,
                "description": config.description,
                "config": config.to_dict(),
            }
            for mode, config in MODE_CONFIGS.items()
        ]
