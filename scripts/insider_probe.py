#!/usr/bin/env python3
"""
Insider Probe Script - Identify accounts with high probability of insider trading.

Based on analysis of 17 documented insider trader cases, this script probes
Polymarket for accounts matching the insider footprint with reasonable variance.

Key footprint characteristics (from documented cases):
- 70% had account age < 30 days
- 70% had entry odds < 30%
- 70% had single market/category focus
- 65% had zero prior trades before their "hit"
- 80%+ win rate (15/17 cases had 100%, 1 had 80%)
- No hedging positions (YES + NO on same market)
- Large position sizes ($20K+ cumulative)

Output: data/insider_probe_results.json (separate from mass scan data)

Usage:
    python3 scripts/insider_probe.py                # Live detection mode
    python3 scripts/insider_probe.py --historical   # Historical detection (skips account age filter)
"""
import argparse
import asyncio
import json
import sys
import re
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Set
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient
from src.api.gamma import GammaAPIClient
from src.discovery.analyzer import AccountAnalyzer, LeaderboardEntry
from src.discovery.models import DiscoveryMode
from src.discovery.service import LeaderboardClient


# =============================================================================
# INSIDER FOOTPRINT CONFIGURATION
# =============================================================================
# These thresholds are calibrated from 17 documented insider cases.
# We use SOFT thresholds with variance to avoid overfitting.

class InsiderConfig:
    """Configuration derived from insider footprint analysis."""

    # === LEADERBOARD FILTERS (Initial Collection) ===
    # Cast a wide net initially, then filter down
    MIN_PROFIT = 5000           # $5K minimum (some insiders started small)
    MAX_PROFIT = 500_000_000    # $500M cap (bigwinner01 was $250M+)
    MAX_ACCOUNTS_TO_SCAN = 5000 # Pull top 5000 profitable accounts

    # === ACCOUNT AGE THRESHOLDS ===
    # 70% of insiders had accounts < 30 days old
    # Allow up to 90 days to catch dormant-then-active patterns (ricosuave666)
    ACCOUNT_AGE_STRONG_SIGNAL = 14    # < 14 days = strong signal
    ACCOUNT_AGE_MODERATE_SIGNAL = 30  # 14-30 days = moderate signal
    ACCOUNT_AGE_WEAK_SIGNAL = 90      # 30-90 days = weak signal (dormancy pattern)

    # === WIN RATE THRESHOLDS ===
    # 15/17 cases had 100% win rate, but Annica had 80%
    # Use minimum 70% to allow variance
    MIN_WIN_RATE = 0.70         # 70% minimum (variance from 80% Annica case)
    STRONG_WIN_RATE = 0.95      # 95%+ is strong signal
    PERFECT_WIN_RATE = 1.00     # 100% is very strong signal
    MIN_RESOLVED_BETS = 2       # Need at least 2 resolved bets for win rate

    # === POSITION SIZE THRESHOLDS ===
    # Insider positions ranged from $2K to $340M
    # Using cumulative position (sum of entries, not single trade)
    MIN_CUMULATIVE_POSITION = 5000    # $5K minimum cumulative
    STRONG_POSITION = 25000           # $25K+ is strong signal
    WHALE_POSITION = 100000           # $100K+ is whale signal

    # === MARKET CONCENTRATION ===
    # 70% of insiders focused on single market/category
    MIN_CONCENTRATION = 0.50    # 50%+ in single category = signal
    STRONG_CONCENTRATION = 0.80 # 80%+ = strong signal
    SINGLE_MARKET_FOCUS = 0.95  # 95%+ = single market focus

    # === ENTRY ODDS ===
    # 70% entered at < 30% odds (betting on longshots)
    MAX_ENTRY_ODDS_SIGNAL = 0.35      # Entry at < 35% odds = signal
    STRONG_LOW_ODDS = 0.15            # Entry at < 15% odds = strong signal
    EXTREME_LONGSHOT = 0.05           # Entry at < 5% = extreme signal

    # === TRADE HISTORY ===
    # 65% had zero or minimal prior trades
    MAX_PRIOR_TRADES_STRONG = 5       # <= 5 trades = strong signal
    MAX_PRIOR_TRADES_MODERATE = 20    # 6-20 trades = moderate signal

    # === SCORING THRESHOLDS ===
    # Multi-dimensional validation to prevent false positives
    MIN_INSIDER_SCORE = 45      # Minimum to flag (LOW priority)
    MEDIUM_SCORE = 55           # MEDIUM priority threshold
    HIGH_SCORE = 70             # HIGH priority threshold
    CRITICAL_SCORE = 85         # CRITICAL priority threshold

    MIN_SIGNALS_FOR_FLAG = 3    # Need at least 3 signals to flag
    MIN_DIMENSIONS_FOR_HIGH = 2 # Need 2+ dimensions for HIGH priority

    # === CATEGORY RISK WEIGHTS ===
    # Higher weight = more information asymmetry risk
    CATEGORY_RISK = {
        "military": 8,
        "government": 7,
        "elections": 6,
        "politics": 6,
        "corporate": 5,
        "awards": 5,
        "sports": 4,
        "tech": 4,
        "crypto": 3,
        "social": 2,
        "entertainment": 2,
        "other": 1,
    }


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class InsiderPriority(Enum):
    CRITICAL = "critical"   # 85+ score, 5+ signals, 3+ dimensions
    HIGH = "high"           # 70-84 score, 4+ signals, 2+ dimensions
    MEDIUM = "medium"       # 55-69 score, 3+ signals, 2+ dimensions
    LOW = "low"             # 45-54 score, 3+ signals
    NORMAL = "normal"       # Below thresholds


@dataclass
class InsiderSignal:
    """Individual signal contributing to insider score."""
    name: str
    dimension: str  # account, trading, behavioral, contextual
    points: float
    description: str
    value: any = None


@dataclass
class FundingInfo:
    """Funding source and withdrawal information."""
    funding_source: Optional[str] = None
    funding_amount: Optional[float] = None
    funding_tx: Optional[str] = None
    withdrawal_dest: Optional[str] = None
    withdrawal_amount: Optional[float] = None
    withdrawal_tx: Optional[str] = None


@dataclass
class InsiderProbeResult:
    """Result of insider probe for a single account."""
    wallet_address: str
    total_pnl: float

    # Scoring
    insider_score: float
    priority: str
    confidence_interval: tuple  # (low, high)

    # Signal breakdown
    signals: List[dict]
    signal_count: int
    dimensions_active: int
    dimension_scores: Dict[str, float]

    # Account metrics
    account_age_days: int
    total_trades: int
    win_rate: float
    resolved_bets: int

    # Trading patterns
    avg_entry_odds: float
    cumulative_position: float
    single_largest_position: float
    unique_markets: int
    primary_category: str
    category_concentration: float
    has_hedging: bool

    # Timing
    trades_in_last_24h: int
    trades_in_off_hours: int  # 0-6 AM UTC

    # Funding (critical for sybil detection)
    funding_info: Optional[dict] = None

    # Metadata
    probed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ProbeStats:
    """Track probe progress."""
    total_scanned: int = 0
    passed_initial_filter: int = 0
    flagged_critical: int = 0
    flagged_high: int = 0
    flagged_medium: int = 0
    flagged_low: int = 0
    funding_extracted: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None

    @property
    def total_flagged(self) -> int:
        return self.flagged_critical + self.flagged_high + self.flagged_medium + self.flagged_low


# =============================================================================
# INSIDER SCORING ENGINE
# =============================================================================

class InsiderScorer:
    """Score accounts for insider trading probability."""

    def __init__(self, config: InsiderConfig = None, historical_mode: bool = False):
        self.config = config or InsiderConfig()
        self.historical_mode = historical_mode

    def score_account(
        self,
        wallet: str,
        account_age_days: int,
        total_trades: int,
        win_rate: float,
        resolved_bets: int,
        avg_entry_odds: float,
        cumulative_position: float,
        single_largest_position: float,
        unique_markets: int,
        primary_category: str,
        category_concentration: float,
        has_hedging: bool,
        trades_in_off_hours: int,
        total_pnl: float,
    ) -> InsiderProbeResult:
        """
        Score an account using the 4-dimension insider model.

        Dimensions:
        1. Account Signals (age, prior trades)
        2. Trading Signals (position size, entry odds, win rate)
        3. Behavioral Signals (concentration, hedging, timing)
        4. Contextual Signals (market category risk)
        """
        signals = []
        dimension_scores = {
            "account": 0.0,
            "trading": 0.0,
            "behavioral": 0.0,
            "contextual": 0.0,
        }

        cfg = self.config

        # =====================================================================
        # DIMENSION 1: ACCOUNT SIGNALS (Max 25 pts)
        # =====================================================================

        # Account age scoring (soft gradient)
        # SKIP in historical mode - accounts that were new when they traded are old now
        if self.historical_mode:
            # In historical mode, give a moderate baseline score to all accounts
            # This avoids penalizing accounts that were new when they made insider trades
            pts = 8  # Moderate baseline - equivalent to ~14 day old account
            signals.append(InsiderSignal(
                "historical_mode", "account", pts,
                "Historical mode: account age check bypassed",
                account_age_days
            ))
        elif account_age_days < 1:
            pts = 15
            signals.append(InsiderSignal(
                "ultra_fresh_account", "account", pts,
                f"Account < 1 day old",
                account_age_days
            ))
        elif account_age_days <= cfg.ACCOUNT_AGE_STRONG_SIGNAL:
            # Linear decay: 15 -> 10 over 14 days
            pts = 15 - (5 * (account_age_days / 14))
            signals.append(InsiderSignal(
                "fresh_account", "account", pts,
                f"Account {account_age_days} days old (< 14 days)",
                account_age_days
            ))
        elif account_age_days <= cfg.ACCOUNT_AGE_MODERATE_SIGNAL:
            # Linear decay: 10 -> 5 over 14-30 days
            pts = 10 - (5 * ((account_age_days - 14) / 16))
            signals.append(InsiderSignal(
                "young_account", "account", pts,
                f"Account {account_age_days} days old (14-30 days)",
                account_age_days
            ))
        elif account_age_days <= cfg.ACCOUNT_AGE_WEAK_SIGNAL:
            # Linear decay: 5 -> 0 over 30-90 days
            pts = 5 - (5 * ((account_age_days - 30) / 60))
            if pts > 0:
                signals.append(InsiderSignal(
                    "moderate_age", "account", pts,
                    f"Account {account_age_days} days old (possible dormancy pattern)",
                    account_age_days
                ))
        else:
            pts = 0

        dimension_scores["account"] += max(0, pts)

        # Prior trades scoring
        if total_trades <= 2:
            pts = 10
            signals.append(InsiderSignal(
                "minimal_history", "account", pts,
                f"Only {total_trades} trades (< 3)",
                total_trades
            ))
        elif total_trades <= cfg.MAX_PRIOR_TRADES_STRONG:
            pts = 8 - (3 * ((total_trades - 2) / 3))
            signals.append(InsiderSignal(
                "limited_history", "account", pts,
                f"Only {total_trades} trades (3-5)",
                total_trades
            ))
        elif total_trades <= cfg.MAX_PRIOR_TRADES_MODERATE:
            pts = 5 - (5 * ((total_trades - 5) / 15))
            if pts > 0:
                signals.append(InsiderSignal(
                    "some_history", "account", pts,
                    f"{total_trades} trades (6-20)",
                    total_trades
                ))
        else:
            pts = 0

        dimension_scores["account"] += max(0, pts)

        # =====================================================================
        # DIMENSION 2: TRADING SIGNALS (Max 35 pts)
        # =====================================================================

        # Cumulative position size (key metric)
        if cumulative_position >= cfg.WHALE_POSITION:
            pts = 12
            signals.append(InsiderSignal(
                "whale_position", "trading", pts,
                f"Whale position ${cumulative_position:,.0f} (> $100K)",
                cumulative_position
            ))
        elif cumulative_position >= cfg.STRONG_POSITION:
            pts = 10
            signals.append(InsiderSignal(
                "large_position", "trading", pts,
                f"Large position ${cumulative_position:,.0f} ($25K-$100K)",
                cumulative_position
            ))
        elif cumulative_position >= cfg.MIN_CUMULATIVE_POSITION:
            # Gradient from 4 -> 10 for $5K -> $25K
            pts = 4 + (6 * ((cumulative_position - 5000) / 20000))
            signals.append(InsiderSignal(
                "significant_position", "trading", pts,
                f"Significant position ${cumulative_position:,.0f} ($5K-$25K)",
                cumulative_position
            ))
        else:
            pts = 0

        dimension_scores["trading"] += max(0, min(12, pts))

        # Entry odds scoring
        if avg_entry_odds > 0:  # Only if we have odds data
            if avg_entry_odds <= cfg.EXTREME_LONGSHOT:
                pts = 8
                signals.append(InsiderSignal(
                    "extreme_longshot", "trading", pts,
                    f"Entry at {avg_entry_odds*100:.1f}% odds (< 5%)",
                    avg_entry_odds
                ))
            elif avg_entry_odds <= cfg.STRONG_LOW_ODDS:
                pts = 6
                signals.append(InsiderSignal(
                    "strong_longshot", "trading", pts,
                    f"Entry at {avg_entry_odds*100:.1f}% odds (5-15%)",
                    avg_entry_odds
                ))
            elif avg_entry_odds <= cfg.MAX_ENTRY_ODDS_SIGNAL:
                pts = 4
                signals.append(InsiderSignal(
                    "low_odds_entry", "trading", pts,
                    f"Entry at {avg_entry_odds*100:.1f}% odds (15-35%)",
                    avg_entry_odds
                ))
            else:
                pts = 0

            dimension_scores["trading"] += max(0, pts)

        # Win rate scoring (need minimum resolved bets)
        if resolved_bets >= cfg.MIN_RESOLVED_BETS:
            if win_rate >= cfg.PERFECT_WIN_RATE:
                pts = 15
                signals.append(InsiderSignal(
                    "perfect_win_rate", "trading", pts,
                    f"100% win rate ({resolved_bets} resolved bets)",
                    win_rate
                ))
            elif win_rate >= cfg.STRONG_WIN_RATE:
                pts = 12
                signals.append(InsiderSignal(
                    "near_perfect_win_rate", "trading", pts,
                    f"{win_rate*100:.0f}% win rate (95%+)",
                    win_rate
                ))
            elif win_rate >= 0.85:
                pts = 8
                signals.append(InsiderSignal(
                    "high_win_rate", "trading", pts,
                    f"{win_rate*100:.0f}% win rate (85-95%)",
                    win_rate
                ))
            elif win_rate >= cfg.MIN_WIN_RATE:
                pts = 4
                signals.append(InsiderSignal(
                    "elevated_win_rate", "trading", pts,
                    f"{win_rate*100:.0f}% win rate (70-85%)",
                    win_rate
                ))
            else:
                pts = 0

            dimension_scores["trading"] += max(0, pts)

        # =====================================================================
        # DIMENSION 3: BEHAVIORAL SIGNALS (Max 25 pts)
        # =====================================================================

        # Market concentration
        if category_concentration >= cfg.SINGLE_MARKET_FOCUS:
            pts = 10
            signals.append(InsiderSignal(
                "single_market_focus", "behavioral", pts,
                f"{category_concentration*100:.0f}% in {primary_category} (single focus)",
                category_concentration
            ))
        elif category_concentration >= cfg.STRONG_CONCENTRATION:
            pts = 8
            signals.append(InsiderSignal(
                "high_concentration", "behavioral", pts,
                f"{category_concentration*100:.0f}% in {primary_category} (high concentration)",
                category_concentration
            ))
        elif category_concentration >= cfg.MIN_CONCENTRATION:
            pts = 5
            signals.append(InsiderSignal(
                "moderate_concentration", "behavioral", pts,
                f"{category_concentration*100:.0f}% in {primary_category} (moderate concentration)",
                category_concentration
            ))
        else:
            pts = 0

        dimension_scores["behavioral"] += max(0, pts)

        # No hedging bonus (insiders don't hedge)
        if not has_hedging:
            pts = 5
            signals.append(InsiderSignal(
                "no_hedging", "behavioral", pts,
                "No hedging positions detected (one-directional betting)",
                has_hedging
            ))
            dimension_scores["behavioral"] += pts

        # Off-hours trading (0-6 AM UTC)
        if total_trades > 0 and trades_in_off_hours > 0:
            off_hours_pct = trades_in_off_hours / total_trades
            if off_hours_pct >= 0.5:
                pts = 5
                signals.append(InsiderSignal(
                    "off_hours_trading", "behavioral", pts,
                    f"{off_hours_pct*100:.0f}% of trades in off-hours (0-6 AM UTC)",
                    off_hours_pct
                ))
                dimension_scores["behavioral"] += pts
            elif off_hours_pct >= 0.3:
                pts = 3
                signals.append(InsiderSignal(
                    "some_off_hours", "behavioral", pts,
                    f"{off_hours_pct*100:.0f}% of trades in off-hours",
                    off_hours_pct
                ))
                dimension_scores["behavioral"] += pts

        # Single market vs diversified
        if unique_markets <= 1:
            pts = 5
            signals.append(InsiderSignal(
                "single_market_only", "behavioral", pts,
                f"Only {unique_markets} market traded",
                unique_markets
            ))
            dimension_scores["behavioral"] += pts
        elif unique_markets <= 3:
            pts = 3
            signals.append(InsiderSignal(
                "very_few_markets", "behavioral", pts,
                f"Only {unique_markets} markets traded",
                unique_markets
            ))
            dimension_scores["behavioral"] += pts

        # =====================================================================
        # DIMENSION 4: CONTEXTUAL SIGNALS (Max 20 pts)
        # =====================================================================

        # Market category risk
        category_lower = primary_category.lower() if primary_category else "other"
        category_risk = cfg.CATEGORY_RISK.get(category_lower, 1)

        if category_risk >= 6:
            pts = category_risk
            signals.append(InsiderSignal(
                "high_risk_category", "contextual", pts,
                f"Trading in high-risk category: {primary_category}",
                category_risk
            ))
            dimension_scores["contextual"] += pts
        elif category_risk >= 4:
            pts = category_risk
            signals.append(InsiderSignal(
                "moderate_risk_category", "contextual", pts,
                f"Trading in moderate-risk category: {primary_category}",
                category_risk
            ))
            dimension_scores["contextual"] += pts

        # Profit relative to account age (efficiency signal)
        if account_age_days > 0 and total_pnl > 0:
            profit_per_day = total_pnl / account_age_days
            if profit_per_day >= 10000:  # $10K+ per day
                pts = 8
                signals.append(InsiderSignal(
                    "extreme_efficiency", "contextual", pts,
                    f"${profit_per_day:,.0f}/day profit efficiency",
                    profit_per_day
                ))
                dimension_scores["contextual"] += pts
            elif profit_per_day >= 2000:  # $2K+ per day
                pts = 5
                signals.append(InsiderSignal(
                    "high_efficiency", "contextual", pts,
                    f"${profit_per_day:,.0f}/day profit efficiency",
                    profit_per_day
                ))
                dimension_scores["contextual"] += pts

        # =====================================================================
        # AGGREGATE SCORING
        # =====================================================================

        # Calculate raw score (sum of dimensions, capped)
        raw_score = sum(dimension_scores.values())

        # Cap at 105 before normalization
        capped_score = min(raw_score, 105)

        # Normalize to 0-100
        normalized_score = (capped_score / 105) * 100

        # Count active dimensions (dimensions with score > 0)
        dimensions_active = sum(1 for v in dimension_scores.values() if v > 0)

        # Calculate confidence interval based on signal count
        signal_count = len(signals)
        if signal_count < 3:
            ci_width = 15
        elif signal_count < 5:
            ci_width = 10
        elif signal_count < 7:
            ci_width = 7
        else:
            ci_width = 5

        confidence_interval = (
            max(0, normalized_score - ci_width),
            min(100, normalized_score + ci_width)
        )

        # Determine priority with validation
        if normalized_score >= cfg.CRITICAL_SCORE and signal_count >= 5 and dimensions_active >= 3:
            priority = InsiderPriority.CRITICAL
        elif normalized_score >= cfg.HIGH_SCORE and signal_count >= 4 and dimensions_active >= 2:
            priority = InsiderPriority.HIGH
        elif normalized_score >= cfg.MEDIUM_SCORE and signal_count >= 3 and dimensions_active >= 2:
            priority = InsiderPriority.MEDIUM
        elif normalized_score >= cfg.MIN_INSIDER_SCORE and signal_count >= cfg.MIN_SIGNALS_FOR_FLAG:
            priority = InsiderPriority.LOW
        else:
            priority = InsiderPriority.NORMAL

        # Single-dimension downgrade (prevent false positives)
        if priority in [InsiderPriority.HIGH, InsiderPriority.CRITICAL] and dimensions_active < 2:
            priority = InsiderPriority.MEDIUM
            normalized_score = min(normalized_score, 69)

        return InsiderProbeResult(
            wallet_address=wallet,
            total_pnl=total_pnl,
            insider_score=round(normalized_score, 1),
            priority=priority.value,
            confidence_interval=confidence_interval,
            signals=[asdict(s) for s in signals],
            signal_count=signal_count,
            dimensions_active=dimensions_active,
            dimension_scores=dimension_scores,
            account_age_days=account_age_days,
            total_trades=total_trades,
            win_rate=win_rate,
            resolved_bets=resolved_bets,
            avg_entry_odds=avg_entry_odds,
            cumulative_position=cumulative_position,
            single_largest_position=single_largest_position,
            unique_markets=unique_markets,
            primary_category=primary_category or "Unknown",
            category_concentration=category_concentration,
            has_hedging=has_hedging,
            trades_in_last_24h=0,  # Will be set externally
            trades_in_off_hours=trades_in_off_hours,
        )


# =============================================================================
# FUNDING EXTRACTION
# =============================================================================

async def extract_funding_info(wallet: str, session: aiohttp.ClientSession, api_key: str = None) -> Optional[FundingInfo]:
    """Extract funding source from Etherscan V2 API (covers Polygon).

    Requires ETHERSCAN_API_KEY or POLYGONSCAN_API_KEY environment variable.
    Get a free key at: https://etherscan.io/myapikey
    """
    import os
    api_key = api_key or os.getenv("ETHERSCAN_API_KEY") or os.getenv("POLYGONSCAN_API_KEY")

    if not api_key:
        return None  # Silently skip if no API key

    try:
        # Etherscan V2 API for Polygon token transfers
        url = "https://api.etherscan.io/v2/api"
        params = {
            "chainid": 137,  # Polygon mainnet
            "module": "account",
            "action": "tokentx",
            "address": wallet,
            "contractaddress": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",  # USDC on Polygon
            "page": 1,
            "offset": 100,
            "sort": "asc",  # Oldest first to find initial funding
            "apikey": api_key,
        }

        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return None

            data = await resp.json()
            if data.get("status") != "1" or not data.get("result"):
                # Try native MATIC transfers as fallback
                params["action"] = "txlist"
                del params["contractaddress"]
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp2:
                    data = await resp2.json()
                    if data.get("status") != "1" or not data.get("result"):
                        return None

            transfers = data["result"]
            funding_info = FundingInfo()

            # Find first significant inbound transfer (funding source)
            for tx in transfers:
                to_addr = tx.get("to", "").lower()
                if to_addr == wallet.lower():
                    # Check for USDC transfers (6 decimals) or native MATIC (18 decimals)
                    value = tx.get("value", "0")
                    decimals = 6 if "tokenDecimal" in tx else 18
                    amount = float(value) / (10 ** decimals)

                    if amount >= 100:  # Minimum $100 to count as funding
                        funding_info.funding_source = tx.get("from", "").lower()
                        funding_info.funding_amount = amount
                        funding_info.funding_tx = tx.get("hash")
                        break

            # Find last significant outbound transfer (withdrawal dest)
            for tx in reversed(transfers):
                from_addr = tx.get("from", "").lower()
                if from_addr == wallet.lower():
                    value = tx.get("value", "0")
                    decimals = 6 if "tokenDecimal" in tx else 18
                    amount = float(value) / (10 ** decimals)

                    if amount >= 100:
                        funding_info.withdrawal_dest = tx.get("to", "").lower()
                        funding_info.withdrawal_amount = amount
                        funding_info.withdrawal_tx = tx.get("hash")
                        break

            return funding_info

    except Exception as e:
        return None


# =============================================================================
# MAIN PROBE LOGIC
# =============================================================================

async def probe_account(
    wallet: str,
    leaderboard_pnl: float,
    analyzer: AccountAnalyzer,
    scorer: InsiderScorer,
    session: aiohttp.ClientSession,
    stats: ProbeStats,
) -> Optional[InsiderProbeResult]:
    """Probe a single account for insider signals."""
    try:
        # Run deep analysis (fetch trades, compute metrics)
        result = await analyzer.deep_analysis(
            wallet_address=wallet,
            lookback_days=365,
            include_insider_signals=True,
            max_trades=1000,  # Enough to detect patterns
        )

        if result.error or not result.pl_metrics or not result.pattern_metrics:
            return None

        pl = result.pl_metrics
        pm = result.pattern_metrics

        # Extract key metrics for scoring
        account_age_days = pm.account_age_days if pm.account_age_days else 0
        total_trades = pm.total_trades if pm.total_trades else 0

        # Win rate calculation - use market_win_rate which includes redemptions
        # market_win_rate = markets_profitable / (markets_profitable + markets_unprofitable)
        # This is more accurate than sell-based win_rate which misses held positions
        markets_resolved = pl.markets_profitable + getattr(pl, 'markets_unprofitable', 0)
        if markets_resolved > 0:
            win_rate = pl.market_win_rate
            resolved_bets = markets_resolved
        else:
            # Fallback to sell-based win rate if no market-level data
            resolved_bets = pl.win_count + pl.loss_count
            win_rate = pl.win_count / resolved_bets if resolved_bets > 0 else 0

        # Entry odds (average)
        avg_entry_odds = float(pm.avg_entry_odds) if pm.avg_entry_odds else 0.5

        # Position sizing
        cumulative_position = float(pl.total_realized_pnl) if pl.total_realized_pnl else 0
        single_largest_position = float(pm.max_position_size_usd) if pm.max_position_size_usd else 0

        # Market concentration
        unique_markets = pm.unique_markets_traded if pm.unique_markets_traded else 0
        primary_category = pm.primary_category if pm.primary_category else "Unknown"
        category_concentration = pm.category_concentration if pm.category_concentration else 0

        # Hedging detection (simplified - check if they have both YES and NO)
        has_hedging = False  # Would need position analysis

        # Off-hours trading (estimate from pattern metrics)
        trades_in_off_hours = 0  # Would need trade timestamp analysis

        # Score the account
        probe_result = scorer.score_account(
            wallet=wallet,
            account_age_days=account_age_days,
            total_trades=total_trades,
            win_rate=win_rate,
            resolved_bets=resolved_bets,
            avg_entry_odds=avg_entry_odds,
            cumulative_position=abs(cumulative_position),
            single_largest_position=single_largest_position,
            unique_markets=unique_markets,
            primary_category=primary_category,
            category_concentration=category_concentration,
            has_hedging=has_hedging,
            trades_in_off_hours=trades_in_off_hours,
            total_pnl=leaderboard_pnl,
        )

        # Only extract funding for flagged accounts
        if probe_result.priority != InsiderPriority.NORMAL.value:
            funding_info = await extract_funding_info(wallet, session)
            if funding_info:
                probe_result.funding_info = asdict(funding_info)
                stats.funding_extracted += 1

        return probe_result

    except Exception as e:
        stats.errors += 1
        return None


async def main():
    """Main probe execution."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Insider Probe - Detect insider trading patterns")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Historical detection mode: bypasses account age filter since historical insiders were new when they traded but are now old"
    )
    args = parser.parse_args()

    config = InsiderConfig()
    historical_mode = args.historical

    output_file = PROJECT_ROOT / "data" / "insider_probe_results.json"
    funding_index_file = PROJECT_ROOT / "data" / "insider_funding_sources.json"

    print("=" * 70)
    print("INSIDER PROBE - Scanning for Probable Insider Traders")
    if historical_mode:
        print("MODE: HISTORICAL (account age check BYPASSED)")
    else:
        print("MODE: LIVE (account age check ACTIVE)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Min Profit: ${config.MIN_PROFIT:,}")
    print(f"  Max Accounts to Scan: {config.MAX_ACCOUNTS_TO_SCAN:,}")
    print(f"  Min Insider Score: {config.MIN_INSIDER_SCORE}")
    print(f"  Min Signals Required: {config.MIN_SIGNALS_FOR_FLAG}")
    if historical_mode:
        print(f"  Account Age Check: BYPASSED (historical mode)")
    print()

    stats = ProbeStats(start_time=datetime.now())
    scorer = InsiderScorer(config, historical_mode=historical_mode)

    flagged_accounts: List[InsiderProbeResult] = []
    funding_sources: Dict[str, List[str]] = defaultdict(list)  # source -> [wallets]

    async with aiohttp.ClientSession() as session:
        async with LeaderboardClient() as leaderboard_client:
            async with AccountAnalyzer(mode=DiscoveryMode.WIDE_NET_PROFITABILITY) as analyzer:

                # Step 1: Fetch leaderboard accounts from ALL categories
                print("Step 1: Fetching leaderboard accounts from all categories...")

                leaderboard_entries = []
                categories = ["OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE",
                             "WEATHER", "ECONOMICS", "TECH", "FINANCE"]

                for category in categories:
                    try:
                        print(f"  Fetching {category}...")
                        entries = await leaderboard_client.get_leaderboard_full(
                            category=category,
                            limit=config.MAX_ACCOUNTS_TO_SCAN // len(categories) + 200,
                        )
                        for entry in entries:
                            pnl = float(entry.total_pnl) if entry.total_pnl else 0
                            if pnl >= config.MIN_PROFIT and pnl <= config.MAX_PROFIT:
                                leaderboard_entries.append((entry.wallet_address, pnl, entry.num_trades))
                        print(f"    Got {len(entries)} from {category}")
                    except Exception as e:
                        print(f"  Error fetching {category}: {e}")

                # Dedupe and apply initial filters
                seen = set()
                unique_entries = []
                for wallet, pnl, num_trades in leaderboard_entries:
                    if wallet.lower() not in seen:
                        seen.add(wallet.lower())
                        unique_entries.append((wallet, pnl))

                print(f"  Found {len(unique_entries):,} unique accounts meeting profit criteria")

                # Step 2: Probe each account
                print(f"\nStep 2: Probing accounts for insider signals...")

                semaphore = asyncio.Semaphore(3)  # Conservative rate limiting

                async def probe_one(wallet: str, pnl: float) -> Optional[InsiderProbeResult]:
                    async with semaphore:
                        stats.total_scanned += 1
                        result = await probe_account(wallet, pnl, analyzer, scorer, session, stats)

                        if result and result.priority != InsiderPriority.NORMAL.value:
                            stats.passed_initial_filter += 1

                            if result.priority == InsiderPriority.CRITICAL.value:
                                stats.flagged_critical += 1
                            elif result.priority == InsiderPriority.HIGH.value:
                                stats.flagged_high += 1
                            elif result.priority == InsiderPriority.MEDIUM.value:
                                stats.flagged_medium += 1
                            elif result.priority == InsiderPriority.LOW.value:
                                stats.flagged_low += 1

                        return result

                # Process in batches for progress reporting
                batch_size = 25
                all_results = []

                for i in range(0, len(unique_entries), batch_size):
                    batch = unique_entries[i:i + batch_size]
                    tasks = [probe_one(w, p) for w, p in batch]
                    results = await asyncio.gather(*tasks)

                    for r in results:
                        if r and r.priority != InsiderPriority.NORMAL.value:
                            all_results.append(r)

                            # Track funding sources
                            if r.funding_info and r.funding_info.get("funding_source"):
                                funding_sources[r.funding_info["funding_source"]].append(r.wallet_address)

                    # Progress update
                    pct = (stats.total_scanned / len(unique_entries)) * 100
                    elapsed = (datetime.now() - stats.start_time).total_seconds()
                    rate = stats.total_scanned / elapsed if elapsed > 0 else 0
                    eta = (len(unique_entries) - stats.total_scanned) / rate if rate > 0 else 0

                    print(f"  [{stats.total_scanned}/{len(unique_entries)} ({pct:.1f}%)] "
                          f"Flagged: {stats.total_flagged} (C:{stats.flagged_critical} H:{stats.flagged_high} "
                          f"M:{stats.flagged_medium} L:{stats.flagged_low}) | "
                          f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")

                flagged_accounts = all_results

    # Sort by score (highest first)
    flagged_accounts.sort(key=lambda x: x.insider_score, reverse=True)

    # Step 3: Save results
    print(f"\nStep 3: Saving results...")

    output_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "mode": "historical" if historical_mode else "live",
        "config": {
            "min_profit": config.MIN_PROFIT,
            "max_accounts_scanned": config.MAX_ACCOUNTS_TO_SCAN,
            "min_insider_score": config.MIN_INSIDER_SCORE,
            "min_signals": config.MIN_SIGNALS_FOR_FLAG,
            "historical_mode": historical_mode,
            "account_age_check": "bypassed" if historical_mode else "active",
        },
        "stats": {
            "total_scanned": stats.total_scanned,
            "total_flagged": stats.total_flagged,
            "critical": stats.flagged_critical,
            "high": stats.flagged_high,
            "medium": stats.flagged_medium,
            "low": stats.flagged_low,
            "funding_extracted": stats.funding_extracted,
            "errors": stats.errors,
            "duration_seconds": (datetime.now() - stats.start_time).total_seconds(),
        },
        "priority_distribution": {
            "critical": [a.to_dict() for a in flagged_accounts if a.priority == "critical"],
            "high": [a.to_dict() for a in flagged_accounts if a.priority == "high"],
            "medium": [a.to_dict() for a in flagged_accounts if a.priority == "medium"],
            "low": [a.to_dict() for a in flagged_accounts if a.priority == "low"],
        },
        "accounts": [a.to_dict() for a in flagged_accounts],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"  Results saved to: {output_file}")

    # Save funding source index (for sybil detection)
    funding_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "sources": {
            source: {
                "funded_wallets": wallets,
                "count": len(wallets),
            }
            for source, wallets in funding_sources.items()
            if len(wallets) >= 1  # Keep all for now, filter clusters later
        }
    }

    with open(funding_index_file, "w") as f:
        json.dump(funding_data, f, indent=2)

    print(f"  Funding index saved to: {funding_index_file}")

    # Summary
    print(f"\n{'=' * 70}")
    print("PROBE COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nResults Summary:")
    print(f"  Total Scanned:  {stats.total_scanned:,}")
    print(f"  Total Flagged:  {stats.total_flagged:,} ({stats.total_flagged/max(1,stats.total_scanned)*100:.1f}%)")
    print(f"    CRITICAL:     {stats.flagged_critical}")
    print(f"    HIGH:         {stats.flagged_high}")
    print(f"    MEDIUM:       {stats.flagged_medium}")
    print(f"    LOW:          {stats.flagged_low}")
    print(f"  Funding Extracted: {stats.funding_extracted}")
    print(f"  Unique Funding Sources: {len(funding_sources)}")
    print(f"  Duration: {(datetime.now() - stats.start_time).total_seconds()/60:.1f} minutes")

    # Show top flagged accounts
    if flagged_accounts:
        print(f"\nTop 10 Flagged Accounts:")
        print("-" * 70)
        for i, acc in enumerate(flagged_accounts[:10]):
            print(f"  {i+1}. {acc.wallet_address[:16]}... | "
                  f"Score: {acc.insider_score:.0f} ({acc.priority.upper()}) | "
                  f"P/L: ${acc.total_pnl:,.0f} | "
                  f"Signals: {acc.signal_count} | "
                  f"Age: {acc.account_age_days}d")


if __name__ == "__main__":
    asyncio.run(main())
