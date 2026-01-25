"""Insider Scoring Engine.

Implements the 5-dimension scoring system with:
- Variance-calibrated thresholds (soft edges, not hard cutoffs)
- Cumulative position tracking
- Minimum signal count validation
- Confidence intervals

Based on analysis of 17 documented insider cases.
See: docs/insider-scanner/VARIANCE-CALIBRATION.md
See: docs/insider-scanner/SCORING-SYSTEM.md
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import math

from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS - From Variance Calibration Analysis
# =============================================================================

# Position size thresholds (observed: $1,167 to $340M, median $25K-$40K)
POSITION_SIZE_THRESHOLDS = {
    "micro": 5_000,       # < $5K - low signal
    "small": 20_000,      # $5K-$20K - common insider range
    "medium": 100_000,    # $20K-$100K - elevated
    "large": 1_000_000,   # $100K-$1M - high
    "whale": 10_000_000,  # > $1M - very high
}

# Entry odds thresholds (observed: 0.2% to 71%, sweet spot 4%-22%)
ENTRY_ODDS_THRESHOLDS = {
    "extreme_longshot": 0.05,   # < 5%
    "longshot": 0.10,           # 5-10%
    "moderate_long": 0.20,      # 10-20%
    "moderate": 0.35,           # 20-35%
    "consensus": 0.60,          # 35-60%
    "high_prob": 1.0,           # > 60%
}

# Win rate thresholds (observed: 80% to 100%)
WIN_RATE_THRESHOLDS = {
    "perfect": 1.0,      # 100%
    "near_perfect": 0.95, # 95-99%
    "excellent": 0.90,   # 90-95%
    "very_good": 0.80,   # 80-90%
    "good": 0.70,        # 70-80%
}

# Account age thresholds (observed: 2 hours to months)
ACCOUNT_AGE_THRESHOLDS = {
    "brand_new": 1,       # < 1 day
    "very_fresh": 7,      # 1-7 days
    "fresh": 14,          # 7-14 days
    "recent": 30,         # 14-30 days
    "established": 90,    # 30-90 days
}


class MarketCategory(str, Enum):
    """Market categories with associated risk levels."""
    MILITARY = "military"           # Highest risk (8 pts)
    GOVERNMENT_POLICY = "policy"    # Very high risk (7 pts)
    ELECTION = "election"           # High risk (6 pts)
    CORPORATE = "corporate"         # Medium-high risk (5 pts)
    AWARDS = "awards"               # Medium-high risk (5 pts)
    SPORTS = "sports"               # Medium risk (4 pts)
    TECH_LAUNCH = "tech_launch"     # Medium risk (4 pts)
    SOCIAL_MEDIA = "social"         # Low risk (2 pts)
    OTHER = "other"                 # No risk (0 pts)


@dataclass
class Signal:
    """A single detection signal."""
    name: str
    category: str  # account, trading, behavioral, contextual, cluster
    weight: float
    raw_value: Any = None
    threshold: Any = None
    description: str = ""


@dataclass
class ScoringResult:
    """Result of scoring a wallet."""
    score: float
    confidence_low: float
    confidence_high: float
    priority: str
    dimensions: Dict[str, float]
    signals: List[Signal]
    signal_count: int
    active_dimensions: int
    downgraded: bool
    downgrade_reason: Optional[str]


class InsiderScorer:
    """Scores wallets for insider trading likelihood.

    Uses variance-calibrated scoring with:
    - Soft thresholds (gradients, not hard cutoffs)
    - Minimum signal count requirements
    - Confidence intervals
    - Cumulative position analysis
    """

    # Dimension maximums
    MAX_ACCOUNT = 25
    MAX_TRADING = 35
    MAX_BEHAVIORAL = 25
    MAX_CONTEXTUAL = 20
    MAX_CLUSTER = 20

    # Minimum requirements for each priority level
    # MEDIUM threshold lowered from 55 to 50 based on Fed Chair cluster
    # analysis - ensures we catch insiders using accumulation patterns
    MIN_REQUIREMENTS = {
        "critical": {"score": 85, "signals": 5, "dimensions": 3},
        "high": {"score": 70, "signals": 4, "dimensions": 2},
        "medium": {"score": 50, "signals": 3, "dimensions": 2},
        "low": {"score": 40, "signals": 2, "dimensions": 1},
    }

    def __init__(self, variance_factor: float = 1.2):
        """Initialize scorer.

        Args:
            variance_factor: Multiplier to widen threshold bands (default 1.2 = 20% wider)
        """
        self.variance_factor = variance_factor

    def score_wallet(
        self,
        wallet_address: str,
        account_age_days: Optional[int] = None,
        transaction_count: Optional[int] = None,
        positions: Optional[List[Dict]] = None,
        trades: Optional[List[Dict]] = None,
        market_category: Optional[MarketCategory] = None,
        event_hours_away: Optional[float] = None,
        funding_source: Optional[str] = None,
        flagged_funders: Optional[set] = None,
        cluster_wallets: Optional[List[str]] = None,
    ) -> ScoringResult:
        """Score a wallet for insider trading likelihood.

        Args:
            wallet_address: The wallet to score
            account_age_days: Days since account creation
            transaction_count: Total historical transactions
            positions: List of position dicts with market_id, side, size, entry_odds
            trades: List of trade dicts with timestamp, size, market_id
            market_category: Category of primary market
            event_hours_away: Hours until market resolution
            funding_source: Address that funded this wallet
            flagged_funders: Set of known flagged funding sources
            cluster_wallets: Other wallets in same cluster

        Returns:
            ScoringResult with score, confidence interval, signals
        """
        signals: List[Signal] = []

        # Dimension 1: Account Signals
        account_score = self._score_account(
            account_age_days, transaction_count, signals
        )

        # Dimension 2: Trading Signals
        trading_score = self._score_trading(
            positions, trades, signals
        )

        # Dimension 3: Behavioral Signals
        behavioral_score = self._score_behavioral(
            positions, trades, signals
        )

        # Dimension 4: Contextual Signals
        contextual_score = self._score_contextual(
            market_category, event_hours_away, signals
        )

        # Dimension 5: Cluster Signals
        cluster_score = self._score_cluster(
            funding_source, flagged_funders, cluster_wallets, signals
        )

        # Aggregate dimensions
        dimensions = {
            "account": min(account_score, self.MAX_ACCOUNT),
            "trading": min(trading_score, self.MAX_TRADING),
            "behavioral": min(behavioral_score, self.MAX_BEHAVIORAL),
            "contextual": min(contextual_score, self.MAX_CONTEXTUAL),
            "cluster": min(cluster_score, self.MAX_CLUSTER),
        }

        # Calculate composite score
        base_score = (
            dimensions["account"] +
            dimensions["trading"] +
            dimensions["behavioral"] +
            dimensions["contextual"]
        )

        # Add cluster as bonus (can push higher)
        cluster_bonus = dimensions["cluster"] * 0.5

        # Normalize to 0-100 (base max = 105)
        normalized = (base_score / 105) * 100

        # Add cluster bonus
        raw_score = normalized + cluster_bonus

        # Count signals and active dimensions
        signal_count = len(signals)
        active_dimensions = sum(1 for d, s in dimensions.items() if s > 0)

        # Calculate confidence interval based on signal count
        confidence_width = self._calculate_confidence_width(signal_count)

        # VARIANCE: Single-dimension downgrade check
        downgraded = False
        downgrade_reason = None

        if raw_score >= 70 and active_dimensions < 2:
            raw_score = min(raw_score, 69)
            downgraded = True
            downgrade_reason = "Single-dimension flag, downgraded to MEDIUM"

        # Clamp score
        final_score = min(max(raw_score, 0), 100)

        # Determine priority
        priority = self._get_priority(final_score, signal_count, active_dimensions)

        return ScoringResult(
            score=round(final_score, 2),
            confidence_low=round(max(final_score - confidence_width, 0), 2),
            confidence_high=round(min(final_score + confidence_width, 100), 2),
            priority=priority,
            dimensions=dimensions,
            signals=signals,
            signal_count=signal_count,
            active_dimensions=active_dimensions,
            downgraded=downgraded,
            downgrade_reason=downgrade_reason,
        )

    def _score_account(
        self,
        account_age_days: Optional[int],
        transaction_count: Optional[int],
        signals: List[Signal],
    ) -> float:
        """Score account characteristics."""
        score = 0.0

        # Account age (soft gradient scoring)
        if account_age_days is not None:
            age_score = self._score_account_age(account_age_days)
            score += age_score
            if age_score > 0:
                signals.append(Signal(
                    name="account_age",
                    category="account",
                    weight=age_score,
                    raw_value=account_age_days,
                    threshold=ACCOUNT_AGE_THRESHOLDS,
                    description=f"Account {account_age_days} days old"
                ))

        # Transaction history (soft gradient)
        if transaction_count is not None:
            tx_score = self._score_transaction_count(transaction_count)
            score += tx_score
            if tx_score > 0:
                signals.append(Signal(
                    name="transaction_count",
                    category="account",
                    weight=tx_score,
                    raw_value=transaction_count,
                    description=f"Only {transaction_count} prior transactions"
                ))

        return score

    def _score_account_age(self, days: int) -> float:
        """Soft gradient scoring for account age.

        Peak at very fresh, gradual decline to established.
        """
        if days < 1:
            return 15.0  # Brand new
        elif days < 7:
            # Linear decay from 15 to 12 over 6 days
            return 15.0 - (3.0 * (days / 7))
        elif days < 14:
            # Decay from 12 to 8
            return 12.0 - (4.0 * ((days - 7) / 7))
        elif days < 30:
            # Decay from 8 to 4
            return 8.0 - (4.0 * ((days - 14) / 16))
        elif days < 90:
            # Decay from 4 to 0
            return 4.0 - (4.0 * ((days - 30) / 60))
        else:
            return 0.0  # Established account

    def _score_transaction_count(self, count: int) -> float:
        """Soft gradient scoring for transaction count."""
        if count == 0:
            return 10.0
        elif count <= 2:
            return 8.0
        elif count <= 5:
            return 5.0
        elif count <= 10:
            return 2.0
        else:
            return 0.0

    def _score_trading(
        self,
        positions: Optional[List[Dict]],
        trades: Optional[List[Dict]],
        signals: List[Signal],
    ) -> float:
        """Score trading behavior."""
        score = 0.0

        if not positions:
            return score

        # Calculate cumulative position data
        cumulative_data = self._calculate_cumulative_positions(positions)

        # Position size score (using cumulative)
        if cumulative_data["dominant_size"] > 0:
            size_score = self._score_position_size(cumulative_data["dominant_size"])
            score += size_score
            if size_score > 0:
                signals.append(Signal(
                    name="position_size_cumulative",
                    category="trading",
                    weight=size_score,
                    raw_value=cumulative_data["dominant_size"],
                    description=f"Cumulative position ${cumulative_data['dominant_size']:,.0f}"
                ))

            # Split entry bonus
            if cumulative_data["is_split_entry"]:
                score += 2
                signals.append(Signal(
                    name="split_entry_pattern",
                    category="trading",
                    weight=2,
                    raw_value=cumulative_data["entry_count"],
                    description=f"Split entry detected ({cumulative_data['entry_count']} entries)"
                ))

        # Entry odds score
        avg_odds = self._calculate_avg_entry_odds(positions)
        if avg_odds is not None:
            odds_score = self._score_entry_odds(avg_odds)
            score += odds_score
            if odds_score > 0:
                signals.append(Signal(
                    name="entry_odds",
                    category="trading",
                    weight=odds_score,
                    raw_value=avg_odds,
                    description=f"Average entry odds {avg_odds*100:.1f}%"
                ))

        # Win rate score
        win_rate = self._calculate_win_rate(positions)
        if win_rate is not None:
            wr_score = self._score_win_rate(win_rate)
            score += wr_score
            if wr_score > 0:
                signals.append(Signal(
                    name="win_rate",
                    category="trading",
                    weight=wr_score,
                    raw_value=win_rate,
                    description=f"Win rate {win_rate*100:.0f}%"
                ))

        return score

    def _calculate_cumulative_positions(self, positions: List[Dict]) -> Dict:
        """Calculate cumulative position data across all markets.

        Returns dict with:
        - dominant_size: Largest cumulative position in USD
        - entry_count: Number of entries for dominant position
        - is_split_entry: Whether entries were split (evasion)
        - avg_entry_size: Average size per entry
        """
        if not positions:
            return {
                "dominant_size": 0,
                "entry_count": 0,
                "is_split_entry": False,
                "avg_entry_size": 0,
            }

        # Group by market + side
        market_positions: Dict[str, Dict] = {}

        for pos in positions:
            key = f"{pos.get('market_id', '')}:{pos.get('side', '')}"
            if key not in market_positions:
                market_positions[key] = {
                    "total_usd": 0,
                    "entry_count": 0,
                    "entries": [],
                }

            size = float(pos.get("size_usd", pos.get("size", 0)))
            market_positions[key]["total_usd"] += size
            market_positions[key]["entry_count"] += 1
            market_positions[key]["entries"].append(size)

        # Find dominant position
        if not market_positions:
            return {
                "dominant_size": 0,
                "entry_count": 0,
                "is_split_entry": False,
                "avg_entry_size": 0,
            }

        dominant_key = max(market_positions.keys(), key=lambda k: market_positions[k]["total_usd"])
        dominant = market_positions[dominant_key]

        # Calculate split entry detection
        avg_entry = dominant["total_usd"] / dominant["entry_count"] if dominant["entry_count"] > 0 else 0
        is_split = (
            dominant["entry_count"] > 1 and
            avg_entry < dominant["total_usd"] * 0.5
        )

        return {
            "dominant_size": dominant["total_usd"],
            "entry_count": dominant["entry_count"],
            "is_split_entry": is_split,
            "avg_entry_size": avg_entry,
        }

    def _score_position_size(self, size: float) -> float:
        """Score based on cumulative position size."""
        # Apply variance factor to thresholds
        micro = POSITION_SIZE_THRESHOLDS["micro"] * self.variance_factor
        small = POSITION_SIZE_THRESHOLDS["small"] * self.variance_factor
        medium = POSITION_SIZE_THRESHOLDS["medium"] * self.variance_factor
        large = POSITION_SIZE_THRESHOLDS["large"] * self.variance_factor

        if size < micro:
            return 0
        elif size < small:
            return 4  # Small signal
        elif size < medium:
            return 7  # Medium signal
        elif size < large:
            return 10  # Large signal
        else:
            return 12  # Whale signal

    def _calculate_avg_entry_odds(self, positions: List[Dict]) -> Optional[float]:
        """Calculate weighted average entry odds."""
        if not positions:
            return None

        total_value = 0
        total_weight = 0

        for pos in positions:
            odds = pos.get("entry_odds")
            size = pos.get("size_usd", pos.get("size", 1))

            if odds is not None:
                total_value += float(odds) * float(size)
                total_weight += float(size)

        if total_weight == 0:
            return None

        return total_value / total_weight

    def _score_entry_odds(self, odds: float) -> float:
        """Score based on entry odds (with variance).

        The 10-30% range is the "insider sweet spot" - low enough to be
        profitable but not so extreme it's obvious. Increased scoring
        for this range based on Fed Chair cluster analysis (Jan 2026).
        """
        if odds < 0.05:
            return 8  # Extreme longshot
        elif odds < 0.10:
            return 7  # Longshot (increased from 6)
        elif odds < 0.20:
            return 6  # Insider sweet spot low (increased from 4)
        elif odds < 0.30:
            return 5  # Insider sweet spot high (increased from 2)
        elif odds < 0.45:
            return 2  # Moderate
        elif odds < 0.60:
            return 1  # Consensus (variance allowance for GayPride case)
        else:
            return 0  # High probability

    def _calculate_win_rate(self, positions: List[Dict]) -> Optional[float]:
        """Calculate win rate from resolved positions."""
        resolved = [p for p in positions if p.get("resolved")]
        if len(resolved) < 3:  # Minimum trades for win rate
            return None

        wins = sum(1 for p in resolved if p.get("won"))
        return wins / len(resolved)

    def _score_win_rate(self, win_rate: float) -> float:
        """Score based on win rate (min 3 resolved trades)."""
        if win_rate >= 1.0:
            return 15  # Perfect
        elif win_rate >= 0.95:
            return 12  # Near-perfect
        elif win_rate >= 0.90:
            return 10  # Excellent
        elif win_rate >= 0.80:
            return 8  # Very good (Annica case)
        elif win_rate >= 0.70:
            return 4  # Good
        else:
            return 0

    def _score_behavioral(
        self,
        positions: Optional[List[Dict]],
        trades: Optional[List[Dict]],
        signals: List[Signal],
    ) -> float:
        """Score behavioral patterns."""
        score = 0.0

        if positions:
            # Market concentration
            concentration = self._calculate_market_concentration(positions)
            conc_score = self._score_concentration(concentration)
            score += conc_score
            if conc_score > 0:
                signals.append(Signal(
                    name="market_concentration",
                    category="behavioral",
                    weight=conc_score,
                    raw_value=concentration,
                    description=f"{concentration*100:.0f}% in single market"
                ))

            # No hedging check
            has_hedging = self._check_hedging(positions)
            if not has_hedging:
                score += 5
                signals.append(Signal(
                    name="no_hedging",
                    category="behavioral",
                    weight=5,
                    description="No hedging positions detected"
                ))

        if trades:
            # Off-hours trading
            off_hours_pct = self._calculate_off_hours_pct(trades)
            if off_hours_pct > 0.5:  # >50% off-hours
                score += 5
                signals.append(Signal(
                    name="off_hours_trading",
                    category="behavioral",
                    weight=5,
                    raw_value=off_hours_pct,
                    description=f"{off_hours_pct*100:.0f}% trades during off-hours (0-6 AM UTC)"
                ))

        return score

    def _calculate_market_concentration(self, positions: List[Dict]) -> float:
        """Calculate what % of value is in single market."""
        if not positions:
            return 0.0

        market_values: Dict[str, float] = {}
        total_value = 0

        for pos in positions:
            market_id = pos.get("market_id", "unknown")
            value = float(pos.get("size_usd", pos.get("size", 0)))
            market_values[market_id] = market_values.get(market_id, 0) + value
            total_value += value

        if total_value == 0:
            return 0.0

        max_market_value = max(market_values.values()) if market_values else 0
        return max_market_value / total_value

    def _score_concentration(self, concentration: float) -> float:
        """Score based on market concentration."""
        if concentration >= 1.0:
            return 10  # Single market only
        elif concentration >= 0.90:
            return 8  # >90% one market
        elif concentration >= 0.80:
            return 5  # >80% one market
        elif concentration >= 0.50:
            return 2  # >50% one market
        else:
            return 0

    def _check_hedging(self, positions: List[Dict]) -> bool:
        """Check if wallet has hedging positions (both YES and NO on same market)."""
        market_sides: Dict[str, set] = {}

        for pos in positions:
            market_id = pos.get("market_id", "")
            side = pos.get("side", "")
            if market_id:
                if market_id not in market_sides:
                    market_sides[market_id] = set()
                market_sides[market_id].add(side)

        # Hedging = having both YES and NO on same market
        for sides in market_sides.values():
            if "YES" in sides and "NO" in sides:
                return True
        return False

    def _calculate_off_hours_pct(self, trades: List[Dict]) -> float:
        """Calculate % of trades during off-hours (0-6 AM UTC)."""
        if not trades:
            return 0.0

        off_hours_count = 0
        for trade in trades:
            timestamp = trade.get("timestamp")
            if timestamp:
                if isinstance(timestamp, datetime):
                    hour = timestamp.hour
                elif isinstance(timestamp, (int, float)):
                    hour = datetime.fromtimestamp(timestamp).hour
                else:
                    continue

                if 0 <= hour <= 6:
                    off_hours_count += 1

        return off_hours_count / len(trades)

    def _score_contextual(
        self,
        market_category: Optional[MarketCategory],
        event_hours_away: Optional[float],
        signals: List[Signal],
    ) -> float:
        """Score contextual factors."""
        score = 0.0

        # Market category risk
        if market_category:
            cat_score = self._score_market_category(market_category)
            score += cat_score
            if cat_score > 0:
                signals.append(Signal(
                    name="market_category",
                    category="contextual",
                    weight=cat_score,
                    raw_value=market_category.value,
                    description=f"High-risk market category: {market_category.value}"
                ))

        # Event timing
        if event_hours_away is not None:
            timing_score = self._score_event_timing(event_hours_away)
            score += timing_score
            if timing_score > 0:
                signals.append(Signal(
                    name="event_timing",
                    category="contextual",
                    weight=timing_score,
                    raw_value=event_hours_away,
                    description=f"Position placed {event_hours_away:.1f}h before event"
                ))

        return score

    def _score_market_category(self, category: MarketCategory) -> float:
        """Score based on market category risk level.

        GOVERNMENT_POLICY increased to 10 (from 7) based on Fed Chair
        cluster analysis - these markets have highest insider risk.
        """
        category_scores = {
            MarketCategory.MILITARY: 10,          # Increased from 8
            MarketCategory.GOVERNMENT_POLICY: 10, # Increased from 7
            MarketCategory.ELECTION: 7,           # Increased from 6
            MarketCategory.CORPORATE: 5,
            MarketCategory.AWARDS: 5,
            MarketCategory.SPORTS: 4,
            MarketCategory.TECH_LAUNCH: 4,
            MarketCategory.SOCIAL_MEDIA: 2,
            MarketCategory.OTHER: 0,
        }
        return category_scores.get(category, 0)

    def _score_event_timing(self, hours_away: float) -> float:
        """Score based on how close to event resolution."""
        if hours_away < 6:
            return 8  # < 6 hours
        elif hours_away < 24:
            return 6  # 6-24 hours
        elif hours_away < 72:
            return 4  # 24-72 hours
        elif hours_away < 168:  # 1 week
            return 2
        else:
            return 0

    def _score_cluster(
        self,
        funding_source: Optional[str],
        flagged_funders: Optional[set],
        cluster_wallets: Optional[List[str]],
        signals: List[Signal],
    ) -> float:
        """Score cluster/sybil indicators."""
        score = 0.0

        # Flagged funding source match (INSTANT HIGH SCORE)
        if funding_source and flagged_funders and funding_source.lower() in flagged_funders:
            score += 15
            signals.append(Signal(
                name="flagged_funding_source",
                category="cluster",
                weight=15,
                raw_value=funding_source,
                description="Funded by known suspicious source"
            ))

        # Cluster membership
        if cluster_wallets and len(cluster_wallets) > 0:
            cluster_size = len(cluster_wallets)
            cluster_score = min(cluster_size * 3, 10)  # Max 10 for cluster
            score += cluster_score
            signals.append(Signal(
                name="cluster_membership",
                category="cluster",
                weight=cluster_score,
                raw_value=cluster_size,
                description=f"Part of {cluster_size}-wallet cluster"
            ))

        return score

    def _calculate_confidence_width(self, signal_count: int) -> float:
        """Calculate confidence interval width based on signal count.

        More signals = narrower interval = higher confidence.
        """
        if signal_count < 3:
            return 15  # Wide interval
        elif signal_count < 5:
            return 10  # Moderate interval
        elif signal_count < 7:
            return 7  # Narrower
        else:
            return 5  # Tight interval

    def _get_priority(
        self,
        score: float,
        signal_count: int,
        active_dimensions: int,
    ) -> str:
        """Determine priority level based on score AND minimum requirements."""
        # Check from highest to lowest
        for priority, reqs in self.MIN_REQUIREMENTS.items():
            if (score >= reqs["score"] and
                signal_count >= reqs["signals"] and
                active_dimensions >= reqs["dimensions"]):
                return priority

        return "normal"

    # =============================================================================
    # FALSE POSITIVE REDUCTION
    # =============================================================================

    def apply_false_positive_modifiers(
        self,
        result: ScoringResult,
        bet_lost: bool = False,
        long_diverse_history: bool = False,
        public_analyst: bool = False,
        high_odds_entry: bool = False,
        split_positions: bool = False,
    ) -> ScoringResult:
        """Apply false positive reduction modifiers.

        Args:
            bet_lost: Position resolved as a loss
            long_diverse_history: >100 trades, >5 categories
            public_analyst: Known public figure with track record
            high_odds_entry: Entered at >60% odds (follower pattern)
            split_positions: Has both YES and NO positions (hedging)

        Returns:
            Modified ScoringResult
        """
        modifier = 0

        if bet_lost:
            modifier -= 40
            result.signals.append(Signal(
                name="bet_lost",
                category="trading",
                weight=-40,
                description="Position resolved as LOSS (strong false positive indicator)"
            ))

        if long_diverse_history:
            modifier -= 20
            result.signals.append(Signal(
                name="diverse_history",
                category="account",
                weight=-20,
                description="Established trader with diverse portfolio"
            ))

        if public_analyst:
            modifier -= 25
            result.signals.append(Signal(
                name="public_analyst",
                category="behavioral",
                weight=-25,
                description="Known public analyst with track record"
            ))

        if high_odds_entry:
            modifier -= 15
            result.signals.append(Signal(
                name="high_odds_entry",
                category="trading",
                weight=-15,
                description="Entered at high odds (follower pattern, not source)"
            ))

        if split_positions:
            modifier -= 10
            result.signals.append(Signal(
                name="split_positions",
                category="behavioral",
                weight=-10,
                description="Has hedging positions (insiders don't hedge)"
            ))

        # Apply modifier
        new_score = max(result.score + modifier, 0)

        # Recalculate priority
        new_priority = self._get_priority(
            new_score,
            result.signal_count,
            result.active_dimensions,
        )

        return ScoringResult(
            score=round(new_score, 2),
            confidence_low=round(max(new_score - 10, 0), 2),
            confidence_high=round(min(new_score + 10, 100), 2),
            priority=new_priority,
            dimensions=result.dimensions,
            signals=result.signals,
            signal_count=result.signal_count,
            active_dimensions=result.active_dimensions,
            downgraded=result.downgraded,
            downgrade_reason=result.downgrade_reason,
        )
