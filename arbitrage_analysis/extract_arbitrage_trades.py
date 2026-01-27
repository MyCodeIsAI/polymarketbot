#!/usr/bin/env python3
"""
Arbitrage Bot Trade Extractor

Extracts and analyzes trade history from known Polymarket arbitrage bot accounts
to reverse engineer their 15-minute crypto market strategies.

Usage:
    python3 arbitrage_analysis/extract_arbitrage_trades.py <wallet_address>
    python3 arbitrage_analysis/extract_arbitrage_trades.py --file wallets.txt

Output:
    - arbitrage_analysis/data/<wallet>_raw_trades.json
    - arbitrage_analysis/data/<wallet>_analysis.json
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType, Activity

# Output directory
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class TradeRecord:
    """Normalized trade record for analysis."""
    id: str
    timestamp: datetime
    condition_id: str
    token_id: str
    outcome: str  # "Yes" or "No"
    side: str  # "BUY" or "SELL"
    price: Decimal
    shares: Decimal
    usd_value: Decimal
    market_title: Optional[str]
    event_slug: Optional[str]
    tx_hash: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "side": self.side,
            "price": str(self.price),
            "shares": str(self.shares),
            "usd_value": str(self.usd_value),
            "market_title": self.market_title,
            "event_slug": self.event_slug,
            "tx_hash": self.tx_hash,
        }


@dataclass
class MarketSideAnalysis:
    """Analysis of one side (YES or NO) of a market."""
    outcome: str
    trade_count: int = 0
    total_shares: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")
    min_price: Decimal = Decimal("1")
    max_price: Decimal = Decimal("0")
    first_trade_ts: Optional[datetime] = None
    last_trade_ts: Optional[datetime] = None
    trades: List[TradeRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "trade_count": self.trade_count,
            "total_shares": str(self.total_shares),
            "total_cost": str(self.total_cost),
            "avg_price": str(self.avg_price),
            "min_price": str(self.min_price),
            "max_price": str(self.max_price),
            "first_trade_ts": self.first_trade_ts.isoformat() if self.first_trade_ts else None,
            "last_trade_ts": self.last_trade_ts.isoformat() if self.last_trade_ts else None,
        }


@dataclass
class MarketPairAnalysis:
    """Analysis of a single market (YES + NO combined)."""
    condition_id: str
    market_title: Optional[str]
    event_slug: Optional[str]
    is_crypto_15m: bool = False

    # Side-level analysis
    yes_side: Optional[MarketSideAnalysis] = None
    no_side: Optional[MarketSideAnalysis] = None

    # Pair-level metrics (calculated after side analysis)
    pair_cost: Decimal = Decimal("0")  # avg_yes + avg_no
    has_both_sides: bool = False
    pair_imbalance: Decimal = Decimal("0")  # abs(yes_shares - no_shares)
    hedge_ratio: Decimal = Decimal("0")  # min_shares / max_shares

    # Timing
    time_between_first_trades: Optional[timedelta] = None
    total_accumulation_time: Optional[timedelta] = None

    def calculate_pair_metrics(self):
        """Calculate pair-level metrics from side analysis."""
        if self.yes_side and self.no_side:
            self.has_both_sides = True

            # Pair cost
            yes_avg = self.yes_side.avg_price if self.yes_side.total_shares > 0 else Decimal("0")
            no_avg = self.no_side.avg_price if self.no_side.total_shares > 0 else Decimal("0")
            self.pair_cost = yes_avg + no_avg

            # Imbalance and hedge ratio
            yes_shares = self.yes_side.total_shares
            no_shares = self.no_side.total_shares
            self.pair_imbalance = abs(yes_shares - no_shares)

            max_shares = max(yes_shares, no_shares)
            min_shares = min(yes_shares, no_shares)
            if max_shares > 0:
                self.hedge_ratio = min_shares / max_shares

            # Timing between sides
            if self.yes_side.first_trade_ts and self.no_side.first_trade_ts:
                self.time_between_first_trades = abs(
                    self.yes_side.first_trade_ts - self.no_side.first_trade_ts
                )

            # Total accumulation time
            all_trades = (self.yes_side.trades or []) + (self.no_side.trades or [])
            if len(all_trades) >= 2:
                timestamps = sorted([t.timestamp for t in all_trades])
                self.total_accumulation_time = timestamps[-1] - timestamps[0]
        else:
            self.has_both_sides = False
            if self.yes_side:
                self.pair_cost = self.yes_side.avg_price
            elif self.no_side:
                self.pair_cost = self.no_side.avg_price

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "market_title": self.market_title,
            "event_slug": self.event_slug,
            "is_crypto_15m": self.is_crypto_15m,
            "yes_side": self.yes_side.to_dict() if self.yes_side else None,
            "no_side": self.no_side.to_dict() if self.no_side else None,
            "pair_cost": str(self.pair_cost),
            "has_both_sides": self.has_both_sides,
            "pair_imbalance": str(self.pair_imbalance),
            "hedge_ratio": str(self.hedge_ratio),
            "time_between_first_trades_sec": self.time_between_first_trades.total_seconds() if self.time_between_first_trades else None,
            "total_accumulation_time_sec": self.total_accumulation_time.total_seconds() if self.total_accumulation_time else None,
        }


@dataclass
class WalletArbitrageProfile:
    """Complete arbitrage analysis profile for a wallet."""
    wallet: str
    extraction_timestamp: datetime
    total_trades: int = 0
    crypto_15m_trades: int = 0
    crypto_15m_markets: int = 0

    # Market-level analysis
    markets: List[MarketPairAnalysis] = field(default_factory=list)

    # Aggregate statistics (calculated)
    avg_pair_cost: Decimal = Decimal("0")
    median_pair_cost: Decimal = Decimal("0")
    min_pair_cost: Decimal = Decimal("1")
    max_pair_cost: Decimal = Decimal("1")

    avg_entry_price: Decimal = Decimal("0")
    median_entry_price: Decimal = Decimal("0")
    min_entry_price: Decimal = Decimal("1")
    max_entry_price: Decimal = Decimal("0")

    pct_markets_both_sides: float = 0.0
    pct_markets_balanced: float = 0.0  # hedge_ratio > 0.9

    # Entry price distribution
    pct_entries_under_30c: float = 0.0
    pct_entries_under_40c: float = 0.0
    pct_entries_under_50c: float = 0.0
    pct_entries_over_70c: float = 0.0

    def calculate_aggregate_stats(self):
        """Calculate aggregate statistics from market-level analysis."""
        if not self.markets:
            return

        # Filter to crypto 15m markets with both sides
        crypto_markets = [m for m in self.markets if m.is_crypto_15m]
        both_sides_markets = [m for m in crypto_markets if m.has_both_sides]

        self.crypto_15m_markets = len(crypto_markets)

        # Pair cost stats
        pair_costs = [m.pair_cost for m in both_sides_markets if m.pair_cost > 0]
        if pair_costs:
            sorted_costs = sorted(pair_costs)
            self.avg_pair_cost = sum(pair_costs) / len(pair_costs)
            self.median_pair_cost = sorted_costs[len(sorted_costs) // 2]
            self.min_pair_cost = min(pair_costs)
            self.max_pair_cost = max(pair_costs)

        # Entry price stats (all buys)
        all_entry_prices = []
        for m in crypto_markets:
            if m.yes_side:
                all_entry_prices.extend([t.price for t in m.yes_side.trades if t.side == "BUY"])
            if m.no_side:
                all_entry_prices.extend([t.price for t in m.no_side.trades if t.side == "BUY"])

        if all_entry_prices:
            sorted_prices = sorted(all_entry_prices)
            self.avg_entry_price = sum(all_entry_prices) / len(all_entry_prices)
            self.median_entry_price = sorted_prices[len(sorted_prices) // 2]
            self.min_entry_price = min(all_entry_prices)
            self.max_entry_price = max(all_entry_prices)

            # Distribution
            total = len(all_entry_prices)
            self.pct_entries_under_30c = len([p for p in all_entry_prices if p < Decimal("0.30")]) / total
            self.pct_entries_under_40c = len([p for p in all_entry_prices if p < Decimal("0.40")]) / total
            self.pct_entries_under_50c = len([p for p in all_entry_prices if p < Decimal("0.50")]) / total
            self.pct_entries_over_70c = len([p for p in all_entry_prices if p > Decimal("0.70")]) / total

        # Market strategy stats
        if crypto_markets:
            self.pct_markets_both_sides = len(both_sides_markets) / len(crypto_markets)
            balanced = [m for m in both_sides_markets if m.hedge_ratio > Decimal("0.9")]
            self.pct_markets_balanced = len(balanced) / len(crypto_markets) if crypto_markets else 0

        # Count crypto 15m trades
        for m in crypto_markets:
            if m.yes_side:
                self.crypto_15m_trades += len(m.yes_side.trades)
            if m.no_side:
                self.crypto_15m_trades += len(m.no_side.trades)

    def to_dict(self) -> dict:
        return {
            "wallet": self.wallet,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "total_trades": self.total_trades,
            "crypto_15m_trades": self.crypto_15m_trades,
            "crypto_15m_markets": self.crypto_15m_markets,
            "aggregate_stats": {
                "avg_pair_cost": str(self.avg_pair_cost),
                "median_pair_cost": str(self.median_pair_cost),
                "min_pair_cost": str(self.min_pair_cost),
                "max_pair_cost": str(self.max_pair_cost),
                "avg_entry_price": str(self.avg_entry_price),
                "median_entry_price": str(self.median_entry_price),
                "min_entry_price": str(self.min_entry_price),
                "max_entry_price": str(self.max_entry_price),
                "pct_markets_both_sides": self.pct_markets_both_sides,
                "pct_markets_balanced": self.pct_markets_balanced,
                "entry_price_distribution": {
                    "under_30c": self.pct_entries_under_30c,
                    "under_40c": self.pct_entries_under_40c,
                    "under_50c": self.pct_entries_under_50c,
                    "over_70c": self.pct_entries_over_70c,
                },
            },
            "markets": [m.to_dict() for m in self.markets],
        }


# =============================================================================
# Utility Functions
# =============================================================================

def is_crypto_15m_market(title: Optional[str], slug: Optional[str]) -> bool:
    """Detect if a market is a 15-minute crypto up/down market."""
    text = ""
    if title:
        text += title.lower() + " "
    if slug:
        text += slug.lower()

    if not text:
        return False

    crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp"]
    updown_keywords = ["up", "down", "up/down", "up or down", "15m", "15-min", "15 min"]

    is_crypto = any(kw in text for kw in crypto_keywords)
    is_updown = any(kw in text for kw in updown_keywords)

    return is_crypto and is_updown


def activity_to_trade_record(activity: Activity) -> TradeRecord:
    """Convert Activity to TradeRecord."""
    # Calculate shares from USD value and price
    if activity.price and activity.price > 0:
        shares = activity.usd_value / activity.price
    else:
        shares = activity.size

    return TradeRecord(
        id=activity.id,
        timestamp=activity.timestamp,
        condition_id=activity.condition_id,
        token_id=activity.token_id,
        outcome=activity.outcome,
        side=activity.side.value if activity.side else "BUY",
        price=activity.price,
        shares=shares,
        usd_value=activity.usd_value,
        market_title=activity.market_title,
        event_slug=activity.event_slug,
        tx_hash=activity.tx_hash,
    )


# =============================================================================
# Extraction Functions
# =============================================================================

async def fetch_all_trades(client: DataAPIClient, wallet: str, max_pages: int = 50) -> List[Activity]:
    """Fetch ALL trade activity for a wallet by paginating."""
    all_trades = []
    offset = 0
    limit = 500

    print(f"Fetching trades for {wallet[:10]}...")

    for page in range(max_pages):
        try:
            activities = await client.get_activity(
                user=wallet,
                activity_type=ActivityType.TRADE,
                limit=limit,
                offset=offset,
            )

            if not activities:
                break

            all_trades.extend(activities)
            print(f"  Page {page + 1}: {len(activities)} trades (total: {len(all_trades)})")

            if len(activities) < limit:
                break

            offset += limit
            await asyncio.sleep(0.05)  # Rate limiting

        except Exception as e:
            print(f"  Error on page {page + 1}: {e}")
            break

    print(f"  Total trades fetched: {len(all_trades)}")
    return all_trades


def analyze_side(trades: List[TradeRecord], outcome: str) -> MarketSideAnalysis:
    """Analyze one side (YES or NO) of a market."""
    side = MarketSideAnalysis(outcome=outcome)
    buy_trades = [t for t in trades if t.side == "BUY"]

    if not buy_trades:
        return side

    side.trade_count = len(buy_trades)
    side.trades = buy_trades

    # Calculate totals
    side.total_shares = sum(t.shares for t in buy_trades)
    side.total_cost = sum(t.usd_value for t in buy_trades)

    # Calculate averages
    if side.total_shares > 0:
        side.avg_price = side.total_cost / side.total_shares

    # Price range
    prices = [t.price for t in buy_trades if t.price > 0]
    if prices:
        side.min_price = min(prices)
        side.max_price = max(prices)

    # Timestamps
    timestamps = sorted([t.timestamp for t in buy_trades])
    if timestamps:
        side.first_trade_ts = timestamps[0]
        side.last_trade_ts = timestamps[-1]

    return side


def normalize_outcome(outcome: str) -> str:
    """Normalize outcome to 'Yes' or 'No' (handles Up/Down for crypto markets)."""
    if not outcome:
        return ""
    outcome_lower = outcome.lower()
    # Crypto up/down markets use "Up"/"Down" instead of "Yes"/"No"
    if outcome_lower in ("yes", "up"):
        return "Yes"
    elif outcome_lower in ("no", "down"):
        return "No"
    return outcome


def analyze_markets(trades: List[TradeRecord]) -> List[MarketPairAnalysis]:
    """Group trades by market and analyze each."""
    # Group by condition_id
    markets_dict: Dict[str, List[TradeRecord]] = defaultdict(list)
    for trade in trades:
        markets_dict[trade.condition_id].append(trade)

    analyses = []
    for condition_id, market_trades in markets_dict.items():
        # Get market metadata from first trade
        first_trade = market_trades[0]

        analysis = MarketPairAnalysis(
            condition_id=condition_id,
            market_title=first_trade.market_title,
            event_slug=first_trade.event_slug,
            is_crypto_15m=is_crypto_15m_market(first_trade.market_title, first_trade.event_slug),
        )

        # Separate by outcome (handles both Yes/No and Up/Down)
        yes_trades = [t for t in market_trades if normalize_outcome(t.outcome) == "Yes"]
        no_trades = [t for t in market_trades if normalize_outcome(t.outcome) == "No"]

        # Analyze each side
        if yes_trades:
            analysis.yes_side = analyze_side(yes_trades, "Yes")
        if no_trades:
            analysis.no_side = analyze_side(no_trades, "No")

        # Calculate pair metrics
        analysis.calculate_pair_metrics()

        analyses.append(analysis)

    return analyses


async def extract_and_analyze_wallet(wallet: str) -> WalletArbitrageProfile:
    """Extract and analyze all arbitrage activity for a wallet."""
    print(f"\n{'='*60}")
    print(f"Extracting: {wallet}")
    print(f"{'='*60}")

    async with DataAPIClient() as client:
        # Fetch all trades
        activities = await fetch_all_trades(client, wallet)

        # Convert to TradeRecords
        trades = [activity_to_trade_record(a) for a in activities]

        # Analyze markets
        print("\nAnalyzing markets...")
        markets = analyze_markets(trades)

        # Create profile
        profile = WalletArbitrageProfile(
            wallet=wallet,
            extraction_timestamp=datetime.utcnow(),
            total_trades=len(trades),
            markets=markets,
        )

        # Calculate aggregates
        profile.calculate_aggregate_stats()

        return profile


def save_results(profile: WalletArbitrageProfile):
    """Save extraction results to files."""
    wallet_short = profile.wallet[:10]

    # Save raw trades
    raw_path = OUTPUT_DIR / f"{wallet_short}_raw_trades.json"
    all_trades = []
    for market in profile.markets:
        if market.yes_side:
            all_trades.extend([t.to_dict() for t in market.yes_side.trades])
        if market.no_side:
            all_trades.extend([t.to_dict() for t in market.no_side.trades])

    with open(raw_path, "w") as f:
        json.dump(all_trades, f, indent=2)
    print(f"\nSaved raw trades to: {raw_path}")

    # Save analysis
    analysis_path = OUTPUT_DIR / f"{wallet_short}_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2, default=str)
    print(f"Saved analysis to: {analysis_path}")

    return raw_path, analysis_path


def print_summary(profile: WalletArbitrageProfile):
    """Print a summary of the analysis."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY: {profile.wallet[:20]}...")
    print(f"{'='*60}")

    print(f"\nTotal Trades: {profile.total_trades:,}")
    print(f"15-min Crypto Trades: {profile.crypto_15m_trades:,}")
    print(f"15-min Crypto Markets: {profile.crypto_15m_markets:,}")

    print(f"\n--- Pair Cost Analysis ---")
    print(f"Avg Pair Cost:    ${float(profile.avg_pair_cost):.4f}")
    print(f"Median Pair Cost: ${float(profile.median_pair_cost):.4f}")
    print(f"Min Pair Cost:    ${float(profile.min_pair_cost):.4f}")
    print(f"Max Pair Cost:    ${float(profile.max_pair_cost):.4f}")

    print(f"\n--- Entry Price Analysis ---")
    print(f"Avg Entry Price:    ${float(profile.avg_entry_price):.4f}")
    print(f"Median Entry Price: ${float(profile.median_entry_price):.4f}")
    print(f"% Under 30¢: {profile.pct_entries_under_30c:.1%}")
    print(f"% Under 40¢: {profile.pct_entries_under_40c:.1%}")
    print(f"% Under 50¢: {profile.pct_entries_under_50c:.1%}")
    print(f"% Over 70¢:  {profile.pct_entries_over_70c:.1%}")

    print(f"\n--- Strategy Signals ---")
    print(f"% Markets with Both Sides: {profile.pct_markets_both_sides:.1%}")
    print(f"% Markets Balanced (>90%): {profile.pct_markets_balanced:.1%}")

    # Top 5 crypto 15m markets by pair cost (lowest = most profitable)
    crypto_markets = [m for m in profile.markets if m.is_crypto_15m and m.has_both_sides]
    if crypto_markets:
        print(f"\n--- Top 5 Best Pair Costs (15m Crypto) ---")
        sorted_markets = sorted(crypto_markets, key=lambda m: m.pair_cost)
        for i, m in enumerate(sorted_markets[:5], 1):
            title = (m.market_title or m.event_slug or "Unknown")[:40]
            print(f"{i}. {title}: ${float(m.pair_cost):.4f} (hedge: {float(m.hedge_ratio):.1%})")


# =============================================================================
# Main
# =============================================================================

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_arbitrage_trades.py <wallet_address>")
        print("       python3 extract_arbitrage_trades.py --file wallets.txt")
        sys.exit(1)

    wallets = []

    if sys.argv[1] == "--file":
        # Load wallets from file
        file_path = Path(sys.argv[2])
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        with open(file_path) as f:
            wallets = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        # Single wallet
        wallets = [sys.argv[1]]

    print(f"\nExtracting {len(wallets)} wallet(s)...")

    for wallet in wallets:
        try:
            profile = await extract_and_analyze_wallet(wallet)
            save_results(profile)
            print_summary(profile)
        except Exception as e:
            print(f"\nError processing {wallet}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
