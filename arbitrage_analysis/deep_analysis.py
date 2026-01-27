#!/usr/bin/env python3
"""
Deep Analysis of Arbitrage Bot Trade Patterns

Analyzes extracted trade data to identify:
1. Entry price thresholds
2. Pair accumulation patterns
3. Timing between trades
4. Position sizing logic
5. Actual profit/loss patterns
"""

import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple
import statistics

DATA_DIR = Path(__file__).parent / "data"


def load_analysis(wallet_prefix: str) -> dict:
    """Load analysis JSON for a wallet."""
    files = list(DATA_DIR.glob(f"{wallet_prefix}*_analysis.json"))
    if not files:
        raise FileNotFoundError(f"No analysis file for {wallet_prefix}")
    with open(files[0]) as f:
        return json.load(f)


def load_raw_trades(wallet_prefix: str) -> list:
    """Load raw trades JSON for a wallet."""
    files = list(DATA_DIR.glob(f"{wallet_prefix}*_raw_trades.json"))
    if not files:
        raise FileNotFoundError(f"No raw trades file for {wallet_prefix}")
    with open(files[0]) as f:
        return json.load(f)


def analyze_timing_patterns(markets: list) -> dict:
    """Analyze timing between YES and NO buys."""
    timing_gaps = []
    accumulation_times = []

    for m in markets:
        if m.get("time_between_first_trades_sec") is not None:
            timing_gaps.append(m["time_between_first_trades_sec"])
        if m.get("total_accumulation_time_sec") is not None:
            accumulation_times.append(m["total_accumulation_time_sec"])

    result = {
        "timing_gap_count": len(timing_gaps),
        "accumulation_count": len(accumulation_times),
    }

    if timing_gaps:
        result["timing_gap_avg_sec"] = statistics.mean(timing_gaps)
        result["timing_gap_median_sec"] = statistics.median(timing_gaps)
        result["timing_gap_min_sec"] = min(timing_gaps)
        result["timing_gap_max_sec"] = max(timing_gaps)

    if accumulation_times:
        result["accumulation_avg_sec"] = statistics.mean(accumulation_times)
        result["accumulation_median_sec"] = statistics.median(accumulation_times)
        result["accumulation_min_sec"] = min(accumulation_times)
        result["accumulation_max_sec"] = max(accumulation_times)

    return result


def analyze_pair_costs(markets: list) -> dict:
    """Analyze pair cost distribution in detail."""
    pair_costs = []
    hedge_ratios = []
    profitable_markets = 0
    unprofitable_markets = 0

    for m in markets:
        if not m.get("has_both_sides"):
            continue

        pair_cost = float(m.get("pair_cost", 0))
        hedge_ratio = float(m.get("hedge_ratio", 0))

        if pair_cost > 0:
            pair_costs.append(pair_cost)
            hedge_ratios.append(hedge_ratio)

            # Profitable if pair cost < 0.98 (accounting for 2% winner fee)
            if pair_cost < 0.98:
                profitable_markets += 1
            else:
                unprofitable_markets += 1

    result = {
        "total_both_sides_markets": len(pair_costs),
        "profitable_markets": profitable_markets,
        "unprofitable_markets": unprofitable_markets,
    }

    if pair_costs:
        result["pair_cost_avg"] = statistics.mean(pair_costs)
        result["pair_cost_median"] = statistics.median(pair_costs)
        result["pair_cost_std"] = statistics.stdev(pair_costs) if len(pair_costs) > 1 else 0

        # Distribution buckets
        result["pct_under_95c"] = len([p for p in pair_costs if p < 0.95]) / len(pair_costs)
        result["pct_under_97c"] = len([p for p in pair_costs if p < 0.97]) / len(pair_costs)
        result["pct_under_98c"] = len([p for p in pair_costs if p < 0.98]) / len(pair_costs)
        result["pct_under_99c"] = len([p for p in pair_costs if p < 0.99]) / len(pair_costs)
        result["pct_under_100c"] = len([p for p in pair_costs if p < 1.00]) / len(pair_costs)
        result["pct_over_100c"] = len([p for p in pair_costs if p >= 1.00]) / len(pair_costs)

    if hedge_ratios:
        result["hedge_ratio_avg"] = statistics.mean(hedge_ratios)
        result["hedge_ratio_median"] = statistics.median(hedge_ratios)

    return result


def analyze_entry_prices(trades: list) -> dict:
    """Analyze entry price patterns from raw trades."""
    buy_prices = []
    sell_prices = []

    for t in trades:
        price = float(t.get("price", 0))
        side = t.get("side", "")

        if price <= 0 or price >= 1:
            continue

        if side == "BUY":
            buy_prices.append(price)
        elif side == "SELL":
            sell_prices.append(price)

    result = {
        "total_buys": len(buy_prices),
        "total_sells": len(sell_prices),
    }

    if buy_prices:
        result["buy_avg"] = statistics.mean(buy_prices)
        result["buy_median"] = statistics.median(buy_prices)
        result["buy_min"] = min(buy_prices)
        result["buy_max"] = max(buy_prices)
        result["buy_std"] = statistics.stdev(buy_prices) if len(buy_prices) > 1 else 0

        # Distribution
        result["buy_pct_under_10c"] = len([p for p in buy_prices if p < 0.10]) / len(buy_prices)
        result["buy_pct_under_20c"] = len([p for p in buy_prices if p < 0.20]) / len(buy_prices)
        result["buy_pct_under_30c"] = len([p for p in buy_prices if p < 0.30]) / len(buy_prices)
        result["buy_pct_under_40c"] = len([p for p in buy_prices if p < 0.40]) / len(buy_prices)
        result["buy_pct_under_50c"] = len([p for p in buy_prices if p < 0.50]) / len(buy_prices)
        result["buy_pct_40_60c"] = len([p for p in buy_prices if 0.40 <= p <= 0.60]) / len(buy_prices)
        result["buy_pct_over_70c"] = len([p for p in buy_prices if p > 0.70]) / len(buy_prices)
        result["buy_pct_over_80c"] = len([p for p in buy_prices if p > 0.80]) / len(buy_prices)

    return result


def analyze_position_sizing(trades: list) -> dict:
    """Analyze position sizing patterns."""
    usd_values = []

    for t in trades:
        usd = float(t.get("usd_value", 0))
        if usd > 0:
            usd_values.append(usd)

    result = {"total_trades": len(usd_values)}

    if usd_values:
        result["avg_trade_size"] = statistics.mean(usd_values)
        result["median_trade_size"] = statistics.median(usd_values)
        result["min_trade_size"] = min(usd_values)
        result["max_trade_size"] = max(usd_values)
        result["std_trade_size"] = statistics.stdev(usd_values) if len(usd_values) > 1 else 0
        result["total_volume"] = sum(usd_values)

        # Size distribution
        result["pct_under_10"] = len([v for v in usd_values if v < 10]) / len(usd_values)
        result["pct_under_50"] = len([v for v in usd_values if v < 50]) / len(usd_values)
        result["pct_under_100"] = len([v for v in usd_values if v < 100]) / len(usd_values)
        result["pct_over_500"] = len([v for v in usd_values if v > 500]) / len(usd_values)
        result["pct_over_1000"] = len([v for v in usd_values if v > 1000]) / len(usd_values)

    return result


def analyze_buy_vs_sell_by_outcome(trades: list) -> dict:
    """Analyze buy/sell patterns by outcome (Up vs Down)."""
    outcomes = defaultdict(lambda: {"buys": 0, "sells": 0, "buy_value": 0, "sell_value": 0})

    for t in trades:
        outcome = t.get("outcome", "Unknown")
        side = t.get("side", "")
        usd = float(t.get("usd_value", 0))

        if side == "BUY":
            outcomes[outcome]["buys"] += 1
            outcomes[outcome]["buy_value"] += usd
        elif side == "SELL":
            outcomes[outcome]["sells"] += 1
            outcomes[outcome]["sell_value"] += usd

    return dict(outcomes)


def classify_strategy(analysis: dict, trades: list) -> dict:
    """Classify the strategy based on patterns."""
    agg = analysis.get("aggregate_stats", {})
    markets = analysis.get("markets", [])

    # Get crypto 15m markets only
    crypto_markets = [m for m in markets if m.get("is_crypto_15m")]
    both_sides = [m for m in crypto_markets if m.get("has_both_sides")]

    # Key metrics
    avg_pair_cost = float(agg.get("avg_pair_cost", 1))
    pct_both_sides = float(agg.get("pct_markets_both_sides", 0))
    pct_balanced = float(agg.get("pct_markets_balanced", 0))
    avg_entry = float(agg.get("avg_entry_price", 0.5))

    # Entry price analysis
    entry_dist = agg.get("entry_price_distribution", {})
    pct_under_40c = float(entry_dist.get("under_40c", 0))
    pct_over_70c = float(entry_dist.get("over_70c", 0))

    classification = {
        "metrics": {
            "avg_pair_cost": avg_pair_cost,
            "pct_both_sides": pct_both_sides,
            "pct_balanced": pct_balanced,
            "avg_entry_price": avg_entry,
            "pct_entries_under_40c": pct_under_40c,
            "pct_entries_over_70c": pct_over_70c,
        },
        "strategies_detected": [],
        "primary_strategy": None,
        "confidence": 0,
    }

    # Strategy detection logic

    # 1. Pure Pair Accumulation (Gabagool style)
    if avg_pair_cost < 0.98 and pct_both_sides > 0.80 and pct_under_40c > 0.30:
        classification["strategies_detected"].append("pair_accumulation")
        if pct_balanced > 0.20:
            classification["strategies_detected"].append("balanced_arbitrage")

    # 2. Latency Arbitrage
    if pct_over_70c > 0.30 or avg_pair_cost > 1.02:
        classification["strategies_detected"].append("latency_arbitrage")

    # 3. Market Making / Spread Capture
    if 0.40 <= avg_entry <= 0.60 and pct_balanced > 0.50:
        classification["strategies_detected"].append("market_making")

    # 4. One-Sided Betting
    if pct_both_sides < 0.50:
        classification["strategies_detected"].append("directional_betting")

    # Determine primary strategy
    if "pair_accumulation" in classification["strategies_detected"]:
        classification["primary_strategy"] = "PAIR_ACCUMULATION"
        classification["confidence"] = min(0.9, pct_both_sides)
    elif "market_making" in classification["strategies_detected"]:
        classification["primary_strategy"] = "MARKET_MAKING"
        classification["confidence"] = pct_balanced
    elif "latency_arbitrage" in classification["strategies_detected"]:
        classification["primary_strategy"] = "LATENCY_ARBITRAGE"
        classification["confidence"] = pct_over_70c
    elif "directional_betting" in classification["strategies_detected"]:
        classification["primary_strategy"] = "DIRECTIONAL"
        classification["confidence"] = 1 - pct_both_sides
    else:
        classification["primary_strategy"] = "MIXED"
        classification["confidence"] = 0.5

    return classification


def print_wallet_analysis(wallet_prefix: str):
    """Print detailed analysis for a wallet."""
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: {wallet_prefix}")
    print(f"{'='*70}")

    analysis = load_analysis(wallet_prefix)
    trades = load_raw_trades(wallet_prefix)
    markets = [m for m in analysis.get("markets", []) if m.get("is_crypto_15m")]

    # Strategy Classification
    classification = classify_strategy(analysis, trades)
    print(f"\n=== STRATEGY CLASSIFICATION ===")
    print(f"Primary Strategy: {classification['primary_strategy']}")
    print(f"Confidence: {classification['confidence']:.1%}")
    print(f"Strategies Detected: {', '.join(classification['strategies_detected']) or 'None'}")

    # Pair Cost Analysis
    pair_analysis = analyze_pair_costs(markets)
    print(f"\n=== PAIR COST ANALYSIS ===")
    print(f"Markets with Both Sides: {pair_analysis.get('total_both_sides_markets', 0)}")
    print(f"Profitable (<$0.98): {pair_analysis.get('profitable_markets', 0)}")
    print(f"Unprofitable (>=$0.98): {pair_analysis.get('unprofitable_markets', 0)}")
    if pair_analysis.get('pair_cost_avg'):
        print(f"Avg Pair Cost: ${pair_analysis['pair_cost_avg']:.4f}")
        print(f"Median Pair Cost: ${pair_analysis['pair_cost_median']:.4f}")
        print(f"Std Dev: ${pair_analysis['pair_cost_std']:.4f}")
        print(f"\nPair Cost Distribution:")
        print(f"  < $0.95: {pair_analysis.get('pct_under_95c', 0):.1%}")
        print(f"  < $0.97: {pair_analysis.get('pct_under_97c', 0):.1%}")
        print(f"  < $0.98: {pair_analysis.get('pct_under_98c', 0):.1%}")
        print(f"  < $0.99: {pair_analysis.get('pct_under_99c', 0):.1%}")
        print(f"  < $1.00: {pair_analysis.get('pct_under_100c', 0):.1%}")
        print(f"  >= $1.00: {pair_analysis.get('pct_over_100c', 0):.1%}")

    # Entry Price Analysis
    entry_analysis = analyze_entry_prices(trades)
    print(f"\n=== ENTRY PRICE ANALYSIS ===")
    print(f"Total Buys: {entry_analysis.get('total_buys', 0):,}")
    print(f"Total Sells: {entry_analysis.get('total_sells', 0):,}")
    if entry_analysis.get('buy_avg'):
        print(f"Avg Buy Price: ${entry_analysis['buy_avg']:.4f}")
        print(f"Median Buy Price: ${entry_analysis['buy_median']:.4f}")
        print(f"Min/Max: ${entry_analysis['buy_min']:.4f} - ${entry_analysis['buy_max']:.4f}")
        print(f"\nBuy Price Distribution:")
        print(f"  < $0.10: {entry_analysis.get('buy_pct_under_10c', 0):.1%}")
        print(f"  < $0.20: {entry_analysis.get('buy_pct_under_20c', 0):.1%}")
        print(f"  < $0.30: {entry_analysis.get('buy_pct_under_30c', 0):.1%}")
        print(f"  < $0.40: {entry_analysis.get('buy_pct_under_40c', 0):.1%}")
        print(f"  < $0.50: {entry_analysis.get('buy_pct_under_50c', 0):.1%}")
        print(f"  $0.40-$0.60: {entry_analysis.get('buy_pct_40_60c', 0):.1%}")
        print(f"  > $0.70: {entry_analysis.get('buy_pct_over_70c', 0):.1%}")
        print(f"  > $0.80: {entry_analysis.get('buy_pct_over_80c', 0):.1%}")

    # Position Sizing
    size_analysis = analyze_position_sizing(trades)
    print(f"\n=== POSITION SIZING ===")
    print(f"Total Volume: ${size_analysis.get('total_volume', 0):,.2f}")
    print(f"Avg Trade Size: ${size_analysis.get('avg_trade_size', 0):.2f}")
    print(f"Median Trade Size: ${size_analysis.get('median_trade_size', 0):.2f}")
    print(f"Size Range: ${size_analysis.get('min_trade_size', 0):.2f} - ${size_analysis.get('max_trade_size', 0):.2f}")
    print(f"\nSize Distribution:")
    print(f"  < $10: {size_analysis.get('pct_under_10', 0):.1%}")
    print(f"  < $50: {size_analysis.get('pct_under_50', 0):.1%}")
    print(f"  < $100: {size_analysis.get('pct_under_100', 0):.1%}")
    print(f"  > $500: {size_analysis.get('pct_over_500', 0):.1%}")
    print(f"  > $1000: {size_analysis.get('pct_over_1000', 0):.1%}")

    # Timing Analysis
    timing = analyze_timing_patterns(markets)
    print(f"\n=== TIMING ANALYSIS ===")
    if timing.get("timing_gap_avg_sec") is not None:
        print(f"Time Between First Sides:")
        print(f"  Avg: {timing['timing_gap_avg_sec']:.1f}s")
        print(f"  Median: {timing['timing_gap_median_sec']:.1f}s")
        print(f"  Range: {timing['timing_gap_min_sec']:.1f}s - {timing['timing_gap_max_sec']:.1f}s")
    if timing.get("accumulation_avg_sec") is not None:
        print(f"Total Accumulation Time:")
        print(f"  Avg: {timing['accumulation_avg_sec']:.1f}s ({timing['accumulation_avg_sec']/60:.1f} min)")
        print(f"  Median: {timing['accumulation_median_sec']:.1f}s ({timing['accumulation_median_sec']/60:.1f} min)")

    # Outcome Analysis
    outcome_analysis = analyze_buy_vs_sell_by_outcome(trades)
    print(f"\n=== OUTCOME ANALYSIS ===")
    for outcome, stats in sorted(outcome_analysis.items()):
        total = stats["buys"] + stats["sells"]
        if total > 0:
            print(f"{outcome}:")
            print(f"  Buys: {stats['buys']:,} (${stats['buy_value']:,.2f})")
            print(f"  Sells: {stats['sells']:,} (${stats['sell_value']:,.2f})")

    return classification


def main():
    print("="*70)
    print("POLYMARKET ARBITRAGE BOT DEEP ANALYSIS")
    print("="*70)

    # Find all analysis files
    analysis_files = list(DATA_DIR.glob("*_analysis.json"))
    wallets = []

    for f in analysis_files:
        prefix = f.name.split("_")[0]
        if prefix not in wallets:
            wallets.append(prefix)

    print(f"\nFound {len(wallets)} wallets to analyze")

    classifications = {}
    for wallet in sorted(wallets):
        try:
            classification = print_wallet_analysis(wallet)
            classifications[wallet] = classification
        except Exception as e:
            print(f"\nError analyzing {wallet}: {e}")

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)
    print(f"\n{'Wallet':<12} {'Strategy':<20} {'Conf':<6} {'AvgPair':<8} {'Both%':<7} {'Bal%':<6} {'AvgEntry':<8}")
    print("-"*70)

    for wallet, cls in sorted(classifications.items()):
        m = cls["metrics"]
        print(f"{wallet:<12} {cls['primary_strategy']:<20} {cls['confidence']:.1%}   ${m['avg_pair_cost']:.3f}  {m['pct_both_sides']:.1%}   {m['pct_balanced']:.1%}  ${m['avg_entry_price']:.3f}")


if __name__ == "__main__":
    main()
