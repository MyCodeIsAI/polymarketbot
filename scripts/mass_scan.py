#!/usr/bin/env python3
"""
Mass scan script to build the default pre-analyzed accounts list.

Scans up to 15,000 accounts from the Polymarket leaderboard and populates
the analyzed_accounts.json file for the Discovery UI's "Load Pre-Analyzed" button.

Usage:
    DISCOVERY_PORT=8766 python3 scripts/mass_scan.py

Output: data/analyzed_accounts.json
"""
import asyncio
import json
import sys
import re
import aiohttp
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode


# =============================================================================
# Profile Views Fetcher (Phase 4)
# =============================================================================

@dataclass
class ProfileStats:
    """Profile statistics from Polymarket."""
    wallet: str
    views: int
    username: Optional[str] = None


async def fetch_profile_views(wallet: str, session: aiohttp.ClientSession) -> Optional[ProfileStats]:
    """Fetch profile views for a single wallet."""
    url = f"https://polymarket.com/profile/{wallet.lower()}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html',
    }

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return ProfileStats(wallet=wallet.lower(), views=0)

            html = await resp.text()

            views = 0
            username = None

            views_match = re.search(r'"views":(\d+)', html)
            if views_match:
                views = int(views_match.group(1))

            name_match = re.search(r'"username":"([^"]*)"', html)
            if name_match and name_match.group(1):
                username = name_match.group(1)

            return ProfileStats(wallet=wallet.lower(), views=views, username=username)

    except Exception:
        return ProfileStats(wallet=wallet.lower(), views=0)


async def fetch_all_profile_views(
    wallets: list[str],
    max_concurrent: int = 10,
) -> Dict[str, ProfileStats]:
    """
    Fetch profile views for all wallets in parallel with rate limiting.

    Args:
        wallets: List of wallet addresses
        max_concurrent: Max concurrent requests

    Returns:
        Dict mapping wallet -> ProfileStats
    """
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = [0]
    total = len(wallets)

    async with aiohttp.ClientSession() as session:
        async def fetch_one(wallet: str):
            async with semaphore:
                stats = await fetch_profile_views(wallet, session)
                if stats:
                    results[wallet.lower()] = stats
                completed[0] += 1
                # Progress update every 100
                if completed[0] % 100 == 0:
                    print(f"    Views: {completed[0]:,}/{total:,} fetched...")
                # Small delay between requests
                await asyncio.sleep(0.1)

        await asyncio.gather(*[fetch_one(w) for w in wallets])

    return results

# Configuration - Override with --test for light scan
TEST_MODE = "--test" in sys.argv or "-t" in sys.argv

MIN_PNL = 10000        # Minimum P/L: $10,000
MAX_PNL = 100000000    # Maximum P/L: $100M (effectively unlimited)
MIN_TRADES = 25        # Minimum trades
MAX_TRADES = 5000      # Maximum trades
MAX_CANDIDATES = 500 if TEST_MODE else 20000  # Max accounts to pull from leaderboard
TRADES_TO_SCRAPE = 50  # Trades to analyze per account (for category detection)
REQUIRE_RECENT_TRADE = True   # Must have traded in last N days
RECENT_TRADE_DAYS = 30        # Days to consider "recent"

# Output file
OUTPUT_FILE = PROJECT_ROOT / "data" / (
    "analyzed_accounts_TEST.json" if TEST_MODE else "analyzed_accounts.json"
)
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "mass_scan_checkpoint.json"

if TEST_MODE:
    print("\n*** TEST MODE: Scanning only 50 accounts ***\n")


def convert_decimals(obj):
    """Recursively convert Decimals to floats for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    return obj


def normalize_result(result: dict) -> dict:
    """Convert scan result format to analyzed_accounts format.

    Maps fields from DeepAnalysisResult.to_dict() to UI2/UI3 expected format.
    Field sources:
    - pl_metrics: PLCurveMetrics (sharpe, profit_factor, win_rate, drawdowns, etc.)
    - pattern_metrics: TradingPatternMetrics (trades, positions, timing, categories)
    - result root: wallet_address, composite_score, total_pnl (from leaderboard)
    """
    # Extract metrics dicts (may be None)
    pl = result.get("pl_metrics") or {}
    pattern = result.get("pattern_metrics") or {}

    # Core metrics - total_pnl from leaderboard (authoritative), fallback to computed
    total_pnl = float(result.get("total_pnl") or pl.get("total_realized_pnl") or 0)

    # Pattern metrics - use correct field names from TradingPatternMetrics
    num_trades = pattern.get("total_trades", 0)
    unique_markets = pattern.get("unique_markets_traded", 0)
    account_age_days = pattern.get("account_age_days", 0)
    active_days = pattern.get("active_days", 0)
    days_since_last_trade = pattern.get("days_since_last_trade", 999)
    trades_per_day = pattern.get("trades_per_day_avg", 0)

    # Position sizes - note the _usd suffix in actual field names
    avg_position = float(pattern.get("avg_position_size_usd") or 0)
    median_position = float(pattern.get("median_position_size_usd") or 0)
    max_position = float(pattern.get("max_position_size_usd") or 0)
    position_std_dev = float(pattern.get("position_size_std_dev") or 0)

    # P/L metrics - from PLCurveMetrics
    # Use market_win_rate (market-level) which includes redemptions, fallback to sell-based win_rate
    market_win_rate = pl.get("market_win_rate", 0)
    sell_win_rate = pl.get("win_rate", 0)
    # Prefer market_win_rate as it counts at market level (profitable markets / resolved markets)
    # instead of trade level (profitable sells / total sells) which misses held-to-resolution positions
    markets_resolved = pl.get("markets_profitable", 0) + pl.get("markets_unprofitable", 0)
    win_rate = market_win_rate if markets_resolved > 0 else sell_win_rate
    profit_factor = pl.get("profit_factor", 1.0)
    sharpe_ratio = pl.get("sharpe_ratio", 0)
    sortino_ratio = pl.get("sortino_ratio", 0)
    max_drawdown_pct = pl.get("max_drawdown_pct", 0)
    avg_drawdown_pct = pl.get("avg_drawdown_pct", 0)
    win_count = pl.get("win_count", 0)
    loss_count = pl.get("loss_count", 0)
    avg_win_size = float(pl.get("avg_win_size") or 0)
    avg_loss_size = float(pl.get("avg_loss_size") or 0)
    gross_profit = float(pl.get("gross_profit") or 0)
    gross_loss = float(pl.get("gross_loss") or 0)

    # Safe division for derived metrics
    pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0
    pnl_per_market = total_pnl / unique_markets if unique_markets > 0 else 0
    trades_per_week = trades_per_day * 7

    # Position consistency (coefficient of variation - lower is more consistent)
    position_consistency = 1 - (position_std_dev / avg_position) if avg_position > 0 else 0
    position_consistency = max(0, min(1, position_consistency))  # Clamp to 0-1

    # Category data
    category_breakdown = pattern.get("category_breakdown", {})
    primary_category = pattern.get("primary_category", "Diversified")
    category_concentration = pattern.get("category_concentration", 0)

    return {
        # Identity
        "wallet_address": result.get("wallet_address", ""),
        "wallet": result.get("wallet_address", ""),

        # P/L metrics
        "total_pnl": total_pnl,
        "pnl_per_trade": pnl_per_trade,
        "pnl_per_market": pnl_per_market,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,

        # Trading volume
        "num_trades": num_trades,
        "trades_fetched": num_trades,
        "unique_markets": unique_markets,
        "trades_per_week": trades_per_week,

        # Position sizing
        "avg_position": avg_position,
        "avg_position_size": avg_position,
        "median_position": median_position,
        "max_position": max_position,
        "position_consistency": position_consistency,

        # Timing
        "account_age_days": account_age_days,
        "active_days": active_days,
        "activity_recency_days": days_since_last_trade,
        "is_currently_active": days_since_last_trade <= 7,

        # Win/loss stats (market-level when available, trade-level fallback)
        "win_rate": win_rate,
        "market_win_rate": market_win_rate,  # Market-level win rate (includes redemptions)
        "sell_win_rate": sell_win_rate,  # Trade-level win rate (sells only)
        "markets_resolved": markets_resolved,  # Number of resolved markets
        "profitable_trades": win_count,
        "losing_trades": loss_count,
        "avg_win_size": avg_win_size,
        "avg_loss_size": avg_loss_size,

        # Risk metrics
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_drawdown_pct": avg_drawdown_pct,
        "max_drawdown_usd": total_pnl * max_drawdown_pct if max_drawdown_pct else 0,
        # High watermark drawdown (compares realized P/L to unrealized losses)
        "off_high_watermark_pct": pl.get("off_high_watermark_pct", 0),
        "unrealized_pnl": pl.get("unrealized_pnl", 0),

        # Trade direction
        "buy_sell_ratio": pattern.get("buy_sell_ratio", 0.5),

        # Odds profile
        "pct_under_5c": pattern.get("pct_trades_under_5c", 0),
        "pct_under_10c": pattern.get("pct_trades_under_10c", 0),
        "pct_under_20c": pattern.get("pct_trades_under_20c", 0),
        "pct_over_80c": pattern.get("pct_trades_over_80c", 0),
        "avg_entry_odds": float(pattern.get("avg_entry_odds") or 0.5),
        "median_entry_odds": float(pattern.get("median_entry_odds") or 0.5),

        # Category/specialization
        "category_breakdown": category_breakdown,
        "primary_category": primary_category,
        "category_concentration": category_concentration,
        "niche_market_pct": pattern.get("niche_market_pct", 0),

        # Category win rates (from result root)
        "category_win_rates": result.get("category_win_rates", {}),

        # Scoring
        "systematic_score": result.get("composite_score", 0),
        "score_breakdown": result.get("score_breakdown", {}),
        "red_flags": result.get("red_flags", []),
        "red_flag_count": result.get("red_flag_count", 0),

        # Streaks (from pl_metrics)
        "longest_win_streak": pl.get("longest_win_streak", 0),
        "longest_loss_streak": pl.get("longest_loss_streak", 0),
        "current_streak": pl.get("current_streak", 0),

        # Concentration risk
        "largest_win_pct": pl.get("largest_win_pct_of_total", 0),
        "top_3_wins_pct": pl.get("top_3_wins_pct_of_total", 0),

        # Sample data for UI display
        "recent_trades_sample": result.get("recent_trades_sample", []),
        "pl_curve_data": result.get("pl_curve_data", []),
        "market_categories": result.get("market_categories", {}),

        # Profile views (populated in Phase 4)
        "profile_views": 0,
        "profile_username": None,
    }


def calculate_distributions(accounts: list) -> dict:
    """Calculate statistics and distributions from accounts."""
    pnl_dist = {
        "1m_plus": 0,
        "100k_plus": 0,
        "50k_plus": 0,
        "20k_plus": 0,
        "10k_plus": 0,
    }

    score_dist = defaultdict(int)
    activity_dist = {
        "active_7d": 0,
        "active_14d": 0,
        "active_30d": 0,
        "stale_30_90d": 0,
        "dead_90d_plus": 0,
    }
    views_dist = {
        "100k_plus": 0,
        "10k_plus": 0,
        "1k_plus": 0,
        "100_plus": 0,
        "under_100": 0,
    }

    for a in accounts:
        pnl = a.get("total_pnl", 0)
        if pnl >= 1000000:
            pnl_dist["1m_plus"] += 1
        if pnl >= 100000:
            pnl_dist["100k_plus"] += 1
        if pnl >= 50000:
            pnl_dist["50k_plus"] += 1
        if pnl >= 20000:
            pnl_dist["20k_plus"] += 1
        if pnl >= 10000:
            pnl_dist["10k_plus"] += 1

        score = a.get("systematic_score", 0)
        if score >= 90:
            score_dist["90-100"] += 1
        elif score >= 80:
            score_dist["80-89"] += 1
        elif score >= 70:
            score_dist["70-79"] += 1
        elif score >= 65:
            score_dist["65-69"] += 1
        else:
            score_dist["<65"] += 1

        recency = a.get("activity_recency_days", 999)
        if recency <= 7:
            activity_dist["active_7d"] += 1
        if recency <= 14:
            activity_dist["active_14d"] += 1
        if recency <= 30:
            activity_dist["active_30d"] += 1
        elif recency <= 90:
            activity_dist["stale_30_90d"] += 1
        else:
            activity_dist["dead_90d_plus"] += 1

        # Profile views distribution
        views = a.get("profile_views", 0)
        if views >= 100000:
            views_dist["100k_plus"] += 1
        elif views >= 10000:
            views_dist["10k_plus"] += 1
        elif views >= 1000:
            views_dist["1k_plus"] += 1
        elif views >= 100:
            views_dist["100_plus"] += 1
        else:
            views_dist["under_100"] += 1

    return {
        "pnl_distribution": pnl_dist,
        "score_distribution": dict(score_dist),
        "activity_distribution": activity_dist,
        "views_distribution": views_dist,
    }


async def run_mass_scan():
    """Run the mass scan to populate analyzed_accounts.json."""
    print("\n" + "=" * 70)
    print("  POLYMARKET MASS ACCOUNT SCAN")
    print("=" * 70)
    print(f"\nTarget: {MAX_CANDIDATES:,} accounts from leaderboard")
    print(f"Criteria:")
    print(f"  P/L Range: ${MIN_PNL:,} - ${MAX_PNL:,}")
    print(f"  Trade Range: {MIN_TRADES} - {MAX_TRADES}")
    print(f"  Trades to Analyze: {TRADES_TO_SCRAPE}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("-" * 70)

    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=MAX_CANDIDATES,
        min_profit=MIN_PNL,  # Minimum P/L filter
        max_profit=MAX_PNL,
        analyze_top_n=TRADES_TO_SCRAPE,
        max_trades=MAX_TRADES,
        min_trades=MIN_TRADES,
        min_score_threshold=0.0,  # Collect ALL accounts - filter in UI2
        max_phase2=MAX_CANDIDATES,  # Process all in phase 2
        max_phase3=MAX_CANDIDATES,  # Process all in phase 3
        persist_to_db=True,  # Enable checkpointing for resume capability
        checkpoint_interval=25,  # Save every 25 accounts
        enable_categorization=True,  # Fetch market data for accurate category detection
    )

    # Progress tracking
    last_update = [datetime.now()]

    def progress_callback(progress):
        """Handle ScanProgress object from service."""
        now = datetime.now()
        # Update every 5 seconds or on significant progress
        if (now - last_update[0]).total_seconds() >= 5 or progress.progress_pct in [25, 50, 75, 100]:
            elapsed = (now - start_time).total_seconds()
            phase = progress.current_phase
            pct = progress.progress_pct

            # Get phase-specific stats
            if phase == "phase1":
                total = progress.phase1_total or progress.candidates_found
                passed = progress.phase1_passed
            elif phase == "phase2":
                total = progress.phase2_total
                passed = progress.phase2_passed
            elif phase == "phase3":
                total = progress.phase3_total
                passed = progress.phase3_passed
            else:
                total = progress.candidates_found
                passed = progress.candidates_passed

            print(f"  [{phase.upper()}] {pct}% | {passed:,} passed | {progress.current_step}")
            last_update[0] = now

    print(f"\nStarting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    start_time = datetime.now()

    results = []
    analyzer_stats = {}

    try:
        async with DiscoveryService() as service:
            # Config is already set via ScanConfig - no manual override needed
            # min_score_threshold=0 ensures ALL accounts are collected
            # min_trades from config is applied automatically by run_scan

            results, scan = await service.run_scan(config, progress_callback=progress_callback)
            analyzer_stats = service._analyzer.get_stats()

            # Convert to dict format if needed
            if results and not isinstance(results[0], dict):
                results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]

    except Exception as e:
        print(f"\nERROR during scan: {e}")
        import traceback
        traceback.print_exc()
        return []

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("  SCAN COMPLETE")
    print("=" * 70)
    print(f"\nTime elapsed: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"\nPhase Statistics:")
    print(f"  Phase 1: {analyzer_stats.get('phase1_passed', 0):,}/{analyzer_stats.get('phase1_processed', 0):,} passed")
    print(f"  Phase 2: {analyzer_stats.get('phase2_passed', 0):,}/{analyzer_stats.get('phase2_processed', 0):,} passed")
    print(f"  Phase 3: {analyzer_stats.get('phase3_passed', 0):,}/{analyzer_stats.get('phase3_processed', 0):,} passed")
    print(f"  API Calls: {analyzer_stats.get('api_calls', 0):,}")
    print(f"\n  FINAL RESULTS: {len(results):,} accounts passed all filters")

    if not results:
        print("\nWARNING: No results! Check scan configuration.")
        return []

    # Debug: Show sample result structure
    if results and TEST_MODE:
        sample = results[0]
        print(f"\n  DEBUG - Sample result keys: {list(sample.keys())[:15]}...")
        print(f"  DEBUG - total_pnl: {sample.get('total_pnl')}")
        print(f"  DEBUG - pl_metrics.total_realized_pnl: {(sample.get('pl_metrics') or {}).get('total_realized_pnl')}")
        print(f"  DEBUG - pattern_metrics.days_since_last_trade: {(sample.get('pattern_metrics') or {}).get('days_since_last_trade')}")
        print(f"  DEBUG - pattern_metrics.primary_category: {(sample.get('pattern_metrics') or {}).get('primary_category')}")

    # Filter by P/L range (leaderboard may not have exact range)
    # total_pnl comes from leaderboard; fallback to pl_metrics.total_realized_pnl
    filtered_results = []
    for r in results:
        pnl = float(r.get("total_pnl") or (r.get("pl_metrics") or {}).get("total_realized_pnl") or 0)
        if MIN_PNL <= pnl <= MAX_PNL:
            filtered_results.append(r)

    print(f"  After P/L filter (${MIN_PNL:,}-${MAX_PNL:,}): {len(filtered_results):,} accounts")

    # Filter by recent trade activity if required
    if REQUIRE_RECENT_TRADE:
        pre_recency = len(filtered_results)
        recency_filtered = []
        for r in filtered_results:
            pattern = r.get("pattern_metrics") or {}
            days_since_trade = pattern.get("days_since_last_trade", 999)
            if days_since_trade <= RECENT_TRADE_DAYS:
                recency_filtered.append(r)
        filtered_results = recency_filtered
        print(f"  After recency filter (traded in last {RECENT_TRADE_DAYS}d): {len(filtered_results):,} accounts (filtered {pre_recency - len(filtered_results):,})")

    # Normalize to analyzed_accounts format
    print("\nNormalizing results to analyzed_accounts format...")
    normalized = [normalize_result(convert_decimals(r)) for r in filtered_results]

    # ================================================================
    # PHASE 4: Fetch Profile Views (parallel, rate-limited)
    # ================================================================
    print(f"\n[PHASE 4] Fetching profile views for {len(normalized):,} accounts...")
    print("  (This may take 10-15 minutes for large datasets)")

    views_start = datetime.now()
    profile_views = await fetch_all_profile_views(
        [a["wallet_address"] for a in normalized],
        max_concurrent=10,  # Reasonable rate limit
    )
    views_elapsed = (datetime.now() - views_start).total_seconds()

    # Attach views to normalized results
    views_found = 0
    for account in normalized:
        wallet = account["wallet_address"].lower()
        if wallet in profile_views:
            account["profile_views"] = profile_views[wallet].views
            account["profile_username"] = profile_views[wallet].username
            if profile_views[wallet].views > 0:
                views_found += 1
        else:
            account["profile_views"] = 0
            account["profile_username"] = None

    print(f"  Profile views fetched in {views_elapsed:.1f}s")
    print(f"  Accounts with views > 0: {views_found:,}/{len(normalized):,}")

    # Sort by systematic_score descending
    normalized.sort(key=lambda x: x.get("systematic_score", 0), reverse=True)

    # Calculate distributions
    distributions = calculate_distributions(normalized)

    # Build final output
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_collected": analyzer_stats.get('phase1_processed', 0),
        "total_analyzed": len(normalized),
        "active_count": distributions["activity_distribution"]["active_14d"],
        "systematic_count": len([a for a in normalized if a.get("systematic_score", 0) >= 65]),
        "quality_count": len([a for a in normalized if a.get("systematic_score", 0) >= 90]),
        "filters_applied": {
            "min_pnl": MIN_PNL,
            "max_pnl": MAX_PNL,
            "min_trades": MIN_TRADES,
            "max_trades": MAX_TRADES,
        },
        "activity_distribution": distributions["activity_distribution"],
        "pnl_distribution": distributions["pnl_distribution"],
        "score_distribution": distributions["score_distribution"],
        "views_distribution": distributions["views_distribution"],
        "accounts": normalized,
    }

    # Save to file
    print(f"\nSaving {len(normalized):,} accounts to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved successfully!")

    # Print summary
    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    if normalized:
        scores = [a["systematic_score"] for a in normalized]
        pnls = [a["total_pnl"] for a in normalized]
        print(f"\n  Accounts: {len(normalized):,}")
        print(f"  Score Range: {min(scores):.0f} - {max(scores):.0f} (avg: {sum(scores)/len(scores):.0f})")
        print(f"  P/L Range: ${min(pnls):,.0f} - ${max(pnls):,.0f}")
        print(f"\n  Score Distribution:")
        for k, v in distributions["score_distribution"].items():
            print(f"    {k}: {v:,}")

    return normalized


if __name__ == "__main__":
    results = asyncio.run(run_mass_scan())
    print(f"\n{'=' * 70}")
    print(f"  Mass scan complete. {len(results):,} accounts saved.")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'=' * 70}\n")
