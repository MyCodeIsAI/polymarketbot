#!/usr/bin/env python3
"""Simple test script for discovery pipeline with detailed debug."""

import asyncio
import sys

sys.path.insert(0, "/home/user/Documents/polymarketbot")

from src.discovery.service import DiscoveryService, ScanConfig, LeaderboardClient
from src.discovery.models import DiscoveryMode
from src.discovery.analyzer import AccountAnalyzer, LeaderboardEntry
from src.api.data import DataAPIClient
from src.api.gamma import GammaAPIClient


async def progress_callback(progress):
    """Print progress updates."""
    print(f"[{progress.progress_pct:3d}%] {progress.current_step}")


async def main():
    print("=" * 60)
    print("Discovery Pipeline Test with Detailed Debug")
    print("=" * 60)

    # Run direct deep analysis on a few accounts to debug scoring
    async with DataAPIClient() as data_client, GammaAPIClient() as gamma_client:
        analyzer = AccountAnalyzer(data_client, gamma_client, mode=DiscoveryMode.NICHE_SPECIALIST)
        await analyzer.__aenter__()

        # Get some candidates from leaderboard
        async with LeaderboardClient() as lb_client:
            entries = await lb_client.get_leaderboard_full(category="ECONOMICS", limit=5)

        print(f"\nAnalyzing {len(entries)} ECONOMICS leaderboard accounts...\n")

        for entry in entries[:3]:
            print("-" * 60)
            print(f"Wallet: {entry.wallet_address[:20]}...")
            print(f"Leaderboard PNL: ${entry.total_pnl:,.2f}")
            print(f"Leaderboard trades: {entry.num_trades}")

            result = await analyzer.deep_analysis(entry.wallet_address, lookback_days=90)

            if result.error:
                print(f"Error: {result.error}")
                continue

            sr = result.scoring_result
            if sr:
                print(f"\n*** SCORING RESULT ***")
                print(f"Composite score: {sr.composite_score:.1f}")
                print(f"Passes threshold (>={analyzer.scoring_engine.config.min_composite_score}): {sr.passes_threshold}")
                print(f"Hard filter passed: {sr.hard_filter_passed}")

                if not sr.hard_filter_passed:
                    print(f"HARD FILTER FAILURES: {sr.hard_filter_failures}")

                print(f"\nComponent scores:")
                print(f"  PL consistency: {sr.pl_consistency_score:.1f}")
                print(f"  Pattern match: {sr.pattern_match_score:.1f}")
                print(f"  Specialization: {sr.specialization_score:.1f}")
                print(f"  Risk: {sr.risk_score:.1f}")

                print(f"\nRed flags ({sr.red_flag_count}):")
                for flag in sr.red_flags:
                    print(f"  - [{flag.get('severity')}] {flag.get('type')}: {flag.get('description')}")

            pl = result.pl_metrics
            if pl:
                print(f"\nPL Metrics (90-day):")
                print(f"  Total realized PNL: ${pl.total_realized_pnl:,.2f}")
                print(f"  Sharpe: {pl.sharpe_ratio:.2f}")
                print(f"  Win rate: {pl.win_rate*100:.1f}%")
                print(f"  Profit factor: {pl.profit_factor:.2f}")
                print(f"  Max drawdown: {pl.max_drawdown_pct*100:.1f}%")
                print(f"  Largest win % of total: {pl.largest_win_pct_of_total*100:.1f}%")

            pattern = result.pattern_metrics
            if pattern:
                print(f"\nPattern Metrics:")
                print(f"  Total trades (90d): {pattern.total_trades}")
                print(f"  Account age: {pattern.account_age_days} days")
                print(f"  Avg position: ${pattern.avg_position_size_usd:.2f}")
                print(f"  Niche market %: {pattern.niche_market_pct*100:.1f}%")
                mainstream = sum(pattern.category_breakdown.get(c, 0) for c in ['politics', 'sports'])
                print(f"  Mainstream %: {mainstream*100:.1f}%")
                print(f"  Categories: {pattern.category_breakdown}")

            # Debug: show sample market titles
            if result.trades:
                print(f"\nSample market titles:")
                unique_titles = set(t.market_title for t in result.trades[:20])
                for title in list(unique_titles)[:5]:
                    print(f"  - {title[:80]}")

            print()

        await analyzer.__aexit__(None, None, None)

    print("=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
