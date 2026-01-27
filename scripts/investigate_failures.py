#!/usr/bin/env python3
"""
Investigate why some accounts fail Phase 3.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig, LeaderboardClient
from src.discovery.models import DiscoveryMode


async def investigate():
    print("=" * 70)
    print("  INVESTIGATING PHASE 3 FAILURES")
    print("=" * 70)

    # Get wallets from MULTIPLE categories like mass_scan does
    print("\n[1] Fetching wallets from multiple categories...")
    all_wallets = set()
    async with LeaderboardClient() as lb_client:
        for category in ['WEATHER', 'ECONOMICS', 'TECH', 'CRYPTO', 'OVERALL']:
            entries = await lb_client.get_leaderboard(category=category, limit=10)
            for e in entries:
                wallet = e.get('proxyWallet', e.get('wallet', ''))
                if wallet:
                    all_wallets.add(wallet)
            print(f"  {category}: {len(entries)} entries")

    wallets = list(all_wallets)[:40]  # Limit to 40 for test
    print(f"  Total unique wallets: {len(wallets)}")

    # Same config as mass_scan
    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=30,
        max_profit=100000,
        analyze_top_n=500,
        max_trades=3000,
        min_trades=75,
        min_score_threshold=0.0,
        max_phase2=30,
        max_phase3=30,
        persist_to_db=False,
    )

    async with DiscoveryService() as service:
        # Set up config
        service._analyzer.set_mode(config.mode)
        scoring_updates = {
            "min_composite_score": config.min_score_threshold,
        }
        if config.min_trades:
            scoring_updates["hard_filters"] = {
                "min_trades": {"threshold": config.min_trades}
            }
        service._analyzer.scoring_engine.update_config(scoring_updates)

        print(f"\n[2] Config:")
        sc = service._analyzer.scoring_engine.config
        print(f"  min_composite_score: {sc.min_composite_score}")
        print(f"  min_trades threshold: {sc.hard_filters['min_trades'].threshold}")

        # Run batch analysis directly
        print(f"\n[3] Running batch_deep_analysis on {len(wallets)} wallets...")
        results = await service._analyzer.batch_deep_analysis(
            wallets,
            lookback_days=1825,
            max_concurrent=3,
            max_trades=500,
        )

        # Categorize results
        passed = []
        failed_hard = []
        failed_soft = []
        no_score = []

        for r in results:
            if not r.scoring_result:
                no_score.append(r)
            elif not r.scoring_result.hard_filter_passed:
                failed_hard.append(r)
            elif not r.scoring_result.passes_threshold:
                failed_soft.append(r)
            else:
                passed.append(r)

        print(f"\n[4] Results breakdown:")
        print(f"  Passed: {len(passed)}")
        print(f"  Failed hard filter: {len(failed_hard)}")
        print(f"  Failed soft filter (score < threshold): {len(failed_soft)}")
        print(f"  No scoring result: {len(no_score)}")

        # Show hard filter failures
        if failed_hard:
            print(f"\n[5] Hard Filter Failures:")
            for r in failed_hard[:10]:
                wallet = r.wallet_address[:16]
                failures = r.scoring_result.hard_filter_failures
                trades = r.pattern_metrics.total_trades if r.pattern_metrics else 0
                print(f"  {wallet}... | Trades: {trades} | Failures: {failures}")

        # Show soft filter failures (shouldn't happen with min_composite_score=0)
        if failed_soft:
            print(f"\n[6] Soft Filter Failures (score < {sc.min_composite_score}):")
            for r in failed_soft[:10]:
                wallet = r.wallet_address[:16]
                score = r.scoring_result.composite_score
                trades = r.pattern_metrics.total_trades if r.pattern_metrics else 0
                print(f"  {wallet}... | Score: {score:.1f} | Trades: {trades}")
                print(f"    This should NOT happen with min_composite_score=0!")

        # Show no score cases
        if no_score:
            print(f"\n[7] No Scoring Result:")
            for r in no_score[:10]:
                wallet = r.wallet_address[:16]
                print(f"  {wallet}... | Error: {r.error}")

        # Show passed examples
        if passed:
            print(f"\n[8] Passed Examples:")
            for r in passed[:5]:
                wallet = r.wallet_address[:16]
                score = r.scoring_result.composite_score
                trades = r.pattern_metrics.total_trades if r.pattern_metrics else 0
                print(f"  {wallet}... | Score: {score:.1f} | Trades: {trades}")

        # Check if the failures are valid
        print("\n" + "-" * 70)
        print("  ANALYSIS")
        print("-" * 70)

        if failed_hard:
            print(f"\n  Hard filter failures are VALID if:")
            print(f"    - min_trades threshold = {sc.hard_filters['min_trades'].threshold}")
            print(f"    - Accounts with fewer trades should be rejected")

            # Verify failures
            invalid_failures = [r for r in failed_hard
                                if r.pattern_metrics and r.pattern_metrics.total_trades >= config.min_trades]
            if invalid_failures:
                print(f"\n  ⚠ WARNING: {len(invalid_failures)} accounts failed incorrectly!")
                for r in invalid_failures[:5]:
                    print(f"    {r.wallet_address[:16]}: {r.pattern_metrics.total_trades} trades >= {config.min_trades}")
            else:
                print(f"\n  ✓ All hard filter failures are valid (trades < {config.min_trades})")

        if failed_soft:
            print(f"\n  ⚠ WARNING: Soft filter failures should NOT occur with min_composite_score=0")

        if not failed_hard and not failed_soft and not no_score:
            print(f"\n  ✓ All accounts passed! No failures to investigate.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(investigate())
