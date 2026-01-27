#!/usr/bin/env python3
"""
Test the batch deep analysis flow with a few wallets.
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig, LeaderboardClient
from src.discovery.models import DiscoveryMode


async def test_batch():
    print("=" * 70)
    print("  BATCH FLOW TEST")
    print("=" * 70)

    # Get some test wallets from leaderboard
    print("\n[1] Fetching wallets from leaderboard...")
    async with LeaderboardClient() as lb_client:
        entries = await lb_client.get_leaderboard(category='OVERALL', limit=50)
        # Get 5 wallets that have trades
        test_wallets = []
        for e in entries[:10]:
            wallet = e.get('proxyWallet', e.get('wallet', ''))
            if wallet:
                test_wallets.append(wallet)
                if len(test_wallets) >= 5:
                    break

    print(f"  Got {len(test_wallets)} wallets")
    for w in test_wallets:
        print(f"    - {w[:16]}...")

    # Create config same as mass_scan.py
    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=15000,
        max_profit=100000,
        analyze_top_n=500,
        max_trades=3000,
        min_trades=75,
        min_score_threshold=0.0,
        max_phase2=15000,
        max_phase3=15000,
        persist_to_db=False,
    )

    async with DiscoveryService() as service:
        # Apply the same config setup as run_scan
        print("\n[2] Setting up scoring config...")
        service._analyzer.set_mode(config.mode)

        scoring_updates = {
            "min_composite_score": config.min_score_threshold,
        }
        if config.min_trades:
            scoring_updates["hard_filters"] = {
                "min_trades": {"threshold": config.min_trades}
            }
        service._analyzer.scoring_engine.update_config(scoring_updates)

        sc = service._analyzer.scoring_engine.config
        print(f"  min_composite_score: {sc.min_composite_score}")
        print(f"  min_trades: {sc.hard_filters['min_trades'].threshold}")

        # Run batch deep analysis
        print("\n[3] Running batch_deep_analysis...")
        results = await service._analyzer.batch_deep_analysis(
            test_wallets,
            lookback_days=1825,
            max_concurrent=3,
            max_trades=500,
        )

        # Check results
        print(f"\n[4] Results:")
        print(f"  Total results: {len(results)}")

        passed = []
        failed = []
        no_score = []

        for r in results:
            if r.scoring_result:
                if r.scoring_result.passes_threshold:
                    passed.append(r)
                else:
                    failed.append(r)
            else:
                no_score.append(r)

        print(f"  Passed: {len(passed)}")
        print(f"  Failed: {len(failed)}")
        print(f"  No score: {len(no_score)}")

        # Show details
        print("\n[5] Detailed results:")
        for r in results:
            wallet = r.wallet_address[:16]
            if r.scoring_result:
                print(f"  {wallet}: score={r.scoring_result.composite_score:.1f}, passes={r.scoring_result.passes_threshold}, hard_passed={r.scoring_result.hard_filter_passed}")
                if r.scoring_result.hard_filter_failures:
                    print(f"    Failures: {r.scoring_result.hard_filter_failures}")
            else:
                print(f"  {wallet}: NO SCORE, error={r.error}")

        # Test the filter logic used in run_scan
        print("\n[6] Filter test (as used in run_scan):")
        passed_results = [
            r for r in results
            if r.scoring_result and r.scoring_result.passes_threshold
        ]
        print(f"  passed_results count: {len(passed_results)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_batch())
