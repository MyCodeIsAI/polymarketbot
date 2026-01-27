#!/usr/bin/env python3
"""
Diagnostic script to debug Phase 3 zero-pass issue.

Tests the exact scoring flow used in run_scan to identify why passes_threshold is False.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode


async def diagnose():
    """Run diagnostic analysis."""
    print("=" * 70)
    print("  PHASE 3 DIAGNOSTIC")
    print("=" * 70)

    # Same config as mass_scan.py
    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=15000,
        max_profit=100000,
        analyze_top_n=500,
        max_trades=3000,
        min_trades=75,
        min_score_threshold=0.0,  # Should be 0!
        max_phase2=15000,
        max_phase3=15000,
        persist_to_db=False,
    )

    print(f"\n[Config]")
    print(f"  min_score_threshold: {config.min_score_threshold}")
    print(f"  min_trades: {config.min_trades}")
    print(f"  analyze_top_n: {config.analyze_top_n}")

    async with DiscoveryService() as service:
        # Exactly what run_scan does
        print("\n[Step 1] Setting mode...")
        service._analyzer.set_mode(config.mode)

        # Check what the default config looks like after set_mode
        scoring_config = service._analyzer.scoring_engine.config
        print(f"  After set_mode:")
        print(f"    min_composite_score: {scoring_config.min_composite_score}")
        print(f"    min_trades threshold: {scoring_config.hard_filters['min_trades'].threshold}")

        # Now apply the update_config
        print("\n[Step 2] Applying scoring updates...")
        scoring_updates = {
            "min_composite_score": config.min_score_threshold,
        }
        if config.min_trades:
            scoring_updates["hard_filters"] = {
                "min_trades": {"threshold": config.min_trades}
            }

        print(f"  Applying: {scoring_updates}")
        service._analyzer.scoring_engine.update_config(scoring_updates)

        # Check after update
        scoring_config = service._analyzer.scoring_engine.config
        print(f"  After update_config:")
        print(f"    min_composite_score: {scoring_config.min_composite_score}")
        print(f"    min_trades threshold: {scoring_config.hard_filters['min_trades'].threshold}")

        # Get a real wallet from the leaderboard
        print("\n[Step 3] Fetching test wallet from leaderboard...")
        from src.discovery.service import LeaderboardClient
        async with LeaderboardClient() as lb_client:
            entries = await lb_client.get_leaderboard(category='OVERALL', limit=50)
            # Find one in the $20k-$100k range
            test_wallet = None
            for e in entries:
                pnl = float(e.get('pnl', e.get('totalPnl', 0)))
                if 20000 <= pnl <= 100000:
                    test_wallet = e.get('proxyWallet', e.get('wallet', ''))
                    print(f"  Found wallet with P/L ${pnl:,.0f}")
                    break

            # If no wallet in range, use the first one that has activity
            if not test_wallet:
                print("  No wallet in P/L range, using first leaderboard wallet")
                test_wallet = entries[0].get('proxyWallet', entries[0].get('wallet', ''))

        print(f"  Testing wallet: {test_wallet[:16]}...")

        # Deep analysis
        result = await service._analyzer.deep_analysis(
            test_wallet,
            lookback_days=1825,
            max_trades=500,
        )

        print(f"\n[Result]")
        print(f"  wallet: {result.wallet_address[:16]}...")
        print(f"  error: {result.error}")
        print(f"  pl_metrics: {type(result.pl_metrics).__name__ if result.pl_metrics else None}")
        print(f"  pattern_metrics: {type(result.pattern_metrics).__name__ if result.pattern_metrics else None}")
        print(f"  scoring_result: {type(result.scoring_result).__name__ if result.scoring_result else None}")

        if result.pattern_metrics:
            print(f"\n  [Pattern Metrics]")
            print(f"    total_trades: {result.pattern_metrics.total_trades}")
            print(f"    unique_markets: {result.pattern_metrics.unique_markets_traded}")
            print(f"    account_age_days: {result.pattern_metrics.account_age_days}")

        if result.scoring_result:
            sr = result.scoring_result
            print(f"\n  [Scoring Result]")
            print(f"    composite_score: {sr.composite_score:.2f}")
            print(f"    passes_threshold: {sr.passes_threshold}")
            print(f"    hard_filter_passed: {sr.hard_filter_passed}")
            print(f"    hard_filter_failures: {sr.hard_filter_failures}")
            print(f"    pl_consistency_score: {sr.pl_consistency_score:.2f}")
            print(f"    pattern_match_score: {sr.pattern_match_score:.2f}")

            # Check the comparison manually
            print(f"\n  [Manual Check]")
            print(f"    composite ({sr.composite_score:.2f}) >= min_composite_score ({scoring_config.min_composite_score})?")
            print(f"    Answer: {sr.composite_score >= scoring_config.min_composite_score}")

            # Filter logic
            passes_filter = result.scoring_result and result.scoring_result.passes_threshold
            print(f"\n  [Filter Logic]")
            print(f"    scoring_result exists: {result.scoring_result is not None}")
            print(f"    passes_threshold: {result.scoring_result.passes_threshold if result.scoring_result else 'N/A'}")
            print(f"    FINAL: Would pass filter: {passes_filter}")

        else:
            print(f"\n  [ERROR] scoring_result is None!")
            print(f"  This wallet would FAIL the filter in run_scan")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(diagnose())
