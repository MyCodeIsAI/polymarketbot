#!/usr/bin/env python3
"""
Test scan script to verify the discovery pipeline.
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode


async def run_test_scan():
    """Run a test scan with specific criteria."""
    print("\n" + "=" * 70)
    print("  DISCOVERY SCAN TEST")
    print("=" * 70)

    # User's criteria
    MIN_PNL = 20000
    MAX_PNL = 100000

    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=100,  # 100 accounts total
        max_profit=MAX_PNL,  # MAX P/L: $100,000
        analyze_top_n=500,   # Trades to scrape: 500
        max_trades=3000,     # Max Trades: 3000
        min_trades=75,       # Min Trades: 75
        max_phase2=100,      # Process all 100 in phase 2
        max_phase3=100,      # Process all in phase 3
        persist_to_db=False, # Don't persist for this test
    )

    print(f"\nScan Configuration:")
    print(f"  Mode: {config.mode.value}")
    print(f"  Max Candidates: {config.max_candidates}")
    print(f"  Min P/L: ${MIN_PNL:,}")
    print(f"  Max P/L: ${config.max_profit:,}")
    print(f"  Min Trades: {config.min_trades}")
    print(f"  Max Trades: {config.max_trades}")
    print(f"  Analyze Top N: {config.analyze_top_n}")
    print("-" * 70)

    # Track progress
    def progress_callback(phase: str, completed: int, passed: int, total: int):
        print(f"  [{phase.upper()}] {completed}/{total} processed, {passed} passed")

    # Run the scan
    print(f"\nStarting scan at {datetime.now().strftime('%H:%M:%S')}...")
    start_time = datetime.now()

    results = []
    analyzer_stats = {}

    try:
        # Use async context manager
        async with DiscoveryService() as service:
            # Update scoring engine - only min_trades (wide_net doesn't have must_be_profitable)
            # P/L filtering happens at Phase 1 via leaderboard already
            service._analyzer.scoring_engine.update_config({
                "hard_filters": {
                    "min_trades": {"threshold": config.min_trades},
                }
            })

            results, scan = await service.run_scan(config, progress_callback=progress_callback)
            analyzer_stats = service._analyzer.get_stats()

            # Results are dicts from to_dict(), extract properly
            if results and isinstance(results[0], dict):
                pass  # Already in dict format
            else:
                # Convert to dict if needed
                results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]

    except Exception as e:
        print(f"\n  ERROR during scan: {e}")
        import traceback
        traceback.print_exc()
        return []

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("  SCAN COMPLETE")
    print("=" * 70)
    print(f"\nTime elapsed: {elapsed:.1f} seconds")

    print(f"\nPhase Statistics:")
    print(f"  Phase 1 (Quick Filter): {analyzer_stats.get('phase1_passed', 0)}/{analyzer_stats.get('phase1_processed', 0)} passed")
    print(f"  Phase 2 (Light Scan):   {analyzer_stats.get('phase2_passed', 0)}/{analyzer_stats.get('phase2_processed', 0)} passed")
    print(f"  Phase 3 (Deep Analysis): {analyzer_stats.get('phase3_passed', 0)}/{analyzer_stats.get('phase3_processed', 0)} passed")
    print(f"  API Calls Made: {analyzer_stats.get('api_calls', 0)}")

    print(f"\n  FINAL RESULTS: {len(results)} accounts passed all filters")

    # Analyze results
    if results:
        print("\n" + "-" * 70)
        print("  RESULT DETAILS")
        print("-" * 70)

        for i, result in enumerate(results[:20], 1):  # Show first 20
            wallet = result.get("wallet_address", "unknown")[:10]
            score = result.get("composite_score", 0)
            passes = result.get("passes_threshold", False)

            # Get P/L metrics
            pl_metrics = result.get("pl_metrics", {})
            computed_pnl = float(pl_metrics.get("total_realized_pnl", 0))

            # Get leaderboard P/L if available
            leaderboard_pnl = float(result.get("total_pnl") or 0)

            # Get pattern metrics
            pattern = result.get("pattern_metrics", {})
            trades = pattern.get("total_trades", 0)
            win_rate = pattern.get("win_rate", 0) * 100 if pattern.get("win_rate") else 0

            # Get hard filter info
            score_breakdown = result.get("score_breakdown", {})
            hard_passed = score_breakdown.get("hard_filter_passed", True)
            hard_failures = score_breakdown.get("hard_filter_failures", [])

            print(f"\n  [{i}] {wallet}...")
            print(f"      Composite Score: {score:.1f} | Passes: {passes}")
            print(f"      Leaderboard P/L: ${leaderboard_pnl:,.0f}")
            print(f"      Computed P/L: ${computed_pnl:,.0f}")
            print(f"      Trades: {trades} | Win Rate: {win_rate:.1f}%")
            print(f"      Hard Filter: {'PASSED' if hard_passed else 'FAILED'}")
            if hard_failures:
                print(f"      Failures: {hard_failures}")

        # Summary statistics
        print("\n" + "-" * 70)
        print("  SUMMARY STATISTICS")
        print("-" * 70)

        scores = [r.get("composite_score", 0) for r in results]
        pnls = [float(r.get("total_pnl") or 0) for r in results]
        trades_list = [r.get("pattern_metrics", {}).get("total_trades", 0) for r in results]

        print(f"\n  Composite Scores:")
        print(f"    Min: {min(scores):.1f} | Max: {max(scores):.1f} | Avg: {sum(scores)/len(scores):.1f}")

        print(f"\n  Leaderboard P/L:")
        print(f"    Min: ${min(pnls):,.0f} | Max: ${max(pnls):,.0f} | Avg: ${sum(pnls)/len(pnls):,.0f}")

        if any(trades_list):
            valid_trades = [t for t in trades_list if t > 0]
            if valid_trades:
                print(f"\n  Trade Counts:")
                print(f"    Min: {min(valid_trades)} | Max: {max(valid_trades)} | Avg: {sum(valid_trades)/len(valid_trades):.0f}")

        # Check for any issues
        print("\n" + "-" * 70)
        print("  VALIDATION CHECKS")
        print("-" * 70)

        issues = []

        # Check P/L range
        for r in results:
            pnl = float(r.get("total_pnl") or 0)
            if pnl < MIN_PNL:
                issues.append(f"  - {r.get('wallet_address', '')[:10]}: P/L ${pnl:,.0f} below ${MIN_PNL:,} minimum")
            if pnl > MAX_PNL:
                issues.append(f"  - {r.get('wallet_address', '')[:10]}: P/L ${pnl:,.0f} above ${MAX_PNL:,} maximum")

        # Check trade counts
        for r in results:
            trades = r.get("pattern_metrics", {}).get("total_trades", 0)
            if trades > 0 and trades < 75:
                issues.append(f"  - {r.get('wallet_address', '')[:10]}: {trades} trades below 75 minimum")
            if trades > 3000:
                issues.append(f"  - {r.get('wallet_address', '')[:10]}: {trades} trades above 3000 maximum")

        # Check passes_threshold
        for r in results:
            if not r.get("passes_threshold", False):
                issues.append(f"  - {r.get('wallet_address', '')[:10]}: passes_threshold=False but in results")

        if issues:
            print(f"\n  Found {len(issues)} issues:")
            for issue in issues[:10]:
                print(issue)
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("\n  All validation checks PASSED")

        # Save results to file for inspection
        output_path = PROJECT_ROOT / "data" / "test_scan_results.json"
        with open(output_path, "w") as f:
            # Convert Decimals to floats for JSON
            def convert(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj

            json.dump(convert(results), f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    else:
        print("\n  WARNING: No results returned!")
        print("  This could indicate an issue with the scan pipeline.")

    return results


if __name__ == "__main__":
    results = asyncio.run(run_test_scan())
    print(f"\n{'=' * 70}")
    print(f"  Test complete. {len(results)} accounts in final results.")
    print(f"{'=' * 70}\n")
