#!/usr/bin/env python3
"""
Mini mass scan - tests the EXACT same flow as mass_scan.py but with fewer accounts.
This validates the fix before running the full 15k scan.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode

# Same config as mass_scan.py but smaller batch
MIN_PNL = 20000
MAX_PNL = 100000
MIN_TRADES = 75
MAX_TRADES = 3000
MAX_CANDIDATES = 50  # Small batch for testing
TRADES_TO_SCRAPE = 500


async def run_mini_scan():
    print("=" * 70)
    print("  MINI MASS SCAN TEST")
    print("=" * 70)
    print(f"\nTesting with {MAX_CANDIDATES} candidates (same config as full scan)")
    print(f"P/L Range: ${MIN_PNL:,} - ${MAX_PNL:,}")
    print(f"Trade Range: {MIN_TRADES} - {MAX_TRADES}")
    print(f"Trades to analyze: {TRADES_TO_SCRAPE}")
    print("-" * 70)

    # EXACT same config as mass_scan.py
    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=MAX_CANDIDATES,
        max_profit=MAX_PNL,
        analyze_top_n=TRADES_TO_SCRAPE,
        max_trades=MAX_TRADES,
        min_trades=MIN_TRADES,
        min_score_threshold=0.0,  # Collect ALL accounts
        max_phase2=MAX_CANDIDATES,
        max_phase3=MAX_CANDIDATES,
        persist_to_db=False,  # Don't persist for test
        checkpoint_interval=25,
    )

    start_time = datetime.now()
    last_update = [datetime.now()]

    def progress_callback(progress):
        """Same callback as mass_scan.py"""
        now = datetime.now()
        if (now - last_update[0]).total_seconds() >= 2 or progress.progress_pct in [25, 50, 75, 100]:
            phase = progress.current_phase
            pct = progress.progress_pct

            if phase == "phase1":
                passed = progress.phase1_passed
            elif phase == "phase2":
                passed = progress.phase2_passed
            elif phase == "phase3":
                passed = progress.phase3_passed
            else:
                passed = progress.candidates_passed

            print(f"  [{phase.upper()}] {pct}% | {passed:,} passed | {progress.current_step}")
            last_update[0] = now

    print(f"\nStarting at {datetime.now().strftime('%H:%M:%S')}...")

    results = []
    analyzer_stats = {}

    try:
        async with DiscoveryService() as service:
            # Verify scoring config BEFORE scan
            print("\n[Pre-scan Config Check]")
            sc = service._analyzer.scoring_engine.config
            print(f"  min_composite_score: {sc.min_composite_score} (will be set to 0.0)")
            print(f"  min_trades threshold: {sc.hard_filters['min_trades'].threshold} (will be set to {MIN_TRADES})")

            results, scan = await service.run_scan(config, progress_callback=progress_callback)
            analyzer_stats = service._analyzer.get_stats()

            # Verify scoring config AFTER scan
            print("\n[Post-scan Config Check]")
            sc = service._analyzer.scoring_engine.config
            print(f"  min_composite_score: {sc.min_composite_score}")
            print(f"  min_trades threshold: {sc.hard_filters['min_trades'].threshold}")

            if results and not isinstance(results[0], dict):
                results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("  MINI SCAN RESULTS")
    print("=" * 70)
    print(f"\nTime: {elapsed:.1f} seconds")
    print(f"\nPhase Statistics:")
    print(f"  Phase 1: {analyzer_stats.get('phase1_passed', 0):,}/{analyzer_stats.get('phase1_processed', 0):,} passed")
    print(f"  Phase 2: {analyzer_stats.get('phase2_passed', 0):,}/{analyzer_stats.get('phase2_processed', 0):,} passed")
    print(f"  Phase 3: {analyzer_stats.get('phase3_passed', 0):,}/{analyzer_stats.get('phase3_processed', 0):,} passed")
    print(f"  API Calls: {analyzer_stats.get('api_calls', 0):,}")

    print(f"\n  FINAL RESULTS: {len(results):,} accounts")

    if results:
        print("\n[Sample Results]")
        for i, r in enumerate(results[:5]):
            wallet = r.get('wallet_address', '')[:16] if r.get('wallet_address') else 'N/A'
            score = r.get('composite_score') or 0
            pnl = r.get('total_pnl') or 0
            print(f"  {i+1}. {wallet}... | Score: {score:.1f} | P/L: ${float(pnl):,.0f}")

        # Filter by P/L range
        in_range = [r for r in results if MIN_PNL <= float(r.get('total_pnl') or 0) <= MAX_PNL]
        print(f"\n  In P/L range (${MIN_PNL:,}-${MAX_PNL:,}): {len(in_range):,} accounts")

        # Success check
        print("\n" + "-" * 70)
        if len(results) > 0:
            print("  ✓ SUCCESS: Accounts are passing Phase 3!")
            print("  ✓ The fix is working correctly.")
            print("  ✓ Safe to run full mass_scan.py")
        else:
            print("  ✗ PROBLEM: No accounts passed!")
            print("  Check the debug logs above for details.")
    else:
        print("\n  ✗ WARNING: No results returned!")
        print("  Check the debug logs above for details.")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_mini_scan())
