#!/usr/bin/env python3
"""
Quick test to verify total_pnl is properly propagated through scan pipeline.
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode


async def test_pnl_propagation():
    print("=" * 60)
    print("  TESTING P/L PROPAGATION FIX")
    print("=" * 60)

    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=25,  # Small batch
        max_profit=100000,
        analyze_top_n=100,
        max_trades=500,
        min_trades=50,
        min_score_threshold=0.0,
        max_phase2=25,
        max_phase3=25,
        persist_to_db=False,  # No DB for quick test
    )

    print("\nRunning mini scan (25 candidates)...")

    try:
        async with DiscoveryService() as service:
            results, scan = await service.run_scan(config, progress_callback=lambda p: None)

            print(f"\nResults: {len(results)} accounts")

            if not results:
                print("\n⚠ No results returned!")
                return False

            # Check P/L values
            has_pnl = 0
            null_pnl = 0
            zero_pnl = 0

            print("\nSample results:")
            for i, r in enumerate(results[:10]):
                pnl = r.get("total_pnl")
                wallet = r.get("wallet_address", "")[:15]
                score = r.get("composite_score", 0)

                if pnl is None:
                    null_pnl += 1
                    status = "NULL"
                elif pnl == 0:
                    zero_pnl += 1
                    status = "ZERO"
                else:
                    has_pnl += 1
                    status = f"${pnl:,.0f}"

                print(f"  {i+1}. {wallet}... | Score: {score:.1f} | P/L: {status}")

            # Count all results
            for r in results[10:]:
                pnl = r.get("total_pnl")
                if pnl is None:
                    null_pnl += 1
                elif pnl == 0:
                    zero_pnl += 1
                else:
                    has_pnl += 1

            print(f"\n" + "-" * 60)
            print(f"  SUMMARY ({len(results)} accounts):")
            print(f"    With P/L: {has_pnl}")
            print(f"    Null P/L: {null_pnl}")
            print(f"    Zero P/L: {zero_pnl}")
            print("-" * 60)

            if null_pnl > 0:
                print("\n  ⚠ STILL HAVE NULL P/L VALUES - FIX NOT WORKING")
                return False
            elif has_pnl > 0:
                print("\n  ✓ P/L PROPAGATION WORKING - FIX SUCCESSFUL!")
                return True
            else:
                print("\n  ⚠ ALL ZERO P/L - May be expected if accounts are filtered")
                return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_pnl_propagation())
    sys.exit(0 if success else 1)
