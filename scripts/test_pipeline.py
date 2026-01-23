#!/usr/bin/env python3
"""Test the full discovery pipeline end-to-end."""

import asyncio
import sys

sys.path.insert(0, "/home/user/Documents/polymarketbot")

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode


async def progress_callback(progress):
    """Print progress updates."""
    print(f"[{progress.progress_pct:3d}%] {progress.current_step}")


async def main():
    print("=" * 60)
    print("Full Pipeline Test")
    print("=" * 60)

    config = ScanConfig(
        mode=DiscoveryMode.NICHE_SPECIALIST,
        categories=["ECONOMICS"],
        max_candidates=30,
        max_phase2=15,
        max_phase3=8,
        lookback_days=90,
        min_score_threshold=25,  # Low threshold to get results
    )

    print(f"Config: mode={config.mode.value}, categories={config.categories}")
    print(f"Max: {config.max_candidates} -> {config.max_phase2} -> {config.max_phase3}")
    print()

    async with DiscoveryService() as service:
        results, scan = await service.run_scan(config, progress_callback)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Status: {scan.status.value}")
        print(f"Duration: {scan.duration_seconds}s")
        print(f"Pipeline: {scan.candidates_found} -> {scan.candidates_analyzed} -> {scan.candidates_passed}")

        if scan.errors:
            print(f"Errors: {scan.errors[:3]}...")

        if results:
            print(f"\n*** DISCOVERED {len(results)} ACCOUNTS ***")
            for r in results[:5]:
                print(f"\n  Wallet: {r['wallet_address'][:20]}...")
                print(f"  Score: {r['composite_score']:.1f}")
                print(f"  Red flags: {r.get('red_flag_count', 0)}")
                if r.get('pl_metrics'):
                    print(f"  90-day PNL: ${r['pl_metrics'].get('total_realized_pnl', 0)}")
                if r.get('pattern_metrics'):
                    print(f"  Trades: {r['pattern_metrics'].get('total_trades', 0)}")
        else:
            print("\nNo accounts discovered.")

        stats = service.get_scan_stats()
        print(f"\nAPI calls: {stats.get('api_calls', 0)}")

    print("\n" + "=" * 60)
    print("Pipeline test complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
