#!/usr/bin/env python3
"""Collect ALL accounts with >$5000 profit from Polymarket.

No hardcaps. Paginates through all leaderboard categories until PnL drops below threshold.
Stores everything in database for later analysis.
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import LeaderboardClient, LEADERBOARD_CATEGORIES
from src.discovery.analyzer import LeaderboardEntry

# Configuration
MIN_PROFIT_USD = 5000
BATCH_SIZE = 50
MAX_PER_CATEGORY = 10000  # Safety limit per category (should never hit this)


async def collect_profitable_accounts(min_profit: float = MIN_PROFIT_USD):
    """Collect all accounts with profit >= min_profit from all categories."""

    all_accounts: dict[str, LeaderboardEntry] = {}  # wallet -> entry (deduped)
    stats = {cat: 0 for cat in LEADERBOARD_CATEGORIES}

    print(f"\n{'='*60}")
    print(f"Collecting ALL accounts with >= ${min_profit:,.0f} profit")
    print(f"Categories: {len(LEADERBOARD_CATEGORIES)}")
    print(f"{'='*60}\n")

    async with LeaderboardClient() as client:
        for category in LEADERBOARD_CATEGORIES:
            print(f"[{category:12}] Scanning...", end="", flush=True)

            offset = 0
            category_count = 0
            lowest_pnl_seen = float('inf')

            while offset < MAX_PER_CATEGORY:
                try:
                    # Fetch batch from leaderboard (sorted by PnL descending)
                    raw_entries = await client.get_leaderboard(
                        category=category,
                        time_period="ALL",
                        order_by="PNL",
                        limit=BATCH_SIZE,
                        offset=offset,
                    )

                    if not raw_entries:
                        break

                    batch_added = 0
                    should_stop = False

                    for raw in raw_entries:
                        pnl = float(raw.get("pnl", 0))
                        lowest_pnl_seen = min(lowest_pnl_seen, pnl)

                        # Stop if we've dropped below threshold
                        if pnl < min_profit:
                            should_stop = True
                            break

                        wallet = raw.get("proxyWallet", "").lower()
                        if not wallet or wallet in all_accounts:
                            continue

                        # Add to collection
                        entry = LeaderboardEntry(
                            wallet_address=wallet,
                            rank=int(raw.get("rank", 0)),
                            total_pnl=Decimal(str(pnl)),
                            volume=Decimal(str(raw.get("vol", 0))),
                            num_trades=0,  # Not in leaderboard response
                            position_count=0,
                            categories=[category],
                        )
                        all_accounts[wallet] = entry
                        batch_added += 1
                        category_count += 1

                    if should_stop:
                        break

                    offset += BATCH_SIZE
                    await asyncio.sleep(0.03)  # Rate limit

                except Exception as e:
                    print(f" ERROR: {e}")
                    break

            stats[category] = category_count
            print(f" {category_count:4} accounts (lowest PnL seen: ${lowest_pnl_seen:,.0f})")

    return all_accounts, stats


async def main():
    start = datetime.now()

    accounts, stats = await collect_profitable_accounts(MIN_PROFIT_USD)

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total unique accounts: {len(accounts):,}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nBy category:")
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat:12}: {count:4}")

    # Show PnL distribution (sample)
    pnls = sorted([float(a.total_pnl) for a in accounts.values()], reverse=True)
    print(f"\nPnL Distribution:")
    print(f"  Top 1:     ${pnls[0]:>12,.0f}" if pnls else "  No accounts")
    print(f"  Top 10:    ${pnls[9]:>12,.0f}" if len(pnls) > 9 else "")
    print(f"  Top 100:   ${pnls[99]:>12,.0f}" if len(pnls) > 99 else "")
    print(f"  Median:    ${pnls[len(pnls)//2]:>12,.0f}" if pnls else "")
    print(f"  Lowest:    ${pnls[-1]:>12,.0f}" if pnls else "")

    # Save to simple file for verification
    output_file = PROJECT_ROOT / "data" / "profitable_accounts.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(f"# Accounts with >= ${MIN_PROFIT_USD:,.0f} profit\n")
        f.write(f"# Collected: {datetime.now().isoformat()}\n")
        f.write(f"# Total: {len(accounts)}\n\n")
        for wallet, entry in sorted(accounts.items(), key=lambda x: -float(x[1].total_pnl)):
            f.write(f"{wallet},{entry.total_pnl},{entry.num_trades}\n")

    print(f"\nSaved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
