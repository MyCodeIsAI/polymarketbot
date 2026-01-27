#!/usr/bin/env python3
"""Audit what data redeems have vs trades to understand proper metric handling."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType


async def audit_data():
    """Compare TRADE vs REDEEM data fields."""

    print("=" * 70)
    print("AUDIT: TRADE vs REDEEM DATA FIELDS")
    print("=" * 70)

    wallet = "0x204f72f35326db932158cba6adff0b9a1da95e14"

    async with DataAPIClient() as client:
        # Get sample trade and redeem
        trades = await client.get_activity(
            user=wallet,
            activity_type=ActivityType.TRADE,
            limit=5,
        )

        redeems = await client.get_activity(
            user=wallet,
            activity_type=ActivityType.REDEEM,
            limit=5,
        )

        print("\n1. SAMPLE TRADE ACTIVITY:")
        print("-" * 50)
        if trades:
            t = trades[0]
            print(f"  type: {t.type}")
            print(f"  timestamp: {t.timestamp}")
            print(f"  condition_id: {t.condition_id[:30]}...")
            print(f"  token_id: {t.token_id[:30] if t.token_id else 'EMPTY'}...")
            print(f"  side: {t.side}")
            print(f"  size: {t.size}")
            print(f"  price: {t.price}")
            print(f"  usd_value: {t.usd_value}")
            print(f"  outcome: {t.outcome}")
            print(f"  event_slug: {getattr(t, 'event_slug', 'N/A')}")
            print(f"  market_title: {getattr(t, 'market_title', 'N/A')}")

        print("\n2. SAMPLE REDEEM ACTIVITY:")
        print("-" * 50)
        if redeems:
            r = redeems[0]
            print(f"  type: {r.type}")
            print(f"  timestamp: {r.timestamp}")
            print(f"  condition_id: {r.condition_id[:30]}...")
            print(f"  token_id: {r.token_id[:30] if r.token_id else 'EMPTY'}...")
            print(f"  side: {r.side}")
            print(f"  size: {r.size}")
            print(f"  price: {r.price}")
            print(f"  usd_value: {r.usd_value}")
            print(f"  outcome: {r.outcome}")
            print(f"  event_slug: {getattr(r, 'event_slug', 'N/A')}")
            print(f"  market_title: {getattr(r, 'market_title', 'N/A')}")

        print("\n3. METRIC AUDIT - WHICH METRICS SHOULD INCLUDE REDEEMS?")
        print("-" * 50)

        metrics_audit = [
            ("num_trades", "NO", "Redeems are not trades - they're position closings"),
            ("position_sizes", "NO", "Redeem size=0 or payout amount, not position size"),
            ("avg_position", "NO", "Same as position_sizes"),
            ("activity_dates", "MAYBE", "Redeems show activity but not trading decisions"),
            ("account_age", "YES", "Earliest trade/redeem gives account age"),
            ("activity_recency", "YES", "Recent redeem = still active"),
            ("unique_markets", "NO", "Redeem market data is incomplete"),
            ("buy_sell_ratio", "NO", "Redeems are neither buys nor sells"),
            ("trades_per_week", "NO", "Should only count actual trades"),
            ("trades_last_7d", "MAYBE", "Could include redeems as 'activity'"),
            ("is_currently_active", "YES", "Recent redeem = still active"),
            ("P/L calculation", "YES", "Core purpose - already implemented"),
            ("win_rate", "YES", "From P/L calculation (redeems = wins)"),
            ("drawdown", "YES", "Part of P/L curve - already implemented"),
            ("sharpe_ratio", "YES", "Part of P/L calculation - already implemented"),
        ]

        for metric, should_include, reason in metrics_audit:
            print(f"\n  {metric}:")
            print(f"    Include redeems? {should_include}")
            print(f"    Reason: {reason}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(audit_data())
