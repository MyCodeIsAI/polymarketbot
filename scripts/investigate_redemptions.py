#!/usr/bin/env python3
"""Investigate REDEEM activities to understand resolution P/L tracking."""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType


async def investigate_redemptions():
    """Investigate how REDEEM activities work."""

    print("=" * 70)
    print("INVESTIGATING REDEEM (RESOLUTION) ACTIVITIES")
    print("=" * 70)

    # Load a test wallet
    wallet = "0x204f72f35326db932158cba6adff0b9a1da95e14"

    async with DataAPIClient() as client:
        # 1. Check what activity types exist for this account
        print("\n1. CHECKING ALL ACTIVITY TYPES FOR ACCOUNT")
        print("-" * 50)

        activity_types = {}

        # Fetch activities without type filter to see all types
        for activity_type in [ActivityType.TRADE, ActivityType.REDEEM, ActivityType.SPLIT, ActivityType.MERGE]:
            activities = await client.get_activity(
                user=wallet,
                activity_type=activity_type,
                limit=100,
            )
            activity_types[activity_type.value] = len(activities)
            print(f"  {activity_type.value}: {len(activities)} activities")

        # 2. Look at REDEEM activities in detail
        print("\n2. REDEEM ACTIVITY DETAILS")
        print("-" * 50)

        redeem_activities = await client.get_activity(
            user=wallet,
            activity_type=ActivityType.REDEEM,
            limit=20,
        )

        if redeem_activities:
            print(f"\nFound {len(redeem_activities)} REDEEM activities")

            # Get raw data to see all fields
            params = {
                "user": wallet.lower(),
                "type": "REDEEM",
                "limit": 5,
            }
            response = await client.get("/activity", params=params)
            raw_data = response.data if isinstance(response.data, list) else response.data.get("activity", [])

            print("\nRaw REDEEM activity data (first 3):")
            for i, item in enumerate(raw_data[:3]):
                print(f"\n--- REDEEM {i+1} ---")
                print(json.dumps(item, indent=2, default=str))

            # Parse and show key fields
            print("\n\nParsed REDEEM activities:")
            for i, act in enumerate(redeem_activities[:5]):
                print(f"\n  REDEEM {i+1}:")
                print(f"    condition_id: {act.condition_id[:30]}...")
                print(f"    token_id: {act.token_id[:30] if act.token_id else 'N/A'}...")
                print(f"    size: {act.size}")
                print(f"    price: {act.price}")
                print(f"    usd_value: {act.usd_value}")
                print(f"    outcome: {act.outcome}")
                print(f"    timestamp: {act.timestamp}")
        else:
            print("  No REDEEM activities found for this account")

        # 3. Find an account with both TRADES and REDEEMS
        print("\n3. FINDING ACCOUNT WITH TRADES AND REDEEMS")
        print("-" * 50)

        try:
            with open(PROJECT_ROOT / "data" / "analyzed_accounts.json") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])

            for acc in accounts[:15]:
                test_wallet = acc.get("wallet")
                if not test_wallet:
                    continue

                trades = await client.get_activity(
                    user=test_wallet,
                    activity_type=ActivityType.TRADE,
                    limit=50,
                )

                redeems = await client.get_activity(
                    user=test_wallet,
                    activity_type=ActivityType.REDEEM,
                    limit=50,
                )

                if len(trades) > 0 and len(redeems) > 0:
                    print(f"\n  FOUND: {test_wallet[:30]}...")
                    print(f"    TRADES: {len(trades)}")
                    print(f"    REDEEMS: {len(redeems)}")

                    # Check if any trades and redeems share the same condition_id
                    trade_conditions = set(t.condition_id for t in trades if t.condition_id)
                    redeem_conditions = set(r.condition_id for r in redeems if r.condition_id)
                    overlap = trade_conditions & redeem_conditions

                    print(f"    Shared condition_ids: {len(overlap)}")

                    if overlap:
                        # Show a complete position lifecycle
                        sample_condition = list(overlap)[0]
                        print(f"\n    POSITION LIFECYCLE for {sample_condition[:30]}...")

                        relevant_trades = [t for t in trades if t.condition_id == sample_condition]
                        relevant_redeems = [r for r in redeems if r.condition_id == sample_condition]

                        print(f"    BUYs:")
                        for t in relevant_trades[:3]:
                            side = t.side.value if t.side else "?"
                            print(f"      {side}: ${t.usd_value} @ {t.price} ({t.outcome})")

                        print(f"    REDEEMs:")
                        for r in relevant_redeems[:3]:
                            print(f"      REDEEM: ${r.usd_value} - {r.size} shares ({r.outcome})")

                    break

                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        # 4. Calculate P/L including redemptions
        print("\n4. P/L CALCULATION WITH REDEMPTIONS")
        print("-" * 50)

        # Explanation of how it should work
        print("""
  How Resolution P/L works on Polymarket:

  1. BUY: User buys "Yes" tokens at $0.60/share
     - Cost basis = $0.60 * shares

  2. RESOLUTION:
     a) If market resolves YES:
        - REDEEM activity: User claims $1.00/share
        - P/L = $1.00 - $0.60 = +$0.40/share (WIN)

     b) If market resolves NO:
        - NO REDEEM activity (tokens worthless)
        - P/L = $0.00 - $0.60 = -$0.60/share (LOSS)

  3. SELL (optional): User sells before resolution
     - SELL activity: User sells at market price
     - P/L = sell_price - buy_price

  To properly track P/L:
  - Track all BUY positions by (token_id + condition_id)
  - On SELL: Calculate realized P/L, reduce position
  - On REDEEM: Calculate resolved P/L ($1.00 - avg_cost), close position
  - For LOSSES: Need to check if market resolved and position has no REDEEM
        """)

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(investigate_redemptions())
