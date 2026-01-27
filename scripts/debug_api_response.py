#!/usr/bin/env python3
"""Debug API response to understand the actual data structure."""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType


async def debug_api():
    """Debug the raw API response."""
    # Test wallet
    wallet = "0x204f72f35326db932158cba6adff0b9a1da95e14"

    async with DataAPIClient() as client:
        # Make a raw request to see the actual response
        print("=" * 60)
        print("DEBUGGING RAW API RESPONSE")
        print("=" * 60)

        # 1. First, let's see the raw response structure
        params = {
            "user": wallet.lower(),
            "type": "TRADE",
            "limit": 5,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }

        response = await client.get("/activity", params=params)

        print("\n1. RAW API RESPONSE (first 5 trades):")
        print("-" * 40)

        if isinstance(response.data, list):
            raw_data = response.data
        else:
            raw_data = response.data.get("activity", response.data.get("data", []))

        for i, item in enumerate(raw_data[:3]):
            print(f"\n--- Trade {i+1} ---")
            print(json.dumps(item, indent=2, default=str))

        # 2. Check what keys are present
        if raw_data:
            print("\n2. AVAILABLE KEYS IN API RESPONSE:")
            print("-" * 40)
            all_keys = set()
            for item in raw_data:
                all_keys.update(item.keys())
            for key in sorted(all_keys):
                sample = raw_data[0].get(key, "N/A")
                print(f"  {key}: {type(sample).__name__} = {str(sample)[:50]}")

        # 3. Get parsed Activity objects
        print("\n3. PARSED ACTIVITY OBJECTS:")
        print("-" * 40)
        activities = await client.get_activity(
            user=wallet,
            activity_type=ActivityType.TRADE,
            limit=5,
        )

        for i, act in enumerate(activities[:3]):
            print(f"\n--- Activity {i+1} ---")
            print(f"  side: {act.side}")
            print(f"  token_id: {act.token_id}")
            print(f"  condition_id: {act.condition_id}")
            print(f"  price: {act.price}")
            print(f"  usd_value: {act.usd_value}")
            print(f"  outcome: {act.outcome}")

        # 4. Check buy vs sell distribution
        print("\n4. BUY/SELL DISTRIBUTION:")
        print("-" * 40)

        all_trades = await client.get_activity(
            user=wallet,
            activity_type=ActivityType.TRADE,
            limit=500,
        )

        buys = [t for t in all_trades if t.side and t.side.value == "BUY"]
        sells = [t for t in all_trades if t.side and t.side.value == "SELL"]

        print(f"  Total trades fetched: {len(all_trades)}")
        print(f"  BUY trades: {len(buys)}")
        print(f"  SELL trades: {len(sells)}")

        # 5. Find an account that has SELL trades
        print("\n5. LOOKING FOR ACCOUNT WITH SELL TRADES...")
        print("-" * 40)

        # Load analyzed accounts
        try:
            with open(PROJECT_ROOT / "data" / "analyzed_accounts.json") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])

            # Check a few accounts for sells
            for acc in accounts[:10]:
                test_wallet = acc.get("wallet")
                if not test_wallet:
                    continue

                test_trades = await client.get_activity(
                    user=test_wallet,
                    activity_type=ActivityType.TRADE,
                    limit=100,
                )

                test_sells = [t for t in test_trades if t.side and t.side.value == "SELL"]

                if test_sells:
                    print(f"\n  FOUND: {test_wallet[:20]}... has {len(test_sells)} sells!")

                    # Get more trades from this account
                    all_test = await client.get_activity(
                        user=test_wallet,
                        activity_type=ActivityType.TRADE,
                        limit=500,
                    )

                    all_buys = len([t for t in all_test if t.side and t.side.value == "BUY"])
                    all_sells = len([t for t in all_test if t.side and t.side.value == "SELL"])

                    print(f"  Total: {len(all_test)} trades, {all_buys} buys, {all_sells} sells")

                    # Show a sample sell trade
                    print(f"\n  Sample SELL trade:")
                    sell = test_sells[0]
                    print(f"    token_id: {sell.token_id}")
                    print(f"    condition_id: {sell.condition_id}")
                    print(f"    price: {sell.price}")
                    print(f"    usd_value: {sell.usd_value}")
                    break
                else:
                    print(f"  {test_wallet[:16]}... - no sells in first 100 trades")

                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    asyncio.run(debug_api())
