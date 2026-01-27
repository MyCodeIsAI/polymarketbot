#!/usr/bin/env python3
"""Verify all the fixes work correctly."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from decimal import Decimal
from datetime import datetime
from src.api.data import DataAPIClient, ActivityType


# Import the category detection function
from analyze_profitable import detect_market_category, calculate_drawdown_metrics, fetch_all_activities


async def verify_all_fixes():
    """Run all verification tests."""

    print("=" * 70)
    print("VERIFICATION TEST SUITE")
    print("=" * 70)

    # ==========================================================================
    # TEST 1: Token ID Parsing
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: TOKEN_ID PARSING FIX")
    print("=" * 70)

    async with DataAPIClient() as client:
        wallet = "0x204f72f35326db932158cba6adff0b9a1da95e14"
        activities = await client.get_activity(
            user=wallet,
            activity_type=ActivityType.TRADE,
            limit=5,
        )

        print(f"\nFetched {len(activities)} activities")
        print("\nChecking token_id parsing:")
        for i, act in enumerate(activities[:3]):
            token_id = act.token_id
            has_token = bool(token_id)
            status = "PASS" if has_token else "FAIL"
            print(f"  Trade {i+1}: token_id = {token_id[:30] if token_id else 'EMPTY'}... [{status}]")

        all_have_tokens = all(bool(a.token_id) for a in activities)
        print(f"\n  Overall: {'PASS - All activities have token_id' if all_have_tokens else 'FAIL - Some activities missing token_id'}")

    # ==========================================================================
    # TEST 2: Market Category Detection
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: MARKET CATEGORY DETECTION")
    print("=" * 70)

    test_cases = [
        ("fr2-bas-mon-2026-01-23", "Will SC Bastia vs. Montpellier HSC end in a draw?", "sports"),
        ("fl1-aja-psg-2026-01-23", "Will Paris Saint-Germain FC win on 2026-01-23?", "sports"),
        ("nfl-week-17", "Will the Chiefs beat the Ravens?", "sports"),
        ("president-2024", "Who will win the 2024 Presidential Election?", "politics"),
        ("btc-price", "Will Bitcoin reach $100k by end of 2025?", "crypto"),
        ("fed-rates", "Will the Fed cut interest rates in December?", "finance"),
        ("", "", "other"),
    ]

    print("\nTesting category detection:")
    all_pass = True
    for event_slug, title, expected in test_cases:
        result = detect_market_category(event_slug, title)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        display_title = title[:40] + "..." if len(title) > 40 else title
        print(f"  [{status}] '{display_title}' -> {result} (expected: {expected})")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    # ==========================================================================
    # TEST 3: Find Account with SELL Trades and Verify Metrics
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: METRICS CALCULATION (FINDING ACCOUNT WITH SELL TRADES)")
    print("=" * 70)

    # Load accounts and find one with sells
    import json
    test_wallet = None
    test_sells_count = 0

    async with DataAPIClient() as client:
        try:
            with open(PROJECT_ROOT / "data" / "analyzed_accounts.json") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])

            print(f"\nSearching for account with SELL trades...")
            for acc in accounts[:20]:  # Check first 20
                wallet = acc.get("wallet")
                if not wallet:
                    continue

                trades = await client.get_activity(
                    user=wallet,
                    activity_type=ActivityType.TRADE,
                    limit=200,
                )

                sells = [t for t in trades if t.side and t.side.value == "SELL"]
                if len(sells) >= 10:  # Need at least 10 sells for meaningful test
                    test_wallet = wallet
                    test_sells_count = len(sells)
                    print(f"  FOUND: {wallet[:30]}... has {len(sells)} sells!")
                    break

                await asyncio.sleep(0.1)

            if not test_wallet:
                print("  WARNING: No account with sufficient sells found in first 20 accounts")
                print("  This may indicate most traders hold to resolution (don't sell)")
        except Exception as e:
            print(f"  Error loading accounts: {e}")

        if test_wallet:
            print(f"\nFetching all trades for selected account...")
            all_trades = await client.get_activity(
                user=test_wallet,
                activity_type=ActivityType.TRADE,
                limit=500,
            )

            buys = [t for t in all_trades if t.side and t.side.value == "BUY"]
            sells = [t for t in all_trades if t.side and t.side.value == "SELL"]

            print(f"  Total trades: {len(all_trades)}")
            print(f"  BUY trades: {len(buys)}")
            print(f"  SELL trades: {len(sells)}")

            if sells:
                print(f"\n  Sample SELL trade:")
                sell = sells[0]
                print(f"    token_id: {sell.token_id[:40] if sell.token_id else 'EMPTY'}...")
                print(f"    price: {sell.price}")
                print(f"    usd_value: {sell.usd_value}")

            # Calculate metrics
            print("\n  Calculating drawdown metrics...")
            metrics = calculate_drawdown_metrics(all_trades, 50000)  # Assume some total PnL

            print(f"\n  CALCULATED METRICS:")
            print(f"    Win count: {metrics.get('win_count', 0)}")
            print(f"    Loss count: {metrics.get('loss_count', 0)}")
            print(f"    Win rate: {metrics.get('actual_win_rate', 0) * 100:.1f}%")
            print(f"    Gross profit: ${metrics.get('gross_profit', 0):,.2f}")
            print(f"    Gross loss: ${metrics.get('gross_loss', 0):,.2f}")
            print(f"    Profit factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"    Max drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
            print(f"    Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"    P/L smoothness: {metrics.get('pl_curve_smoothness', 0):.3f}")

            # Verify non-zero metrics
            has_wins = metrics.get('win_count', 0) > 0
            has_losses = metrics.get('loss_count', 0) > 0
            has_profit = metrics.get('gross_profit', 0) > 0

            print(f"\n  Verification:")
            print(f"    Has wins: {'PASS' if has_wins else 'FAIL'}")
            print(f"    Has losses: {'PASS' if has_losses else 'FAIL'}")
            print(f"    Has profit calculated: {'PASS' if has_profit else 'FAIL'}")
        else:
            print("\n  SKIPPED: No suitable account found")

    # ==========================================================================
    # TEST 4: Category Detection on Real Data
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: CATEGORY DETECTION ON REAL ACTIVITY DATA")
    print("=" * 70)

    # Use first account from analyzed data for category test
    test_wallet_for_cat = "0x204f72f35326db932158cba6adff0b9a1da95e14"

    async with DataAPIClient() as client:
        activities = await client.get_activity(
            user=test_wallet_for_cat,
            activity_type=ActivityType.TRADE,
            limit=100,
        )

        categories = {}
        for act in activities:
            cat = detect_market_category(
                event_slug=getattr(act, 'event_slug', None),
                market_title=getattr(act, 'market_title', None)
            )
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n  Category distribution from {len(activities)} trades:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            pct = (count / len(activities)) * 100
            print(f"    {cat}: {count} trades ({pct:.1f}%)")

        print(f"\n  Unique categories: {len(categories)}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(verify_all_fixes())
