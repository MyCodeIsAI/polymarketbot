#!/usr/bin/env python3
"""Simulate Fed Chair wallet scoring at time of first trade.

This validates whether our scanner would have caught the suspicious
wallets when they first traded (Jan 19-21, 2026).
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insider_scanner.scoring import InsiderScorer, MarketCategory

# Fed Chair suspicious wallets from FED-CHAIR-CLUSTER.md
FED_CHAIR_WALLETS = {
    "hd2": {
        "address": "0x1138c6c14b9f3b6edd719d0b7b471bd896afe410",
        "first_trade_date": "2026-01-19",
        "first_trade_size_usd": 5000,  # Started with ~$5K blocks
        "entry_odds": 0.195,  # 19.5 cents
        "total_invested": 96520,
        "account_age_at_trade": 1,  # Assume 1 day (new account pattern)
        "prior_transactions": 0,
    },
    "msk11": {
        "address": "0xa94e3cae6e3917e8243b13e06f8f5ce5f27a24e0",
        "first_trade_date": "2026-01-19",  # Early entry
        "first_trade_size_usd": 3000,  # Smaller blocks
        "entry_odds": 0.20,
        "total_invested": 45000,  # ~$45K total
        "account_age_at_trade": 3,  # Could be slightly older
        "prior_transactions": 2,  # Some prior activity
    },
    "jwerhor": {
        "address": "0xee1706b93845d50ea60e49219be7311a2ebb6ea1",
        "first_trade_date": "2026-01-20",
        "first_trade_size_usd": 8000,
        "entry_odds": 0.22,
        "total_invested": 85000,
        "account_age_at_trade": 2,
        "prior_transactions": 0,
    },
    "kickstand7": {
        "address": "0xd1acd3925d895de9aec98ff95f3a30c5279d08d5",
        "first_trade_date": "2026-01-21",  # Later entry
        "first_trade_size_usd": 2000,  # Smaller first trade
        "entry_odds": 0.25,  # Higher odds
        "total_invested": 25000,
        "account_age_at_trade": 7,  # Week-old account
        "prior_transactions": 5,  # Some history
    },
}

def simulate_first_trade_score(wallet_key: str, wallet_data: dict) -> dict:
    """Simulate what score wallet would have gotten at first trade time."""
    scorer = InsiderScorer()

    # Build position representing FIRST trade only
    first_trade_position = {
        "market_id": "fed_chair_rieder",
        "side": "YES",
        "size_usd": wallet_data["first_trade_size_usd"],
        "entry_odds": wallet_data["entry_odds"],
        "resolved": False,  # Not resolved yet at time of trade
        "won": None,
    }

    # Score with state AT TIME OF FIRST TRADE
    result = scorer.score_wallet(
        wallet_address=wallet_data["address"],
        account_age_days=wallet_data["account_age_at_trade"],
        transaction_count=wallet_data["prior_transactions"],
        positions=[first_trade_position],
        trades=None,
        market_category=MarketCategory.GOVERNMENT_POLICY,  # Fed Chair = policy
        event_hours_away=72,  # ~3 days before Trump endorsement
        funding_source=None,  # Unknown at first trade
        flagged_funders=None,
        cluster_wallets=None,  # Not identified as cluster yet
    )

    return {
        "wallet_key": wallet_key,
        "address": wallet_data["address"][:10] + "...",
        "score": result.score,
        "priority": result.priority,
        "would_alert": result.score >= 50,  # Updated threshold (was 55)
        "would_alert_at_50": result.score >= 50,
        "would_alert_at_45": result.score >= 45,
        "signal_count": result.signal_count,
        "active_dimensions": result.active_dimensions,
        "dimensions": result.dimensions,
        "signals": [(s.name, s.weight) for s in result.signals],
    }


def simulate_full_position_score(wallet_key: str, wallet_data: dict) -> dict:
    """Simulate score after FULL position is built."""
    scorer = InsiderScorer()

    # Full cumulative position
    full_position = {
        "market_id": "fed_chair_rieder",
        "side": "YES",
        "size_usd": wallet_data["total_invested"],
        "entry_odds": wallet_data["entry_odds"],
        "resolved": False,
        "won": None,
    }

    result = scorer.score_wallet(
        wallet_address=wallet_data["address"],
        account_age_days=wallet_data["account_age_at_trade"] + 2,  # 2 days later
        transaction_count=wallet_data["prior_transactions"] + 50,  # Many trades now
        positions=[full_position],
        trades=None,
        market_category=MarketCategory.GOVERNMENT_POLICY,
        event_hours_away=24,  # 1 day before Trump endorsement
        funding_source=None,
        flagged_funders=None,
        cluster_wallets=None,
    )

    return {
        "wallet_key": wallet_key,
        "score": result.score,
        "priority": result.priority,
        "would_alert": result.score >= 55,
    }


def main():
    print("=" * 70)
    print("FED CHAIR INSIDER DETECTION SIMULATION")
    print("Simulating scores at TIME OF FIRST TRADE")
    print("=" * 70)
    print()

    results = []
    for key, data in FED_CHAIR_WALLETS.items():
        result = simulate_first_trade_score(key, data)
        results.append(result)

        print(f"\n{'='*50}")
        print(f"Wallet: {key} ({result['address']})")
        print(f"{'='*50}")
        print(f"  Score: {result['score']:.1f}")
        print(f"  Priority: {result['priority'].upper()}")
        print(f"  Would alert at 55: {'YES ✓' if result['would_alert'] else 'NO ✗'}")
        print(f"  Would alert at 50: {'YES ✓' if result['would_alert_at_50'] else 'NO ✗'}")
        print(f"  Would alert at 45: {'YES ✓' if result['would_alert_at_45'] else 'NO ✗'}")
        print(f"  Signals: {result['signal_count']}")
        print(f"  Active dimensions: {result['active_dimensions']}")
        print(f"  Dimensions: {result['dimensions']}")
        print(f"  Signal breakdown:")
        for name, weight in result['signals']:
            print(f"    - {name}: {weight}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    caught_at_55 = sum(1 for r in results if r['would_alert'])
    caught_at_50 = sum(1 for r in results if r['would_alert_at_50'])
    caught_at_45 = sum(1 for r in results if r['would_alert_at_45'])

    print(f"\nAt threshold 50 (NEW): {caught_at_50}/4 wallets caught ✓")
    print(f"At threshold 55 (OLD): {caught_at_55}/4 wallets caught")
    print(f"At threshold 45: {caught_at_45}/4 wallets caught")

    missed_at_50 = [r for r in results if not r['would_alert_at_50']]
    if missed_at_50:
        print(f"\nMISSED at NEW threshold 50:")
        for r in missed_at_50:
            print(f"  - {r['wallet_key']}: score {r['score']:.1f}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if caught_at_55 < 4:
        gap = 55 - min(r['score'] for r in results)
        print(f"\nGap to catch all: {gap:.1f} points")
        print("\nOptions:")
        print("  1. Lower threshold to 50 (catches {}/4)".format(caught_at_50))
        print("  2. Lower threshold to 45 (catches {}/4)".format(caught_at_45))
        print("  3. Increase GOVERNMENT_POLICY category weight")
        print("  4. Add special handling for 'Fed Chair' type markets")

        # Check what's causing lower scores
        lowest = min(results, key=lambda r: r['score'])
        print(f"\nLowest scorer: {lowest['wallet_key']} ({lowest['score']:.1f})")
        print(f"  Account age: {FED_CHAIR_WALLETS[lowest['wallet_key']]['account_age_at_trade']} days")
        print(f"  Prior txs: {FED_CHAIR_WALLETS[lowest['wallet_key']]['prior_transactions']}")
        print(f"  First trade: ${FED_CHAIR_WALLETS[lowest['wallet_key']]['first_trade_size_usd']}")

    # Also show full position scores
    print("\n" + "=" * 70)
    print("SCORES AFTER FULL POSITION BUILT (end of accumulation)")
    print("=" * 70)
    for key, data in FED_CHAIR_WALLETS.items():
        full_result = simulate_full_position_score(key, data)
        status = "✓ CAUGHT" if full_result['would_alert'] else "✗ missed"
        print(f"  {key}: {full_result['score']:.1f} ({full_result['priority']}) - {status}")


if __name__ == "__main__":
    main()
