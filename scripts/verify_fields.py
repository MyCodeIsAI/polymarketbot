#!/usr/bin/env python3
"""
Verify that all fields needed for UI2/UI3 are being captured.
"""
import asyncio
import sys
from pathlib import Path
from decimal import Decimal

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig, LeaderboardClient
from src.discovery.models import DiscoveryMode

# Import the normalize function from mass_scan
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from mass_scan import normalize_result, convert_decimals


async def verify():
    print("=" * 70)
    print("  VERIFYING DATA RETENTION FOR UI2/UI3")
    print("=" * 70)

    # Get a test wallet
    print("\n[1] Fetching test wallet...")
    async with LeaderboardClient() as lb_client:
        entries = await lb_client.get_leaderboard(category='OVERALL', limit=1)
        test_wallet = entries[0].get('proxyWallet', entries[0].get('wallet', ''))

    print(f"  Testing: {test_wallet[:20]}...")

    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        min_trades=75,
        min_score_threshold=0.0,
    )

    async with DiscoveryService() as service:
        service._analyzer.set_mode(config.mode)
        service._analyzer.scoring_engine.update_config({
            "min_composite_score": 0.0,
            "hard_filters": {"min_trades": {"threshold": 75}}
        })

        # Get deep analysis
        result = await service._analyzer.deep_analysis(
            test_wallet,
            lookback_days=1825,
            max_trades=500,
        )

        # Convert to dict
        raw_dict = result.to_dict()
        raw_dict = convert_decimals(raw_dict)

        print("\n[2] Raw to_dict() fields:")
        print(f"  Keys: {list(raw_dict.keys())}")

        print("\n[3] pl_metrics fields:")
        pl = raw_dict.get("pl_metrics") or {}
        if pl:
            print(f"  {list(pl.keys())}")
        else:
            print("  ⚠ No pl_metrics!")

        print("\n[4] pattern_metrics fields:")
        pattern = raw_dict.get("pattern_metrics") or {}
        if pattern:
            print(f"  {list(pattern.keys())}")
        else:
            print("  ⚠ No pattern_metrics!")

        # Normalize
        normalized = normalize_result(raw_dict)

        print("\n[5] Normalized output fields:")
        print(f"  Total fields: {len(normalized)}")

        # Group by category
        categories = {
            "Identity": ["wallet_address", "wallet"],
            "P/L": ["total_pnl", "pnl_per_trade", "pnl_per_market", "gross_profit", "gross_loss"],
            "Volume": ["num_trades", "trades_fetched", "unique_markets", "trades_per_week"],
            "Position": ["avg_position", "median_position", "max_position", "position_consistency"],
            "Timing": ["account_age_days", "active_days", "activity_recency_days", "is_currently_active"],
            "Win/Loss": ["win_rate", "profitable_trades", "losing_trades", "avg_win_size", "avg_loss_size"],
            "Risk": ["profit_factor", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "avg_drawdown_pct"],
            "Odds": ["pct_under_5c", "pct_under_10c", "pct_under_20c", "avg_entry_odds"],
            "Category": ["category_breakdown", "primary_category", "niche_market_pct", "category_win_rates"],
            "Scoring": ["systematic_score", "score_breakdown", "red_flags"],
            "Display": ["recent_trades_sample", "pl_curve_data", "market_categories"],
        }

        print("\n[6] Field verification by category:")
        all_good = True
        for cat_name, fields in categories.items():
            missing = [f for f in fields if f not in normalized]
            empty = [f for f in fields if f in normalized and not normalized[f] and normalized[f] != 0]

            if missing:
                print(f"  ⚠ {cat_name}: MISSING {missing}")
                all_good = False
            elif empty:
                print(f"  ~ {cat_name}: Empty: {empty}")
            else:
                print(f"  ✓ {cat_name}: All present")

        print("\n[7] Sample values:")
        key_fields = [
            "total_pnl", "num_trades", "win_rate", "profit_factor",
            "sharpe_ratio", "systematic_score", "activity_recency_days",
            "primary_category", "avg_position"
        ]
        for field in key_fields:
            value = normalized.get(field)
            if isinstance(value, float):
                print(f"  {field}: {value:.2f}")
            else:
                print(f"  {field}: {value}")

        print("\n[8] pl_curve_data sample:")
        pl_curve = normalized.get("pl_curve_data", [])
        print(f"  Points: {len(pl_curve)}")
        if pl_curve:
            print(f"  First: {pl_curve[0]}")

        print("\n[9] recent_trades_sample:")
        trades = normalized.get("recent_trades_sample", [])
        print(f"  Trades: {len(trades)}")
        if trades:
            print(f"  First: {list(trades[0].keys()) if isinstance(trades[0], dict) else trades[0]}")

        print("\n" + "-" * 70)
        if all_good:
            print("  ✓ ALL REQUIRED FIELDS PRESENT")
            print("  ✓ Data retention looks good for UI2/UI3")
        else:
            print("  ⚠ Some fields are missing - check above")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(verify())
