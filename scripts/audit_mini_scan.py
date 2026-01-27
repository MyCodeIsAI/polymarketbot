#!/usr/bin/env python3
"""
Run mini mass scan and audit individual account results to verify all fields.
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.service import DiscoveryService, ScanConfig
from src.discovery.models import DiscoveryMode

# Import normalize function
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from mass_scan import normalize_result, convert_decimals

# Same config as mass_scan.py
MIN_PNL = 20000
MAX_PNL = 100000
MIN_TRADES = 75
MAX_TRADES = 3000
MAX_CANDIDATES = 50  # Small batch
TRADES_TO_SCRAPE = 500


async def run_audit():
    print("=" * 70)
    print("  MINI SCAN + FULL AUDIT")
    print("=" * 70)

    config = ScanConfig(
        mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        max_candidates=MAX_CANDIDATES,
        max_profit=MAX_PNL,
        analyze_top_n=TRADES_TO_SCRAPE,
        max_trades=MAX_TRADES,
        min_trades=MIN_TRADES,
        min_score_threshold=0.0,
        max_phase2=MAX_CANDIDATES,
        max_phase3=MAX_CANDIDATES,
        persist_to_db=False,
    )

    start_time = datetime.now()
    results = []

    print(f"\n[1] Running mini scan ({MAX_CANDIDATES} candidates)...")

    try:
        async with DiscoveryService() as service:
            results, scan = await service.run_scan(config, progress_callback=lambda p: None)

            if results and not isinstance(results[0], dict):
                results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Raw results: {len(results)} accounts")

    if not results:
        print("\n  ⚠ No results to audit!")
        return

    # Normalize all results
    print("\n[2] Normalizing results...")
    normalized = []
    for r in results:
        try:
            norm = normalize_result(convert_decimals(r))
            normalized.append(norm)
        except Exception as e:
            print(f"  ⚠ Normalization error: {e}")

    print(f"  Normalized: {len(normalized)} accounts")

    # Audit 3 accounts in detail
    print("\n" + "=" * 70)
    print("  DETAILED AUDIT OF SAMPLE ACCOUNTS")
    print("=" * 70)

    # Required fields for UI2 filtering
    FILTER_FIELDS = [
        "total_pnl", "num_trades", "win_rate", "profit_factor", "sharpe_ratio",
        "max_drawdown_pct", "avg_position", "activity_recency_days", "account_age_days",
        "unique_markets", "systematic_score"
    ]

    # Required fields for UI3 display
    DISPLAY_FIELDS = [
        "wallet_address", "primary_category", "category_breakdown", "category_win_rates",
        "pl_curve_data", "recent_trades_sample", "score_breakdown", "red_flags"
    ]

    # Derived/computed fields
    COMPUTED_FIELDS = [
        "pnl_per_trade", "pnl_per_market", "trades_per_week", "position_consistency",
        "is_currently_active"
    ]

    # All fields we expect
    ALL_REQUIRED = set(FILTER_FIELDS + DISPLAY_FIELDS + COMPUTED_FIELDS)

    audit_accounts = normalized[:3]  # Audit first 3

    for i, account in enumerate(audit_accounts):
        wallet = account.get("wallet_address", "")[:20]
        print(f"\n{'─' * 70}")
        print(f"  ACCOUNT {i+1}: {wallet}...")
        print(f"{'─' * 70}")

        # Check for missing fields
        missing = [f for f in ALL_REQUIRED if f not in account]
        if missing:
            print(f"\n  ⚠ MISSING FIELDS: {missing}")

        # Check for None/empty values
        empty_critical = []
        for field in FILTER_FIELDS:
            val = account.get(field)
            if val is None:
                empty_critical.append(f"{field}=None")
            elif val == "" or val == []:
                empty_critical.append(f"{field}=empty")

        if empty_critical:
            print(f"\n  ⚠ EMPTY CRITICAL FIELDS: {empty_critical}")

        # Print all filter fields
        print(f"\n  [Filter Fields - Used for UI2 filtering]")
        for field in FILTER_FIELDS:
            val = account.get(field)
            if isinstance(val, float):
                print(f"    {field}: {val:.4f}")
            else:
                print(f"    {field}: {val}")

        # Print key display fields
        print(f"\n  [Display Fields - Used for UI3]")
        print(f"    wallet_address: {account.get('wallet_address', '')[:30]}...")
        print(f"    primary_category: {account.get('primary_category')}")
        print(f"    category_breakdown: {len(account.get('category_breakdown', {}))} categories")
        print(f"    category_win_rates: {len(account.get('category_win_rates', {}))} categories")
        print(f"    pl_curve_data: {len(account.get('pl_curve_data', []))} points")
        print(f"    recent_trades_sample: {len(account.get('recent_trades_sample', []))} trades")
        print(f"    score_breakdown: {type(account.get('score_breakdown')).__name__}")
        print(f"    red_flags: {len(account.get('red_flags', []))} flags")

        # Print computed fields
        print(f"\n  [Computed Fields]")
        for field in COMPUTED_FIELDS:
            val = account.get(field)
            if isinstance(val, float):
                print(f"    {field}: {val:.4f}")
            else:
                print(f"    {field}: {val}")

        # Print additional useful fields
        print(f"\n  [Additional Fields]")
        print(f"    sortino_ratio: {account.get('sortino_ratio', 0):.4f}")
        print(f"    avg_drawdown_pct: {account.get('avg_drawdown_pct', 0):.4f}")
        print(f"    profitable_trades: {account.get('profitable_trades', 0)}")
        print(f"    losing_trades: {account.get('losing_trades', 0)}")
        print(f"    buy_sell_ratio: {account.get('buy_sell_ratio', 0):.4f}")
        print(f"    niche_market_pct: {account.get('niche_market_pct', 0):.4f}")

        # Validate pl_curve_data structure
        pl_curve = account.get("pl_curve_data", [])
        if pl_curve:
            print(f"\n  [pl_curve_data sample]")
            sample = pl_curve[0] if pl_curve else {}
            print(f"    First point: {sample}")
            if len(pl_curve) > 1:
                print(f"    Last point: {pl_curve[-1]}")

        # Validate recent_trades_sample structure
        trades = account.get("recent_trades_sample", [])
        if trades:
            print(f"\n  [recent_trades_sample structure]")
            sample = trades[0] if trades else {}
            if isinstance(sample, dict):
                print(f"    Fields: {list(sample.keys())}")
            else:
                print(f"    Format: {type(sample)}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Check all accounts for issues
    issues = []
    for j, acc in enumerate(normalized):
        missing = [f for f in FILTER_FIELDS if acc.get(f) is None]
        if missing:
            issues.append(f"Account {j}: missing {missing}")

    if issues:
        print(f"\n  ⚠ ISSUES FOUND:")
        for issue in issues[:10]:
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print(f"\n  ✓ All {len(normalized)} accounts have required filter fields")

    # Stats
    print(f"\n  Statistics:")
    pnls = [a.get("total_pnl", 0) for a in normalized]
    scores = [a.get("systematic_score", 0) for a in normalized]
    trades = [a.get("num_trades", 0) for a in normalized]
    recency = [a.get("activity_recency_days", 999) for a in normalized]

    print(f"    P/L range: ${min(pnls):,.0f} - ${max(pnls):,.0f}")
    print(f"    Score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"    Trade count range: {min(trades)} - {max(trades)}")
    print(f"    Activity recency: {min(recency)} - {max(recency)} days")

    # Count with data
    has_pl_curve = sum(1 for a in normalized if a.get("pl_curve_data"))
    has_trades_sample = sum(1 for a in normalized if a.get("recent_trades_sample"))
    has_categories = sum(1 for a in normalized if a.get("category_breakdown"))

    print(f"\n  Data presence:")
    print(f"    With pl_curve_data: {has_pl_curve}/{len(normalized)}")
    print(f"    With recent_trades_sample: {has_trades_sample}/{len(normalized)}")
    print(f"    With category_breakdown: {has_categories}/{len(normalized)}")

    print("\n" + "=" * 70)
    if not issues and has_pl_curve == len(normalized):
        print("  ✓ AUDIT PASSED - All fields properly populated")
    else:
        print("  ⚠ AUDIT HAS WARNINGS - Review above")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_audit())
