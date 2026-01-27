#!/usr/bin/env python3
"""
Verify that all calculated metrics are correct by comparing against raw trade data.

This script:
1. Fetches raw trade data for a sample account
2. Manually calculates key metrics
3. Compares with generated metrics to verify accuracy
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from collections import defaultdict
import json
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType


async def fetch_all_trades(client: DataAPIClient, wallet: str, max_pages: int = 20) -> tuple[list, list]:
    """Fetch all trades AND redeems for a wallet."""
    all_trades = []
    all_redeems = []
    offset = 0
    limit = 500

    # Fetch TRADE activities
    for page in range(max_pages):
        try:
            activities = await client.get_activity(
                user=wallet,
                activity_type=ActivityType.TRADE,
                limit=limit,
                offset=offset,
            )
            if not activities:
                break
            all_trades.extend(activities)
            if len(activities) < limit:
                break
            offset += limit
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"  Error fetching trades page {page}: {e}")
            break

    # Fetch REDEEM activities
    offset = 0
    for page in range(max_pages // 2):
        try:
            activities = await client.get_activity(
                user=wallet,
                activity_type=ActivityType.REDEEM,
                limit=limit,
                offset=offset,
            )
            if not activities:
                break
            all_redeems.extend(activities)
            if len(activities) < limit:
                break
            offset += limit
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"  Error fetching redeems page {page}: {e}")
            break

    return all_trades, all_redeems


def calculate_metrics_manually(trades: list, redeems: list = None) -> dict:
    """Calculate all metrics from raw trade AND redeem data manually."""
    redeems = redeems or []

    if not trades:
        return {"error": "No trades"}

    # Combine and sort all activities chronologically
    all_activities = list(trades) + list(redeems)
    sorted_activities = sorted(all_activities, key=lambda t: t.timestamp)

    # ====== Position tracking for P/L calculation ======
    # Track by condition_id (REDEEM doesn't have token_id)
    positions = {}  # condition_id -> {"size": Decimal, "cost": Decimal}
    cumulative_pnl = []
    running_pnl = Decimal("0")
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    win_count = 0
    loss_count = 0
    trade_pnls = []  # Individual trade P/L values

    for activity in sorted_activities:
        activity_type = getattr(activity.type, 'value', str(activity.type)) if hasattr(activity, 'type') else "TRADE"
        condition_id = activity.condition_id or "unknown"

        if activity_type == "REDEEM":
            # REDEEM: Market resolved - user claiming winnings
            redeem_amount = Decimal(str(activity.usd_value)) if activity.usd_value else Decimal("0")

            if condition_id in positions and redeem_amount > 0:
                pos = positions[condition_id]
                cost_basis = pos["cost"]

                # P/L from resolution = redeem amount - cost basis
                realized_pnl = redeem_amount - cost_basis
                running_pnl += realized_pnl
                trade_pnls.append(float(realized_pnl))

                if realized_pnl > 0:
                    gross_profit += realized_pnl
                    win_count += 1
                elif realized_pnl < 0:
                    gross_loss += abs(realized_pnl)
                    loss_count += 1

                del positions[condition_id]

            elif redeem_amount > 0:
                # Redeem without tracked position - estimate
                estimated_cost = redeem_amount * Decimal("0.5")
                estimated_profit = redeem_amount - estimated_cost
                running_pnl += estimated_profit
                trade_pnls.append(float(estimated_profit))
                gross_profit += estimated_profit
                win_count += 1

        else:
            # TRADE: Buy or Sell
            trade_size = Decimal(str(activity.usd_value)) if activity.usd_value else Decimal("0")
            if trade_size <= 0:
                continue

            side = getattr(activity.side, 'value', str(activity.side)) if activity.side else "BUY"
            price = Decimal(str(activity.price)) if activity.price else Decimal("0.5")

            if price > 0:
                shares = trade_size / price
            else:
                shares = trade_size

            if side.upper() == "BUY":
                if condition_id not in positions:
                    positions[condition_id] = {"size": Decimal("0"), "cost": Decimal("0")}
                positions[condition_id]["size"] += shares
                positions[condition_id]["cost"] += trade_size
            else:  # SELL
                if condition_id in positions:
                    pos = positions[condition_id]
                    if pos["size"] > 0:
                        avg_cost = pos["cost"] / pos["size"]
                        shares_to_sell = min(shares, pos["size"])
                        cost_basis = avg_cost * shares_to_sell
                        sale_proceeds = shares_to_sell * price
                        realized_pnl = sale_proceeds - cost_basis

                        running_pnl += realized_pnl
                        trade_pnls.append(float(realized_pnl))

                        if realized_pnl > 0:
                            gross_profit += realized_pnl
                            win_count += 1
                        elif realized_pnl < 0:
                            gross_loss += abs(realized_pnl)
                            loss_count += 1

                        pos["size"] -= shares_to_sell
                        pos["cost"] -= cost_basis
                        if pos["size"] <= 0:
                            del positions[condition_id]

        cumulative_pnl.append(float(running_pnl))

    # ====== Calculate drawdown from P/L curve ======
    peak = 0.0
    max_drawdown_pct = 0.0
    max_drawdown_usd = 0.0
    drawdowns = []
    current_drawdown_start = None
    min_since_peak = 0.0

    for i, pnl_point in enumerate(cumulative_pnl):
        if pnl_point > peak:
            if current_drawdown_start is not None and peak > 0:
                dd_usd = peak - min_since_peak
                dd_pct = (dd_usd / peak) * 100
                if dd_pct > 1:
                    drawdowns.append({"pct": dd_pct, "usd": dd_usd})
                    if dd_pct > max_drawdown_pct:
                        max_drawdown_pct = dd_pct
                        max_drawdown_usd = dd_usd
                current_drawdown_start = None
            peak = pnl_point
            min_since_peak = pnl_point
        elif pnl_point < peak:
            if current_drawdown_start is None:
                current_drawdown_start = i
                min_since_peak = pnl_point
            else:
                min_since_peak = min(min_since_peak, pnl_point)

    # Check if currently in drawdown
    current_dd_pct = 0.0
    if current_drawdown_start is not None and peak > 0:
        current_dd_usd = peak - min_since_peak
        current_dd_pct = (current_dd_usd / peak) * 100

    # ====== Basic metrics ======
    # Use original trades list (not including redeems) for basic metrics
    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    total_trades = len(sorted_trades)
    unique_markets = len(set(t.condition_id for t in sorted_trades if t.condition_id))

    # Position sizes (from trades only)
    sizes = [float(t.usd_value) for t in sorted_trades if t.usd_value and float(t.usd_value) > 0]
    avg_position = sum(sizes) / len(sizes) if sizes else 0
    max_position = max(sizes) if sizes else 0

    # Account age (include redeems for most recent activity)
    all_timestamps = [a.timestamp for a in sorted_activities]
    if len(all_timestamps) >= 2:
        first_activity = min(all_timestamps)
        last_activity = max(all_timestamps)
        account_age_days = max(1, (last_activity - first_activity).days)
    else:
        account_age_days = 1

    trades_per_week = (total_trades / max(1, account_age_days)) * 7

    # Win rate and profit factor
    total_closed_trades = win_count + loss_count
    win_rate = win_count / total_closed_trades if total_closed_trades > 0 else 0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)

    # Average win/loss
    avg_win = float(gross_profit / win_count) if win_count > 0 else 0
    avg_loss = float(gross_loss / loss_count) if loss_count > 0 else 0

    # P/L curve smoothness
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
    if len(cumulative_pnl) > 10 and final_pnl > 0:
        expected = [i * (final_pnl / len(cumulative_pnl)) for i in range(len(cumulative_pnl))]
        deviations = [abs(a - e) for a, e in zip(cumulative_pnl, expected)]
        avg_dev = sum(deviations) / len(deviations)
        smoothness = max(0, min(1, 1 - (avg_dev / max(1, final_pnl))))
    else:
        smoothness = 0.5

    # Sharpe ratio (simplified)
    if len(trade_pnls) > 1:
        import statistics
        mean_pnl = statistics.mean(trade_pnls)
        std_pnl = statistics.stdev(trade_pnls)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": total_trades,
        "unique_markets": unique_markets,
        "realized_pnl": float(running_pnl),
        "avg_position_size": round(avg_position, 2),
        "max_position_size": round(max_position, 2),
        "account_age_days": account_age_days,
        "trades_per_week": round(trades_per_week, 2),
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate * 100, 2),  # as percentage
        "profit_factor": round(min(profit_factor, 10), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "current_drawdown_pct": round(current_dd_pct, 2),
        "pl_curve_smoothness": round(smoothness, 3),
        "sharpe_ratio": round(sharpe, 3),
        "gross_profit": round(float(gross_profit), 2),
        "gross_loss": round(float(gross_loss), 2),
        "num_drawdowns": len(drawdowns),
    }


async def verify_account(wallet: str):
    """Verify metrics for a single account."""
    print(f"\n{'='*60}")
    print(f"VERIFYING METRICS FOR: {wallet}")
    print(f"{'='*60}")

    async with DataAPIClient() as client:
        print("\n1. Fetching raw trade AND redeem data...")
        trades, redeems = await fetch_all_trades(client, wallet)
        print(f"   Fetched {len(trades)} trades, {len(redeems)} redeems")

        if not trades:
            print("   ERROR: No trades found!")
            return

        print("\n2. Calculating metrics manually from raw data (including redeems)...")
        manual_metrics = calculate_metrics_manually(trades, redeems)

        print("\n3. MANUAL CALCULATION RESULTS:")
        print(f"   {'Metric':<25} {'Value':>15}")
        print(f"   {'-'*40}")
        for key, value in manual_metrics.items():
            print(f"   {key:<25} {str(value):>15}")

        # Load from analyzed file if available
        print("\n4. Checking against pre-analyzed data...")
        try:
            with open("PROJECT_ROOT / "data" / "analyzed_accounts.json"") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])
                stored = next((a for a in accounts if a.get("wallet", "").lower() == wallet.lower()), None)

                if stored:
                    print("\n   STORED VALUES (from analyzed_accounts.json):")
                    print(f"   {'Metric':<25} {'Stored':>12} {'Manual':>12} {'Match':>8}")
                    print(f"   {'-'*57}")

                    comparisons = [
                        ("num_trades", stored.get("num_trades"), manual_metrics["total_trades"]),
                        ("unique_markets", stored.get("unique_markets"), manual_metrics["unique_markets"]),
                        ("win_rate (%)", stored.get("win_rate", 0) * 100, manual_metrics["win_rate"]),
                        ("profit_factor", stored.get("profit_factor"), manual_metrics["profit_factor"]),
                        ("max_drawdown_pct", stored.get("max_drawdown_pct"), manual_metrics["max_drawdown_pct"]),
                        ("avg_position", stored.get("avg_position"), manual_metrics["avg_position_size"]),
                        ("pl_smoothness", stored.get("pl_curve_smoothness"), manual_metrics["pl_curve_smoothness"]),
                    ]

                    for name, stored_val, manual_val in comparisons:
                        if stored_val is None:
                            stored_val = 0
                        stored_val = round(float(stored_val), 2)
                        manual_val = round(float(manual_val), 2)
                        match = "✓" if abs(stored_val - manual_val) < 0.1 * max(abs(stored_val), abs(manual_val), 1) else "✗"
                        print(f"   {name:<25} {stored_val:>12} {manual_val:>12} {match:>8}")
                else:
                    print(f"   Account not found in pre-analyzed data")

        except FileNotFoundError:
            print("   Pre-analyzed file not found")
        except Exception as e:
            print(f"   Error loading pre-analyzed data: {e}")

        return manual_metrics


async def main():
    # Test with a few accounts from the analyzed file
    print("Loading sample accounts to verify...")

    try:
        with open("PROJECT_ROOT / "data" / "analyzed_accounts.json"") as f:
            data = json.load(f)
            accounts = data.get("accounts", [])

        # Pick accounts with different characteristics
        # 1. An account with non-zero drawdown
        dd_account = next((a for a in accounts if a.get("max_drawdown_pct", 0) > 5), None)
        # 2. A high win rate account
        wr_account = next((a for a in accounts if a.get("win_rate", 0) > 0.6), None)
        # 3. First account (usually high P/L)
        first_account = accounts[0] if accounts else None

        test_wallets = []
        if dd_account:
            test_wallets.append(("High Drawdown", dd_account["wallet"]))
        if wr_account:
            test_wallets.append(("High Win Rate", wr_account["wallet"]))
        if first_account and first_account["wallet"] not in [w for _, w in test_wallets]:
            test_wallets.append(("Top P/L", first_account["wallet"]))

        if not test_wallets:
            print("No accounts found to test!")
            return

        print(f"\nWill verify {len(test_wallets)} accounts:")
        for label, wallet in test_wallets:
            print(f"  - {label}: {wallet[:16]}...")

        for label, wallet in test_wallets:
            print(f"\n\n{'#'*60}")
            print(f"# Testing: {label}")
            print(f"{'#'*60}")
            await verify_account(wallet)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
