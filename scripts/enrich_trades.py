#!/usr/bin/env python3
"""
Trade enrichment script - fetch 50k trades per account and recompute metrics.

Reads accounts from data/analyzed_accounts.json, fetches additional trades,
and saves enriched data back.

Usage:
    python3 scripts/enrich_trades.py
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient
from src.api.gamma import GammaAPIClient
from src.discovery.analyzer import AccountAnalyzer
from src.discovery.models import DiscoveryMode


# Configuration
MAX_TRADES_PER_ACCOUNT = 3000  # Matches mass_scan MAX_TRADES filter
MAX_CONCURRENT = 5  # Concurrent account fetches
LOOKBACK_DAYS = 365  # 1 year of history
CHECKPOINT_INTERVAL = 25  # Save progress every N accounts


@dataclass
class EnrichmentStats:
    """Track enrichment progress."""
    total_accounts: int = 0
    completed: int = 0
    failed: int = 0
    total_trades_fetched: int = 0
    start_time: Optional[datetime] = None

    def elapsed_str(self) -> str:
        if not self.start_time:
            return "0:00"
        elapsed = (datetime.now() - self.start_time).total_seconds()
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"

    def eta_str(self) -> str:
        if not self.start_time or self.completed == 0:
            return "??:??"
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.completed / elapsed
        remaining = self.total_accounts - self.completed
        if rate > 0:
            eta_secs = remaining / rate
            mins, secs = divmod(int(eta_secs), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                return f"{hours}:{mins:02d}:{secs:02d}"
            return f"{mins}:{secs:02d}"
        return "??:??"


async def enrich_single_account(
    wallet: str,
    analyzer: AccountAnalyzer,
    existing_data: dict,
    stats: EnrichmentStats,
) -> Optional[dict]:
    """Fetch full trade history and recompute metrics for a single account."""
    try:
        # Run deep analysis with 50k trade limit
        result = await analyzer.deep_analysis(
            wallet_address=wallet,
            lookback_days=LOOKBACK_DAYS,
            include_insider_signals=False,  # Skip for speed
            max_trades=MAX_TRADES_PER_ACCOUNT,
        )

        if result.error or not result.pl_metrics:
            return None

        pl = result.pl_metrics
        pm = result.pattern_metrics

        # Track trades fetched
        trades_count = pm.total_trades if pm else 0
        stats.total_trades_fetched += trades_count

        # Compute position_consistency from std_dev / avg
        position_consistency = 0.0
        if pm and pm.avg_position_size_usd and float(pm.avg_position_size_usd) > 0:
            std_dev = float(pm.position_size_std_dev) if pm.position_size_std_dev else 0
            avg_pos = float(pm.avg_position_size_usd)
            position_consistency = 1 - min(1, std_dev / avg_pos) if avg_pos > 0 else 0

        # Build enriched account data - preserve existing fields, update computed ones
        enriched = {
            **existing_data,
            # Core metrics from new analysis
            "num_trades": trades_count,
            "trades_fetched": trades_count,
            "unique_markets": pm.unique_markets_traded if pm else existing_data.get("unique_markets", 0),

            # P/L metrics (don't override total_pnl from leaderboard - it's authoritative)
            "gross_profit": float(pl.gross_profit) if pl.gross_profit else 0.0,
            "gross_loss": float(pl.gross_loss) if pl.gross_loss else 0.0,
            # Compute pnl_per_trade and pnl_per_market from available data
            "pnl_per_trade": float(pl.total_realized_pnl) / trades_count if pl and trades_count > 0 else 0.0,
            "pnl_per_market": float(pl.total_realized_pnl) / (pl.markets_profitable + pl.markets_unprofitable) if pl and (pl.markets_profitable + pl.markets_unprofitable) > 0 else 0.0,

            # Win/loss metrics (using correct PLCurveMetrics attribute names)
            "win_rate": float(pl.win_rate) if pl.win_rate else 0.0,
            "profitable_trades": pl.win_count if pl else 0,
            "losing_trades": pl.loss_count if pl else 0,
            "avg_win_size": float(pl.avg_win_size) if pl.avg_win_size else 0.0,
            "avg_loss_size": float(pl.avg_loss_size) if pl.avg_loss_size else 0.0,
            "profit_factor": float(pl.profit_factor) if pl.profit_factor else 0.0,
            "market_win_rate": float(pl.market_win_rate) if pl.market_win_rate else 0.0,

            # Pattern metrics (using correct attribute names from TradingPatternMetrics)
            "avg_position_size": float(pm.avg_position_size_usd) if pm and pm.avg_position_size_usd else 0.0,
            "median_position": float(pm.median_position_size_usd) if pm and pm.median_position_size_usd else 0.0,
            "max_position": float(pm.max_position_size_usd) if pm and pm.max_position_size_usd else 0.0,
            "position_consistency": position_consistency,
            "buy_sell_ratio": float(pm.buy_sell_ratio) if pm and pm.buy_sell_ratio else 1.0,
            "pct_under_5c": float(pm.pct_trades_under_5c) if pm else 0.0,
            "pct_under_10c": float(pm.pct_trades_under_10c) if pm else 0.0,
            "pct_under_20c": float(pm.pct_trades_under_20c) if pm else 0.0,

            # Time metrics
            "account_age_days": pm.account_age_days if pm else existing_data.get("account_age_days", 0),
            "active_days": pm.active_days if pm else existing_data.get("active_days", 0),
            "activity_recency_days": pm.days_since_last_trade if pm else existing_data.get("activity_recency_days", 0),
            "is_currently_active": pm.days_since_last_trade <= 7 if pm else existing_data.get("is_currently_active", False),
            "trades_per_week": (pm.trades_per_day_avg * 7) if pm else existing_data.get("trades_per_week", 0.0),

            # Risk metrics
            "sharpe_ratio": float(pl.sharpe_ratio) if pl and pl.sharpe_ratio else 0.0,
            "sortino_ratio": float(pl.sortino_ratio) if pl and pl.sortino_ratio else 0.0,
            "max_drawdown_pct": float(pl.max_drawdown_pct) if pl and pl.max_drawdown_pct else 0.0,
            "avg_drawdown_pct": float(pl.avg_drawdown_pct) if pl and pl.avg_drawdown_pct else 0.0,
            "max_drawdown_usd": int(float(pl.total_realized_pnl) * pl.max_drawdown_pct / 100) if pl and pl.max_drawdown_pct else 0,

            # Enrichment metadata
            "_enriched_at": datetime.utcnow().isoformat(),
            "_trades_analyzed": trades_count,
        }

        return enriched

    except Exception as e:
        print(f"  ERROR: {wallet[:12]}... - {e}")
        return None


async def main():
    """Main enrichment process."""
    input_file = PROJECT_ROOT / "data" / "analyzed_accounts.json"
    output_file = PROJECT_ROOT / "data" / "analyzed_accounts_enriched.json"
    checkpoint_file = PROJECT_ROOT / "data" / "enrichment_checkpoint.json"

    # Load existing data
    print(f"Loading accounts from {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    accounts = data.get("accounts", [])
    print(f"Found {len(accounts)} accounts to enrich")

    # Check for checkpoint (resume interrupted enrichment)
    completed_wallets = set()
    enriched_accounts = []

    if checkpoint_file.exists():
        print(f"Found checkpoint, loading...")
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        completed_wallets = set(checkpoint.get("completed_wallets", []))
        enriched_accounts = checkpoint.get("enriched_accounts", [])
        print(f"  Resuming from checkpoint: {len(completed_wallets)} already done")

    # Filter to remaining accounts
    remaining = [a for a in accounts if a["wallet_address"].lower() not in completed_wallets]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All accounts already enriched!")
        return

    # Initialize
    stats = EnrichmentStats(
        total_accounts=len(remaining),
        start_time=datetime.now(),
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Create analyzer
    async with AccountAnalyzer(mode=DiscoveryMode.WIDE_NET_PROFITABILITY) as analyzer:

        async def process_one(account: dict) -> Optional[dict]:
            async with semaphore:
                wallet = account["wallet_address"]
                result = await enrich_single_account(wallet, analyzer, account, stats)

                stats.completed += 1
                if result:
                    return result
                else:
                    stats.failed += 1
                    return account  # Keep original if enrichment failed

        print(f"\n{'='*70}")
        print(f"Starting enrichment - {MAX_TRADES_PER_ACCOUNT:,} trades per account")
        print(f"Concurrent: {MAX_CONCURRENT} | Lookback: {LOOKBACK_DAYS} days")
        print(f"{'='*70}\n")

        # Process in batches for progress tracking
        batch_size = 10
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            # Process batch concurrently
            tasks = [process_one(acc) for acc in batch]
            results = await asyncio.gather(*tasks)

            # Add results
            for r in results:
                if r:
                    enriched_accounts.append(r)
                    completed_wallets.add(r["wallet_address"].lower())

            # Progress
            pct = (stats.completed / stats.total_accounts) * 100
            avg_trades = stats.total_trades_fetched / max(1, stats.completed - stats.failed)
            print(f"[{stats.elapsed_str()}] {stats.completed}/{stats.total_accounts} ({pct:.1f}%) "
                  f"| Failed: {stats.failed} | Avg trades: {avg_trades:,.0f} | ETA: {stats.eta_str()}")

            # Checkpoint
            if stats.completed % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    "completed_wallets": list(completed_wallets),
                    "enriched_accounts": enriched_accounts,
                    "stats": {
                        "completed": stats.completed,
                        "failed": stats.failed,
                        "total_trades": stats.total_trades_fetched,
                    }
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint, f)
                print(f"  [Checkpoint saved: {len(enriched_accounts)} accounts]")

    # Final save
    print(f"\n{'='*70}")
    print(f"Enrichment complete!")
    print(f"  Processed: {stats.completed}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total trades fetched: {stats.total_trades_fetched:,}")
    print(f"  Time: {stats.elapsed_str()}")
    print(f"{'='*70}\n")

    # Build output structure
    output_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "enriched_at": datetime.utcnow().isoformat(),
        "total_collected": data.get("total_collected", len(enriched_accounts)),
        "total_analyzed": len(enriched_accounts),
        "trades_per_account": MAX_TRADES_PER_ACCOUNT,
        "pnl_distribution": data.get("pnl_distribution", {}),
        "views_distribution": data.get("views_distribution", {}),
        "accounts": enriched_accounts,
    }

    # Save to new file first
    print(f"Saving to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Also update the original file
    print(f"Updating {input_file}...")
    with open(input_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint cleaned up")

    print(f"\nDone! {len(enriched_accounts)} accounts enriched with up to {MAX_TRADES_PER_ACCOUNT:,} trades each")


if __name__ == "__main__":
    asyncio.run(main())
