#!/usr/bin/env python3
"""
Long-running side-by-side trade comparison logger.

Captures both reference account and bot trades in real-time, logging them
side-by-side with clear cycle breaks for easy auditing.

Output: Creates timestamped log files in ./comparison_logs/

Usage:
    python compare_trades_logger.py --reference-wallet 0x... --bot-server http://localhost:8000

The script will run until interrupted (Ctrl+C), creating detailed logs that can be
analyzed later to understand exactly how the bot differs from reference.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp

# =============================================================================
# Configuration
# =============================================================================

REFERENCE_WALLET = os.environ.get("REFERENCE_WALLET", "")
BOT_SERVER = os.environ.get("BOT_SERVER", "http://localhost:8000")
LOG_DIR = Path("./comparison_logs")
GAMMA_API = "https://gamma-api.polymarket.com"

# Poll intervals
REFERENCE_POLL_INTERVAL = 2.0  # seconds
BOT_POLL_INTERVAL = 1.0  # seconds
CYCLE_BOUNDARY_THRESHOLD = 30.0  # seconds gap = new cycle

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade:
    """Unified trade representation."""
    timestamp: float
    source: str  # "REFERENCE" or "BOT"
    condition_id: str
    outcome: str  # "Up" or "Down"
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    slug: str = ""
    asset: str = ""  # e.g., "BTC", "ETH"
    raw: dict = field(default_factory=dict)


@dataclass
class Cycle:
    """A trading cycle (window) with trades from both sources."""
    cycle_id: int
    start_time: float
    end_time: float = 0
    reference_trades: List[Trade] = field(default_factory=list)
    bot_trades: List[Trade] = field(default_factory=list)
    slug: str = ""


# =============================================================================
# State
# =============================================================================

@dataclass
class LoggerState:
    """Global state for the logger."""
    cycles: Dict[str, Cycle] = field(default_factory=dict)  # slug -> Cycle
    current_cycle_id: int = 0
    seen_reference_ids: Set[str] = field(default_factory=set)
    seen_bot_ids: Set[str] = field(default_factory=set)
    last_reference_trade_time: float = 0
    last_bot_trade_time: float = 0
    log_file: Optional[Path] = None
    summary_file: Optional[Path] = None


STATE = LoggerState()


# =============================================================================
# Logging Helpers
# =============================================================================

def init_log_files():
    """Initialize log files with timestamp."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE.log_file = LOG_DIR / f"trades_{timestamp}.log"
    STATE.summary_file = LOG_DIR / f"summary_{timestamp}.json"

    with open(STATE.log_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"TRADE COMPARISON LOG - Started {datetime.now().isoformat()}\n")
        f.write(f"Reference Wallet: {REFERENCE_WALLET}\n")
        f.write(f"Bot Server: {BOT_SERVER}\n")
        f.write("=" * 100 + "\n\n")

    print(f"Logging to: {STATE.log_file}")
    print(f"Summary to: {STATE.summary_file}")


def log_line(line: str, also_print: bool = True):
    """Write a line to the log file and optionally print."""
    if STATE.log_file:
        with open(STATE.log_file, "a") as f:
            f.write(line + "\n")
    if also_print:
        print(line)


def log_cycle_start(cycle: Cycle):
    """Log the start of a new trading cycle."""
    log_line("")
    log_line("=" * 100)
    log_line(f"CYCLE {cycle.cycle_id} START - {cycle.slug}")
    log_line(f"Time: {datetime.fromtimestamp(cycle.start_time).isoformat()}")
    log_line("=" * 100)
    log_line("")
    log_line(f"{'TIMESTAMP':<24} {'SOURCE':<10} {'ASSET':<5} {'SIDE':<5} {'OUTCOME':<6} {'PRICE':<8} {'SIZE':<10} {'DECISION'}")
    log_line("-" * 100)


def log_trade(trade: Trade, decision_info: str = ""):
    """Log a single trade."""
    ts_str = datetime.fromtimestamp(trade.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    source_color = "REF" if trade.source == "REFERENCE" else "BOT"
    log_line(
        f"{ts_str:<24} {source_color:<10} {trade.asset:<5} {trade.side:<5} {trade.outcome:<6} "
        f"${trade.price:<7.3f} {trade.size:<10.1f} {decision_info}"
    )


def log_cycle_summary(cycle: Cycle):
    """Log summary at end of cycle."""
    log_line("")
    log_line("-" * 100)
    log_line(f"CYCLE {cycle.cycle_id} SUMMARY")
    log_line(f"  Reference trades: {len(cycle.reference_trades)}")
    log_line(f"  Bot trades: {len(cycle.bot_trades)}")

    # Calculate decisions for reference
    ref_decisions = count_decisions(cycle.reference_trades)
    bot_decisions = count_decisions(cycle.bot_trades)

    log_line(f"  Reference decisions: {ref_decisions}")
    log_line(f"  Bot decisions: {bot_decisions}")

    ratio = len(cycle.bot_trades) / max(len(cycle.reference_trades), 1)
    decision_ratio = bot_decisions / max(ref_decisions, 1)

    log_line(f"  Trade ratio (bot/ref): {ratio:.2f}x")
    log_line(f"  Decision ratio (bot/ref): {decision_ratio:.2f}x")

    if ratio > 1.5:
        log_line(f"  ⚠️  BOT OVER-TRADING by {(ratio-1)*100:.0f}%")
    elif ratio < 0.7:
        log_line(f"  ⚠️  BOT UNDER-TRADING by {(1-ratio)*100:.0f}%")
    else:
        log_line(f"  ✓  Bot within acceptable range")

    log_line("")


def count_decisions(trades: List[Trade]) -> int:
    """
    Count distinct trading decisions from a list of trades.
    A new decision is triggered by:
    - Gap > 2s
    - Price change > $0.02
    - Outcome change
    """
    if not trades:
        return 0

    buys = sorted([t for t in trades if t.side == "BUY"], key=lambda x: x.timestamp)
    if not buys:
        return 0

    decisions = 0
    current = None

    for trade in buys:
        is_new = False
        if current is None:
            is_new = True
        else:
            gap = trade.timestamp - current["last_ts"]
            price_change = abs(trade.price - current["price"])
            outcome_change = trade.outcome != current["outcome"]
            if gap > 2 or price_change > 0.02 or outcome_change:
                is_new = True

        if is_new:
            decisions += 1
            current = {
                "last_ts": trade.timestamp,
                "price": trade.price,
                "outcome": trade.outcome,
            }
        else:
            current["last_ts"] = trade.timestamp

    return decisions


# =============================================================================
# Reference Account Monitoring
# =============================================================================

async def fetch_reference_trades(session: aiohttp.ClientSession) -> List[Trade]:
    """Fetch recent trades from reference account."""
    if not REFERENCE_WALLET:
        return []

    try:
        url = f"{GAMMA_API}/activity?user={REFERENCE_WALLET}&limit=100"
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception as e:
        print(f"Error fetching reference trades: {e}")
        return []

    trades = []
    for item in data:
        try:
            trade_id = item.get("id", "")
            if trade_id in STATE.seen_reference_ids:
                continue
            STATE.seen_reference_ids.add(trade_id)

            # Parse trade
            ts = item.get("timestamp", 0)
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

            side = item.get("side", "").upper()
            if side not in ["BUY", "SELL"]:
                continue

            # Determine our perspective (reference is the taker)
            our_side = side
            if item.get("type") == "MATCH":
                # If maker, flip the side
                our_side = "SELL" if side == "BUY" else "BUY"

            slug = item.get("market", {}).get("slug", "")
            asset = slug.split("-")[0].upper()[:3] if slug else "UNK"

            trade = Trade(
                timestamp=ts,
                source="REFERENCE",
                condition_id=item.get("conditionId", ""),
                outcome=item.get("outcome", ""),
                side=our_side,
                price=float(item.get("price", 0)),
                size=float(item.get("size", 0)),
                slug=slug,
                asset=asset,
                raw=item,
            )
            trades.append(trade)
        except Exception as e:
            print(f"Error parsing reference trade: {e}")
            continue

    return trades


async def monitor_reference(session: aiohttp.ClientSession):
    """Continuously monitor reference account for new trades."""
    while True:
        try:
            trades = await fetch_reference_trades(session)

            for trade in trades:
                # Check if this starts a new cycle
                slug = trade.slug
                if slug not in STATE.cycles:
                    STATE.current_cycle_id += 1
                    cycle = Cycle(
                        cycle_id=STATE.current_cycle_id,
                        start_time=trade.timestamp,
                        slug=slug,
                    )
                    STATE.cycles[slug] = cycle
                    log_cycle_start(cycle)

                cycle = STATE.cycles[slug]

                # Check for cycle boundary (long gap)
                if cycle.reference_trades:
                    last_ts = cycle.reference_trades[-1].timestamp
                    if trade.timestamp - last_ts > CYCLE_BOUNDARY_THRESHOLD:
                        # End old cycle
                        cycle.end_time = last_ts
                        log_cycle_summary(cycle)

                        # Start new cycle
                        STATE.current_cycle_id += 1
                        cycle = Cycle(
                            cycle_id=STATE.current_cycle_id,
                            start_time=trade.timestamp,
                            slug=slug,
                        )
                        STATE.cycles[slug] = cycle
                        log_cycle_start(cycle)

                # Add trade to cycle
                cycle.reference_trades.append(trade)
                STATE.last_reference_trade_time = trade.timestamp

                # Determine if this is a new decision or continuation
                decision_info = analyze_decision(trade, cycle.reference_trades, "REFERENCE")
                log_trade(trade, decision_info)

        except Exception as e:
            print(f"Reference monitor error: {e}")

        await asyncio.sleep(REFERENCE_POLL_INTERVAL)


# =============================================================================
# Bot Monitoring
# =============================================================================

async def fetch_bot_trades(session: aiohttp.ClientSession) -> List[Trade]:
    """Fetch recent trades from bot server."""
    try:
        url = f"{BOT_SERVER}/api/signals"
        async with session.get(url, timeout=5) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception as e:
        # Bot might not be running
        return []

    trades = []
    for item in data:
        try:
            # Create unique ID from signal data
            trade_id = f"{item.get('timestamp', '')}_{item.get('condition_id', '')}_{item.get('outcome', '')}"
            if trade_id in STATE.seen_bot_ids:
                continue
            STATE.seen_bot_ids.add(trade_id)

            # Only include executed trades (not skipped)
            action = item.get("action", "")
            if action not in ["BUY", "EXECUTE"]:
                continue

            # Parse timestamp
            ts_str = item.get("timestamp", "")
            if isinstance(ts_str, str):
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except:
                    ts = time.time()
            else:
                ts = float(ts_str) if ts_str else time.time()

            slug = item.get("market_title", "").lower().replace(" ", "-")
            asset = slug.split("-")[0].upper()[:3] if slug else "UNK"

            trade = Trade(
                timestamp=ts,
                source="BOT",
                condition_id=item.get("condition_id", ""),
                outcome=item.get("outcome", ""),
                side="BUY",  # Bot signals are buys
                price=float(item.get("price", 0)),
                size=float(item.get("size", 0)) if "size" in item else 0,
                slug=slug,
                asset=asset,
                raw=item,
            )
            trades.append(trade)
        except Exception as e:
            print(f"Error parsing bot trade: {e}")
            continue

    return trades


async def monitor_bot(session: aiohttp.ClientSession):
    """Continuously monitor bot for new trades."""
    while True:
        try:
            trades = await fetch_bot_trades(session)

            for trade in trades:
                slug = trade.slug

                # If we don't have a cycle for this market yet, create one
                if slug and slug not in STATE.cycles:
                    STATE.current_cycle_id += 1
                    cycle = Cycle(
                        cycle_id=STATE.current_cycle_id,
                        start_time=trade.timestamp,
                        slug=slug,
                    )
                    STATE.cycles[slug] = cycle
                    log_cycle_start(cycle)

                if slug and slug in STATE.cycles:
                    cycle = STATE.cycles[slug]
                    cycle.bot_trades.append(trade)

                    decision_info = analyze_decision(trade, cycle.bot_trades, "BOT")
                    log_trade(trade, decision_info)

                STATE.last_bot_trade_time = trade.timestamp

        except Exception as e:
            print(f"Bot monitor error: {e}")

        await asyncio.sleep(BOT_POLL_INTERVAL)


# =============================================================================
# Decision Analysis
# =============================================================================

def analyze_decision(trade: Trade, all_trades: List[Trade], source: str) -> str:
    """Analyze whether this trade is a new decision or continuation."""
    if len(all_trades) < 2:
        return "NEW_DECISION"

    # Find previous trade of same source and side
    prev_trades = [t for t in all_trades[:-1] if t.side == trade.side]
    if not prev_trades:
        return "NEW_DECISION"

    prev = prev_trades[-1]
    gap = trade.timestamp - prev.timestamp
    price_change = abs(trade.price - prev.price)
    outcome_change = trade.outcome != prev.outcome
    same_outcome = trade.outcome == prev.outcome

    if outcome_change:
        return f"SWITCH ({prev.outcome}→{trade.outcome})"

    if gap <= 2 and price_change < 0.02:
        return f"FILL (gap={gap:.1f}s, Δp=${price_change:.3f})"

    if same_outcome:
        if gap >= 6 or price_change >= 0.03:
            reason = "TIME" if gap >= 6 else "PRICE"
            return f"DOUBLE_DOWN/{reason} (gap={gap:.1f}s, Δp=${price_change:.3f})"
        else:
            return f"RAPID_DOUBLE (gap={gap:.1f}s, Δp=${price_change:.3f}) ⚠️"

    return f"NEW (gap={gap:.1f}s)"


# =============================================================================
# Summary Generation
# =============================================================================

def save_summary():
    """Save JSON summary of all cycles."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "reference_wallet": REFERENCE_WALLET,
        "bot_server": BOT_SERVER,
        "total_cycles": len(STATE.cycles),
        "cycles": [],
    }

    for slug, cycle in STATE.cycles.items():
        ref_decisions = count_decisions(cycle.reference_trades)
        bot_decisions = count_decisions(cycle.bot_trades)

        cycle_summary = {
            "cycle_id": cycle.cycle_id,
            "slug": slug,
            "start_time": cycle.start_time,
            "end_time": cycle.end_time or time.time(),
            "reference_trades": len(cycle.reference_trades),
            "bot_trades": len(cycle.bot_trades),
            "reference_decisions": ref_decisions,
            "bot_decisions": bot_decisions,
            "trade_ratio": len(cycle.bot_trades) / max(len(cycle.reference_trades), 1),
            "decision_ratio": bot_decisions / max(ref_decisions, 1),
        }
        summary["cycles"].append(cycle_summary)

    # Overall stats
    total_ref_trades = sum(len(c.reference_trades) for c in STATE.cycles.values())
    total_bot_trades = sum(len(c.bot_trades) for c in STATE.cycles.values())
    total_ref_decisions = sum(count_decisions(c.reference_trades) for c in STATE.cycles.values())
    total_bot_decisions = sum(count_decisions(c.bot_trades) for c in STATE.cycles.values())

    summary["totals"] = {
        "reference_trades": total_ref_trades,
        "bot_trades": total_bot_trades,
        "reference_decisions": total_ref_decisions,
        "bot_decisions": total_bot_decisions,
        "overall_trade_ratio": total_bot_trades / max(total_ref_trades, 1),
        "overall_decision_ratio": total_bot_decisions / max(total_ref_decisions, 1),
    }

    if STATE.summary_file:
        with open(STATE.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {STATE.summary_file}")


# =============================================================================
# Main
# =============================================================================

async def periodic_summary():
    """Periodically save summary and print status."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        save_summary()

        # Print quick status
        total_ref = sum(len(c.reference_trades) for c in STATE.cycles.values())
        total_bot = sum(len(c.bot_trades) for c in STATE.cycles.values())
        print(f"\n[STATUS] Cycles: {len(STATE.cycles)}, Ref trades: {total_ref}, Bot trades: {total_bot}")


async def main():
    """Main entry point."""
    global REFERENCE_WALLET, BOT_SERVER

    parser = argparse.ArgumentParser(description="Compare reference and bot trades")
    parser.add_argument("--reference-wallet", required=True, help="Reference wallet address")
    parser.add_argument("--bot-server", default="http://localhost:8000", help="Bot server URL")
    args = parser.parse_args()

    REFERENCE_WALLET = args.reference_wallet
    BOT_SERVER = args.bot_server

    print("=" * 60)
    print("TRADE COMPARISON LOGGER")
    print("=" * 60)
    print(f"Reference: {REFERENCE_WALLET[:10]}...{REFERENCE_WALLET[-6:]}")
    print(f"Bot: {BOT_SERVER}")
    print(f"Press Ctrl+C to stop and save summary")
    print("=" * 60)

    init_log_files()

    async with aiohttp.ClientSession() as session:
        try:
            await asyncio.gather(
                monitor_reference(session),
                monitor_bot(session),
                periodic_summary(),
            )
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            # Final summary
            log_line("\n" + "=" * 100)
            log_line("SESSION ENDED")
            log_line("=" * 100)

            for cycle in STATE.cycles.values():
                log_cycle_summary(cycle)

            save_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
        save_summary()
