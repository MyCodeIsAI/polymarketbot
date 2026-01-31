#!/usr/bin/env python3
"""
Long-running side-by-side trade comparison logger.

Uses EXISTING reference trade collector data and bot signals.

Output: Creates timestamped log files in ./comparison_logs/

Usage:
    python compare_trades_logger.py

Run with nohup for long sessions:
    nohup python compare_trades_logger.py > comparison_logs/logger.out 2>&1 &
"""

import asyncio
import json
import os
import re
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

# Reference trade data
REFERENCE_WALLET = "0x93c22116e4402c9332ee6db578050e688934c072"
DATA_API = "https://data-api.polymarket.com"
BOT_SERVER = "http://localhost:8001"
LOG_DIR = Path("./comparison_logs")

# Poll intervals
POLL_INTERVAL = 2.0  # seconds
STATUS_INTERVAL = 60.0  # Print status every minute
SUMMARY_INTERVAL = 300.0  # Save summary every 5 minutes

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade:
    """Unified trade representation with full audit data."""
    timestamp: float
    source: str  # "REF" or "BOT"
    condition_id: str
    outcome: str  # "Up" or "Down"
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    slug: str = ""
    asset: str = ""

    # Audit fields
    window_offset: float = 0.0
    signal_type: str = ""
    is_15m_crypto: bool = False

    # Decision tracking
    decision_type: str = ""  # NEW, FILL, DOUBLE_DOWN, SWITCH, RAPID_DBL
    prev_outcome: str = ""
    gap_from_prev: float = 0.0
    price_change: float = 0.0


@dataclass
class Window:
    """A trading window with trades from both sources - keyed by condition_id."""
    window_id: int
    condition_id: str  # Primary key - same across REF and BOT
    slug: str  # Display slug (may differ between sources)
    asset: str
    window_start: float
    first_trade_time: float
    reference_trades: List[Trade] = field(default_factory=list)
    bot_trades: List[Trade] = field(default_factory=list)
    # Signal detection: first expensive trade (≥$0.70) establishes bias
    signal_detected: bool = False
    signal_time: float = 0.0
    signal_outcome: str = ""  # The direction REF signaled
    pre_signal_trades: int = 0  # Count of REF trades before signal (not recorded)


# =============================================================================
# State
# =============================================================================

@dataclass
class LoggerState:
    windows: Dict[str, Window] = field(default_factory=dict)
    current_window_id: int = 0
    seen_reference_hashes: Set[str] = field(default_factory=set)
    seen_bot_ids: Set[str] = field(default_factory=set)
    last_ref_trade: Dict[str, Trade] = field(default_factory=dict)
    last_bot_trade: Dict[str, Trade] = field(default_factory=dict)
    total_ref_trades: int = 0
    total_bot_trades: int = 0
    total_ref_decisions: int = 0
    total_bot_decisions: int = 0
    log_file: Optional[Path] = None
    summary_file: Optional[Path] = None
    trades_jsonl: Optional[Path] = None
    start_time: float = 0
    last_ref_check: float = 0


STATE = LoggerState()


# =============================================================================
# Helpers
# =============================================================================

def extract_window_start(slug: str) -> Optional[float]:
    """Extract window start timestamp from slug like 'btc-updown-15m-1769756400'."""
    match = re.search(r'-(\d{10})$', slug)
    if match:
        return float(match.group(1))
    return None


def is_15m_crypto(slug: str) -> bool:
    """Check if slug is a TRUE 15-minute crypto window.

    STRICT FILTERING: Only accepts slug pattern "*-updown-15m-*" (e.g., "btc-updown-15m-1769756400")
    REJECTS: "up-or-down-*-et" patterns which are actually hourly markets
    """
    slug_lower = slug.lower()
    # ONLY accept true 15-min format: "btc-updown-15m-1769756400"
    # Must contain "-updown-15m-" - this is the validated 15-min market pattern
    return "-updown-15m-" in slug_lower



def get_window_key(slug: str) -> str:
    """Extract a consistent window key from slug (asset + window timestamp)."""
    if not slug:
        return ""
    # Format: btc-updown-15m-1769837400
    parts = slug.lower().split("-")
    if len(parts) >= 4 and "15m" in parts:
        asset = parts[0]
        timestamp = parts[-1]  # The window timestamp at the end
        return f"{asset}_{timestamp}"
    return slug


def get_asset(slug: str) -> str:
    if not slug:
        return "UNK"
    parts = slug.split("-")
    return parts[0].upper()[:3] if parts else "UNK"


def format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def format_full_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Logging
# =============================================================================

def init_log_files():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE.log_file = LOG_DIR / f"comparison_{timestamp}.log"
    STATE.summary_file = LOG_DIR / f"summary_{timestamp}.json"
    STATE.trades_jsonl = LOG_DIR / f"trades_{timestamp}.jsonl"
    STATE.start_time = time.time()

    header = f"""
{'='*120}
TRADE COMPARISON LOGGER (TAKER-ONLY + TRUE 15-MIN MARKETS)
{'='*120}
Started: {datetime.now().isoformat()}
Reference Wallet: {REFERENCE_WALLET}
Data API: {DATA_API}
Bot Server: {BOT_SERVER}

FILTERING:
  - TAKER trades only (usdcSize > 0) - excludes market maker noise
  - TRUE 15-min markets only (slug: *-updown-15m-*) - excludes hourly/4h
  - POST-SIGNAL trades only: REF trades before first expensive (≥$0.70) are SKIPPED
    (Bot waits for signal, so pre-signal REF trades are not comparable)
{'='*120}

LEGEND:
  Source: REF=Reference, BOT=Bot
  Decision Types:
    NEW        = First trade for this outcome in window
    FILL       = Order fill (gap<=2s, price_change<$0.02)
    DOUBLE_DOWN= Same outcome, valid (gap>=6s OR price>=$0.03)
    SWITCH     = Different outcome from previous
    RAPID_DBL  = Same outcome, TOO FAST (gap<6s AND price<$0.03) ⚠️

{'='*120}
"""
    with open(STATE.log_file, "w") as f:
        f.write(header)
    print(header)
    print(f"Log: {STATE.log_file}")
    print(f"JSONL: {STATE.trades_jsonl}")


def log(msg: str, also_print: bool = True):
    if STATE.log_file:
        with open(STATE.log_file, "a") as f:
            f.write(msg + "\n")
    if also_print:
        print(msg)


def log_trade_jsonl(trade: Trade):
    if STATE.trades_jsonl:
        record = {
            "timestamp": trade.timestamp,
            "source": trade.source,
            "slug": trade.slug,
            "asset": trade.asset,
            "condition_id": trade.condition_id,
            "outcome": trade.outcome,
            "side": trade.side,
            "price": trade.price,
            "size": trade.size,
            "window_offset": trade.window_offset,
            "is_15m_crypto": trade.is_15m_crypto,
            "decision_type": trade.decision_type,
            "prev_outcome": trade.prev_outcome,
            "gap_from_prev": trade.gap_from_prev,
            "price_change": trade.price_change,
        }
        with open(STATE.trades_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")


def log_window_start(window: Window):
    log("")
    log("=" * 120)
    log(f"WINDOW {window.window_id}: {window.condition_id[:30]}...")
    log(f"Slug: {window.slug} | Asset: {window.asset}")
    log(f"Start: {format_full_ts(window.window_start)} | First Trade: {format_full_ts(window.first_trade_time)}")
    log(f"(Waiting for signal - first expensive ≥$0.70)")
    log("=" * 120)
    log("")
    log(f"{'TIME':<10} {'SRC':<4} {'SIDE':<5} {'OUT':<5} {'PRICE':<8} {'SIZE':<8} {'OFFSET':<8} {'DECISION':<12} {'DETAILS'}")
    log("-" * 120)


def log_trade(trade: Trade):
    time_str = format_ts(trade.timestamp)
    offset_str = f"{trade.window_offset:.0f}s" if trade.window_offset > 0 else "-"

    details = ""
    if trade.decision_type == "FILL":
        details = f"gap={trade.gap_from_prev:.1f}s Δp=${trade.price_change:.3f}"
    elif trade.decision_type == "DOUBLE_DOWN":
        gate = "TIME" if trade.gap_from_prev >= 6 else "PRICE"
        details = f"{gate}: gap={trade.gap_from_prev:.1f}s Δp=${trade.price_change:.3f}"
    elif trade.decision_type == "SWITCH":
        details = f"{trade.prev_outcome}→{trade.outcome} gap={trade.gap_from_prev:.1f}s"
    elif trade.decision_type == "RAPID_DBL":
        details = f"⚠️ gap={trade.gap_from_prev:.1f}s<6s AND Δp=${trade.price_change:.3f}<$0.03"

    flag = ""
    if trade.source == "BOT" and trade.decision_type == "RAPID_DBL":
        flag = " <<<< SHOULD BE BLOCKED"

    log(f"{time_str:<10} {trade.source:<4} {trade.side:<5} {trade.outcome:<5} ${trade.price:<7.3f} {trade.size:<8.1f} {offset_str:<8} {trade.decision_type:<12} {details}{flag}")
    log_trade_jsonl(trade)


def log_window_summary(window: Window):
    ref_buys = [t for t in window.reference_trades if t.side == "BUY"]
    bot_buys = [t for t in window.bot_trades if t.side == "BUY"]
    ref_decisions = count_decisions(ref_buys)
    bot_decisions = count_decisions(bot_buys)

    log("")
    log("-" * 120)
    log(f"WINDOW {window.window_id} SUMMARY: {window.slug}")

    # Signal info
    if window.signal_detected:
        signal_offset = window.signal_time - window.window_start
        log(f"  Signal: {window.signal_outcome} @ {signal_offset:.0f}s into window")
        log(f"  Pre-signal trades skipped: {window.pre_signal_trades}")
    else:
        log(f"  ⚠️ NO SIGNAL DETECTED (no expensive ≥$0.70 trade)")

    log(f"  Reference (post-signal): {len(ref_buys)} buys, {ref_decisions} decisions")
    log(f"  Bot:                     {len(bot_buys)} buys, {bot_decisions} decisions")

    if ref_buys:
        trade_ratio = len(bot_buys) / len(ref_buys)
        decision_ratio = bot_decisions / max(ref_decisions, 1)
        log(f"  Trade ratio:    {trade_ratio:.2f}x")
        log(f"  Decision ratio: {decision_ratio:.2f}x")

        if trade_ratio > 1.3:
            log(f"  ⚠️  BOT OVER-TRADING by {(trade_ratio-1)*100:.0f}%")
        elif trade_ratio < 0.7:
            log(f"  ⚠️  BOT UNDER-TRADING by {(1-trade_ratio)*100:.0f}%")
        else:
            log(f"  ✓ Within acceptable range")

    bot_rapid = sum(1 for t in bot_buys if t.decision_type == "RAPID_DBL")
    if bot_rapid > 0:
        log(f"  ⚠️  {bot_rapid} RAPID_DBL trades that should have been blocked!")
    log("")


def count_decisions(trades: List[Trade]) -> int:
    if not trades:
        return 0
    decisions = 0
    current = None
    for trade in sorted(trades, key=lambda t: t.timestamp):
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
            current = {"last_ts": trade.timestamp, "price": trade.price, "outcome": trade.outcome}
        else:
            current["last_ts"] = trade.timestamp
    return decisions


# =============================================================================
# Decision Analysis
# =============================================================================

def analyze_decision(trade: Trade, source: str) -> None:
    tracking = STATE.last_ref_trade if source == "REF" else STATE.last_bot_trade
    prev = tracking.get(trade.condition_id)

    if prev is None:
        trade.decision_type = "NEW"
    else:
        trade.gap_from_prev = trade.timestamp - prev.timestamp
        trade.price_change = abs(trade.price - prev.price)
        trade.prev_outcome = prev.outcome
        outcome_change = trade.outcome != prev.outcome
        same_outcome = trade.outcome == prev.outcome

        if outcome_change:
            trade.decision_type = "SWITCH"
        elif trade.gap_from_prev <= 2 and trade.price_change < 0.02:
            trade.decision_type = "FILL"
        elif same_outcome:
            time_gate = trade.gap_from_prev >= 6
            price_gate = trade.price_change >= 0.03
            if time_gate or price_gate:
                trade.decision_type = "DOUBLE_DOWN"
            else:
                trade.decision_type = "RAPID_DBL"
        else:
            trade.decision_type = "NEW"

    tracking[trade.condition_id] = trade

    if source == "REF":
        STATE.total_ref_trades += 1
        if trade.decision_type not in ["FILL"]:
            STATE.total_ref_decisions += 1
    else:
        STATE.total_bot_trades += 1
        if trade.decision_type not in ["FILL"]:
            STATE.total_bot_decisions += 1


# =============================================================================
# Reference Monitoring (from data-api)
# =============================================================================

async def fetch_reference_trades(session: aiohttp.ClientSession) -> List[Trade]:
    """Fetch TAKER trades only from the Polymarket data-api for reference wallet.

    FILTERING:
    - TAKER trades only: usdcSize > 0 (MAKER trades have usdcSize = 0)
    - TRUE 15-min markets only: slug contains "-updown-15m-"
    """
    try:
        url = f"{DATA_API}/activity?user={REFERENCE_WALLET}&limit=200"
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                log(f"Reference API error: {resp.status}")
                return []
            data = await resp.json()
    except Exception as e:
        log(f"Error fetching reference trades: {e}")
        return []

    trades = []
    now = time.time()
    cutoff = now - 3600  # Only process trades from last hour

    for item in data:
        try:
            tx_hash = item.get("transactionHash", "")
            if not tx_hash or tx_hash in STATE.seen_reference_hashes:
                continue

            # CRITICAL: Filter to TAKER trades only (usdcSize > 0)
            # MAKER trades have usdcSize = 0 (market making / rebate farming)
            usdc_size = float(item.get("usdcSize", 0) or 0)
            if usdc_size <= 0:
                continue  # Skip MAKER trades (no capital deployed)

            # Parse timestamp - API returns ISO format or unix
            ts_raw = item.get("timestamp", 0)
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
                except:
                    ts = float(ts_raw) if ts_raw.isdigit() else time.time()
            else:
                ts = float(ts_raw)

            if ts < cutoff:
                continue  # Skip old trades

            STATE.seen_reference_hashes.add(tx_hash)

            # Determine side from maker/taker perspective
            side = item.get("side", "").upper()
            # The API might use "buy"/"sell" or we need to infer from type
            if not side:
                trade_type = item.get("type", "")
                if trade_type in ["buy", "BUY"]:
                    side = "BUY"
                elif trade_type in ["sell", "SELL"]:
                    side = "SELL"
            if side not in ["BUY", "SELL"]:
                continue

            slug = item.get("slug", "") or item.get("eventSlug", "") or item.get("market_slug", "")
            if not is_15m_crypto(slug):
                continue  # Only TRUE 15-min markets (slug: *-updown-15m-*)

            window_start = extract_window_start(slug)
            window_offset = (ts - window_start) if window_start else 0

            trade = Trade(
                timestamp=ts,
                source="REF",
                condition_id=item.get("conditionId", "") or item.get("condition_id", ""),
                outcome=item.get("outcome", ""),
                side=side,
                price=float(item.get("price", 0)),
                size=float(item.get("size", 0) or item.get("amount", 0)),
                slug=slug,
                asset=get_asset(slug),
                window_offset=window_offset,
                is_15m_crypto=True,
            )
            trades.append(trade)
        except Exception as e:
            continue

    return sorted(trades, key=lambda t: t.timestamp)


async def monitor_reference(session: aiohttp.ClientSession):
    """Monitor reference trades by fetching from data-api.

    IMPORTANT: Only records trades AFTER signal detection (first expensive ≥$0.70).
    Pre-signal trades are counted but NOT recorded, since bot waits for signal.
    """
    while True:
        try:
            trades = await fetch_reference_trades(session)

            for trade in trades:
                cond_id = trade.condition_id
                if not cond_id:
                    continue

                window_key = get_window_key(trade.slug) or cond_id
                if window_key not in STATE.windows:
                    STATE.current_window_id += 1
                    window_start = extract_window_start(trade.slug) or trade.timestamp
                    window = Window(
                        window_id=STATE.current_window_id,
                        condition_id=cond_id,
                        slug=trade.slug,
                        asset=trade.asset,
                        window_start=window_start,
                        first_trade_time=trade.timestamp,
                    )
                    STATE.windows[window_key] = window
                    log_window_start(window)

                window = STATE.windows[window_key]

                # SIGNAL DETECTION: First expensive trade (≥$0.70) establishes bias
                # Pre-signal trades are NOT recorded (bot waits for signal)
                if not window.signal_detected:
                    if trade.price >= 0.70:
                        # THIS IS THE SIGNAL - first expensive trade
                        window.signal_detected = True
                        window.signal_time = trade.timestamp
                        window.signal_outcome = trade.outcome
                        log(f"  >>> SIGNAL DETECTED: {trade.outcome} @ ${trade.price:.2f} ({window.pre_signal_trades} pre-signal trades skipped)")
                    else:
                        # Pre-signal trade - count but don't record
                        window.pre_signal_trades += 1
                        continue  # Skip recording this trade

                # Post-signal: record the trade
                analyze_decision(trade, "REF")
                window.reference_trades.append(trade)
                log_trade(trade)

        except Exception as e:
            log(f"Reference monitor error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


# =============================================================================
# Bot Monitoring
# =============================================================================

async def fetch_bot_signals(session: aiohttp.ClientSession) -> List[Trade]:
    """Fetch bot signals - supports both old and predirectional bot formats."""
    try:
        url = f"{BOT_SERVER}/api/signals"
        async with session.get(url, timeout=5) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []

    trades = []
    now = time.time()
    cutoff = now - 3600  # Last hour

    # Handle response format: {"signals": [...]} or [...]
    signals = data.get("signals", data) if isinstance(data, dict) else data

    for item in signals:
        try:
            ts_str = item.get("timestamp", "")

            # Support both formats:
            # Old: condition_id, market_title
            # New (predirectional): slug (condition_id may be missing)
            cond_id = item.get("condition_id", "")
            slug = item.get("slug", "") or item.get("market_title", "")

            # If no condition_id, derive from slug
            if not cond_id and slug:
                cond_id = f"derived_{slug}"

            outcome = item.get("outcome", "")
            trade_id = f"{ts_str}_{cond_id}_{outcome}"

            if trade_id in STATE.seen_bot_ids:
                continue
            STATE.seen_bot_ids.add(trade_id)

            # Support both formats:
            # Old: action = "BUY" or "EXECUTE"
            # New: signal_type = "BIAS_DOMINANT", "BIAS_HEDGE", etc.
            action = item.get("action", "")
            signal_type = item.get("signal_type", "")

            # Accept if has action BUY/EXECUTE OR has signal_type (predirectional)
            if action not in ["BUY", "EXECUTE"] and not signal_type:
                continue

            if isinstance(ts_str, str):
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except:
                    ts = time.time()
            else:
                ts = float(ts_str) if ts_str else time.time()

            if ts < cutoff:
                continue

            # Handle slug from different sources
            if not slug:
                title = item.get("market_title", "")
                if "-15m-" in title.lower():
                    slug = title.lower().replace(" ", "-")
                else:
                    slug = title.lower().replace(" ", "-")

            if not is_15m_crypto(slug):
                continue

            window_start = extract_window_start(slug)
            window_offset = (ts - window_start) if window_start else 0

            # Support size from different formats
            size = float(item.get("size", 0) or item.get("size_usd", 0) or item.get("shares", 0))

            trade = Trade(
                timestamp=ts,
                source="BOT",
                condition_id=cond_id,
                outcome=outcome,
                side="BUY",
                price=float(item.get("price", 0)),
                size=size,
                slug=slug,
                asset=get_asset(slug),
                window_offset=window_offset,
                signal_type=signal_type or action,
                is_15m_crypto=True,
            )
            trades.append(trade)
        except Exception:
            continue

    return sorted(trades, key=lambda t: t.timestamp)


async def monitor_bot(session: aiohttp.ClientSession):
    while True:
        try:
            trades = await fetch_bot_signals(session)

            for trade in trades:
                cond_id = trade.condition_id
                if not cond_id:
                    continue

                window_key = get_window_key(trade.slug) or cond_id
                if window_key not in STATE.windows:
                    STATE.current_window_id += 1
                    window_start = extract_window_start(trade.slug) or trade.timestamp
                    window = Window(
                        window_id=STATE.current_window_id,
                        condition_id=cond_id,
                        slug=trade.slug,
                        asset=trade.asset,
                        window_start=window_start,
                        first_trade_time=trade.timestamp,
                    )
                    STATE.windows[window_key] = window
                    log_window_start(window)

                window = STATE.windows[window_key]
                analyze_decision(trade, "BOT")
                window.bot_trades.append(trade)
                log_trade(trade)

        except Exception as e:
            log(f"Bot monitor error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


# =============================================================================
# Periodic Tasks
# =============================================================================

async def periodic_status():
    while True:
        await asyncio.sleep(STATUS_INTERVAL)
        runtime = time.time() - STATE.start_time
        runtime_str = f"{int(runtime//3600)}h {int((runtime%3600)//60)}m"

        status = (
            f"\n[STATUS {format_ts(time.time())}] "
            f"Runtime: {runtime_str} | "
            f"Windows: {len(STATE.windows)} | "
            f"REF: {STATE.total_ref_trades} trades/{STATE.total_ref_decisions} decisions | "
            f"BOT: {STATE.total_bot_trades} trades/{STATE.total_bot_decisions} decisions"
        )
        if STATE.total_ref_decisions > 0:
            ratio = STATE.total_bot_decisions / STATE.total_ref_decisions
            status += f" | Ratio: {ratio:.2f}x"
        log(status)


async def periodic_summary():
    while True:
        await asyncio.sleep(SUMMARY_INTERVAL)
        save_summary()


def save_summary():
    summary = {
        "generated_at": datetime.now().isoformat(),
        "runtime_seconds": time.time() - STATE.start_time,
        "totals": {
            "windows": len(STATE.windows),
            "ref_trades": STATE.total_ref_trades,
            "bot_trades": STATE.total_bot_trades,
            "ref_decisions": STATE.total_ref_decisions,
            "bot_decisions": STATE.total_bot_decisions,
            "trade_ratio": STATE.total_bot_trades / max(STATE.total_ref_trades, 1),
            "decision_ratio": STATE.total_bot_decisions / max(STATE.total_ref_decisions, 1),
        },
        "windows": [],
    }

    for cond_id, window in STATE.windows.items():
        ref_buys = [t for t in window.reference_trades if t.side == "BUY"]
        bot_buys = [t for t in window.bot_trades if t.side == "BUY"]
        ref_decisions = count_decisions(ref_buys)
        bot_decisions = count_decisions(bot_buys)
        bot_rapid = sum(1 for t in bot_buys if t.decision_type == "RAPID_DBL")

        summary["windows"].append({
            "window_id": window.window_id,
            "condition_id": cond_id,
            "slug": window.slug,
            "asset": window.asset,
            "signal_detected": window.signal_detected,
            "signal_outcome": window.signal_outcome,
            "signal_time_offset": (window.signal_time - window.window_start) if window.signal_detected else None,
            "pre_signal_trades_skipped": window.pre_signal_trades,
            "ref_trades": len(ref_buys),
            "bot_trades": len(bot_buys),
            "ref_decisions": ref_decisions,
            "bot_decisions": bot_decisions,
            "trade_ratio": len(bot_buys) / max(len(ref_buys), 1),
            "decision_ratio": bot_decisions / max(ref_decisions, 1),
            "bot_rapid_doubles": bot_rapid,
        })

    if STATE.summary_file:
        with open(STATE.summary_file, "w") as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# Main
# =============================================================================

async def main():
    init_log_files()

    async with aiohttp.ClientSession() as session:
        try:
            await asyncio.gather(
                monitor_reference(session),
                monitor_bot(session),
                periodic_status(),
                periodic_summary(),
            )
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            log("\n" + "=" * 120)
            log("SESSION ENDED")
            log("=" * 120)
            for window in STATE.windows.values():
                log_window_summary(window)
            save_summary()
            log(f"\nFinal summary: {STATE.summary_file}")
            log(f"All trades: {STATE.trades_jsonl}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
        save_summary()
