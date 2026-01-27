#!/usr/bin/env python3
"""
Polymarket Arbitrage Live Test Server

Provides:
- Real-time price feeds from Polymarket CLOB API
- Actual order submission attempts (will fail without auth - tracks latency)
- Latency monitoring at each pipeline stage
- WebSocket for live dashboard updates
- REST API for trade history and stats
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field
from collections import deque
from statistics import mean, median
import logging
import random  # For simulation fill probability

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import httpx
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="Polymarket Arbitrage Live Test")

# =============================================================================
# Configuration - Optimal defaults from analysis
# =============================================================================

@dataclass
class Config:
    """Strategy parameters - optimized from wallet analysis."""

    # ==========================================================================
    # LIVE MODE TOGGLE - Set to True to submit real orders with API keys
    # ==========================================================================
    LIVE_MODE: bool = False  # <-- CHANGE TO True FOR LIVE TRADING

    # API Keys for live mode (loaded from environment variables)
    # Set these in your shell: export POLYMARKET_API_KEY=xxx POLYMARKET_API_SECRET=xxx
    API_KEY: str = ""
    API_SECRET: str = ""
    PRIVATE_KEY: str = ""  # Wallet private key for signing

    # Entry thresholds (from wallet analysis: optimized by time-to-resolution)
    # Default thresholds (used when time-to-resolution unknown)
    AGGRESSIVE_BUY_THRESHOLD: float = 0.25  # Always buy below this
    STANDARD_BUY_THRESHOLD: float = 0.35    # Buy if pair cost is good
    MAX_ENTRY_PRICE: float = 0.50           # Never buy above this

    # Dynamic thresholds by time-to-resolution (from 17k trade analysis)
    # Key insight: Prices drop 32% closer to resolution, more opportunities appear
    DYNAMIC_THRESHOLDS: bool = True  # Enable time-based threshold adjustment
    THRESHOLDS_BY_TIME = {
        # (min_seconds, max_seconds): (aggressive, standard)
        (0, 30): (0.28, 0.38),      # Final 30s: slightly wider (fill risk)
        (30, 60): (0.22, 0.32),     # 30-60s: tighter (30-60s anomaly in data)
        (60, 120): (0.25, 0.35),    # 1-2min: standard (sweet spot)
        (120, 300): (0.25, 0.35),   # 2-5min: standard (good volume)
        (300, 600): (0.22, 0.32),   # 5-10min: tighter (moderate opps)
        (600, 900): (0.20, 0.30),   # 10-15min: very tight (limited opps)
    }

    # Pair cost targets - for PAIR_ACCUMULATION strategy
    # Markets typically sum to ~$1.00. We accumulate cheap sides over time.
    # The profit comes from getting BOTH sides cheaply, not immediate arbitrage.
    TARGET_PAIR_COST: float = 0.95          # Ideal accumulated pair cost target
    MAX_PAIR_COST: float = 1.10             # Allow buying when market sums to $1.00+

    # Position limits
    MAX_POSITION_PER_MARKET: float = 500.0
    MAX_TOTAL_EXPOSURE: float = 2000.0
    MAX_CONCURRENT_MARKETS: int = 4

    # Timing
    PRICE_CHECK_INTERVAL_SEC: float = 1.0
    MIN_TIME_TO_RESOLUTION_SEC: int = 5  # Only skip if < 5 seconds to resolution (order wouldn't fill anyway)

    # Risk
    MIN_HEDGE_RATIO: float = 0.70           # Accept up to 30% imbalance

    # Paper trading
    STARTING_BALANCE: float = 200.0
    TRADE_SIZE_BASE: float = 15.0

    # ==========================================================================
    # ASSET FILTERING - Control which assets to trade
    # ==========================================================================
    # From wallet analysis:
    #   SOL: 49.7% trades < $0.25 (BEST) - START HERE
    #   ETH: 38.6% trades < $0.25
    #   BTC: 22.1% trades < $0.25 (high volume but fewer opportunities)
    #   XRP: 5.3% trades < $0.25 (AVOID)
    ENABLED_ASSETS: list = None  # None = all, or ["sol"] for SOL only

CONFIG = Config()

# Set to SOL-only for initial testing (best opportunity rate)
CONFIG.ENABLED_ASSETS = ["sol"]  # Change to ["sol", "eth", "btc"] or None for all

# Load API keys from environment if available
CONFIG.API_KEY = os.environ.get("POLYMARKET_API_KEY", "")
CONFIG.API_SECRET = os.environ.get("POLYMARKET_API_SECRET", "")
CONFIG.PRIVATE_KEY = os.environ.get("POLYMARKET_PRIVATE_KEY", "")

# =============================================================================
# Order Book Analysis for Realistic Fill Estimation
# =============================================================================

@dataclass
class OrderBookSnapshot:
    """Snapshot of order book at a point in time."""
    token_id: str
    timestamp: float
    best_ask: float = 0
    best_bid: float = 0
    ask_depth_10: float = 0  # Total shares available within 10% of best ask
    bid_depth_10: float = 0
    spread: float = 0
    asks: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    bids: List[tuple] = field(default_factory=list)

    def calculate_fill_price(self, side: str, size: float) -> tuple:
        """Calculate the actual fill price for a given order size.

        Returns: (fill_price, filled_size, slippage_pct, levels_consumed)
        """
        levels = self.asks if side == "BUY" else self.bids
        if not levels:
            return (0, 0, 0, 0)

        remaining = size
        total_cost = 0
        levels_consumed = 0

        for price, available in levels:
            levels_consumed += 1
            take = min(remaining, available)
            total_cost += take * price
            remaining -= take
            if remaining <= 0:
                break

        filled = size - remaining
        if filled <= 0:
            return (0, 0, 0, 0)

        avg_fill_price = total_cost / filled
        slippage_pct = ((avg_fill_price - self.best_ask) / self.best_ask * 100) if side == "BUY" and self.best_ask > 0 else 0

        return (avg_fill_price, filled, slippage_pct, levels_consumed)

    def get_fill_probability(self, side: str, size: float, max_price: float) -> float:
        """Estimate probability of fill based on available liquidity."""
        levels = self.asks if side == "BUY" else self.bids
        if not levels:
            return 0

        available_at_price = sum(s for p, s in levels if p <= max_price) if side == "BUY" else sum(s for p, s in levels if p >= max_price)

        if available_at_price >= size:
            return 0.95  # High probability if liquidity exists
        elif available_at_price >= size * 0.5:
            return 0.70  # Partial fill likely
        elif available_at_price > 0:
            return 0.40  # Low probability
        return 0.05  # Very low - may get front-run

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id[:16] + "..." if len(self.token_id) > 16 else self.token_id,
            "timestamp": self.timestamp,
            "best_ask": self.best_ask,
            "best_bid": self.best_bid,
            "spread": round(self.spread, 4),
            "ask_depth_10": self.ask_depth_10,
            "bid_depth_10": self.bid_depth_10,
            "ask_levels": len(self.asks),
            "bid_levels": len(self.bids),
        }


@dataclass
class RealisticFillEstimate:
    """Realistic estimate of what a fill would look like."""
    signal_price: float          # Price that triggered signal (outcomePrices)
    best_ask: float              # Actual best ask in order book
    estimated_fill_price: float  # VWAP for our order size
    slippage_pct: float          # Slippage from best ask
    available_liquidity: float   # Shares available at or near our price
    fill_probability: float      # Estimated probability of fill
    levels_consumed: int         # How many price levels we'd eat through
    would_fill: bool             # Whether order would likely fill
    reason: str                  # Explanation

    def to_dict(self) -> dict:
        return {
            "signal_price": round(self.signal_price, 4),
            "best_ask": round(self.best_ask, 4),
            "estimated_fill_price": round(self.estimated_fill_price, 4),
            "slippage_pct": round(self.slippage_pct, 2),
            "available_liquidity": round(self.available_liquidity, 2),
            "fill_probability": round(self.fill_probability, 2),
            "levels_consumed": self.levels_consumed,
            "would_fill": self.would_fill,
            "reason": self.reason,
        }


@dataclass
class DetailedTradeLog:
    """Comprehensive trade log for post-analysis."""
    timestamp: str
    market_slug: str
    condition_id: str
    outcome: str

    # Signal info
    signal_type: str
    signal_price: float
    pair_cost: float

    # Order book state at signal time
    order_book: dict

    # Fill estimation
    fill_estimate: dict

    # Order attempt
    order_size_usd: float
    order_size_shares: float
    order_type: str

    # Timing
    detection_ms: float
    submission_ms: float
    e2e_ms: float

    # Result
    order_status: str
    error_message: Optional[str]

    # Competition tracking
    liquidity_before: float
    liquidity_after: float  # Filled after next check
    was_front_run: bool  # Did liquidity disappear?

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "market_slug": self.market_slug,
            "condition_id": self.condition_id,
            "outcome": self.outcome,
            "signal_type": self.signal_type,
            "signal_price": self.signal_price,
            "pair_cost": self.pair_cost,
            "order_book": self.order_book,
            "fill_estimate": self.fill_estimate,
            "order_size_usd": self.order_size_usd,
            "order_size_shares": self.order_size_shares,
            "order_type": self.order_type,
            "detection_ms": self.detection_ms,
            "submission_ms": self.submission_ms,
            "e2e_ms": self.e2e_ms,
            "order_status": self.order_status,
            "error_message": self.error_message,
            "liquidity_before": self.liquidity_before,
            "liquidity_after": self.liquidity_after,
            "was_front_run": self.was_front_run,
        }


# Trade log storage
TRADE_LOGS: List[DetailedTradeLog] = []
TRADE_LOG_FILE = Path(__file__).parent / "simulation_trades.jsonl"

# Order book cache for competition tracking
ORDER_BOOK_CACHE: Dict[str, OrderBookSnapshot] = {}


def log_trade(trade: DetailedTradeLog):
    """Append trade to log file for post-analysis."""
    TRADE_LOGS.append(trade)
    try:
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write trade log: {e}")


async def fetch_order_book_detailed(token_id: str) -> Optional[OrderBookSnapshot]:
    """Fetch and parse order book with full depth analysis."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{CLOB_API}/book", params={"token_id": token_id})
            if resp.status_code != 200:
                return None

            data = resp.json()
            asks_raw = data.get("asks", [])
            bids_raw = data.get("bids", [])

            # Parse into (price, size) tuples
            def parse_levels(levels):
                parsed = []
                for level in levels:
                    if isinstance(level, dict):
                        parsed.append((float(level["price"]), float(level["size"])))
                    else:
                        parsed.append((float(level[0]), float(level[1])))
                return parsed

            asks = parse_levels(asks_raw)
            bids = parse_levels(bids_raw)

            # Sort: asks ascending, bids descending
            asks.sort(key=lambda x: x[0])
            bids.sort(key=lambda x: x[0], reverse=True)

            best_ask = asks[0][0] if asks else 0
            best_bid = bids[0][0] if bids else 0

            # Calculate depth within 10% of best price
            ask_depth = sum(s for p, s in asks if p <= best_ask * 1.10) if best_ask > 0 else 0
            bid_depth = sum(s for p, s in bids if p >= best_bid * 0.90) if best_bid > 0 else 0

            snapshot = OrderBookSnapshot(
                token_id=token_id,
                timestamp=time.time(),
                best_ask=best_ask,
                best_bid=best_bid,
                ask_depth_10=ask_depth,
                bid_depth_10=bid_depth,
                spread=best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0,
                asks=asks[:20],  # Keep top 20 levels
                bids=bids[:20],
            )

            # Cache for competition tracking
            ORDER_BOOK_CACHE[token_id] = snapshot

            return snapshot

        except Exception as e:
            logger.debug(f"Error fetching order book: {e}")
            return None


def estimate_fill(book: OrderBookSnapshot, side: str, size_usd: float, signal_price: float) -> RealisticFillEstimate:
    """Estimate realistic fill based on order book."""
    if not book or not book.asks:
        return RealisticFillEstimate(
            signal_price=signal_price,
            best_ask=0,
            estimated_fill_price=signal_price,
            slippage_pct=0,
            available_liquidity=0,
            fill_probability=0,
            levels_consumed=0,
            would_fill=False,
            reason="No order book data available",
        )

    # Calculate shares for our USD amount
    shares = size_usd / signal_price if signal_price > 0 else 0

    # Get fill estimate
    fill_price, filled, slippage, levels = book.calculate_fill_price(side, shares)

    # Get fill probability
    max_price = signal_price * 1.05  # Allow 5% slippage
    prob = book.get_fill_probability(side, shares, max_price)

    # Determine if would fill
    would_fill = filled >= shares * 0.95 and slippage < 10  # 95% filled, <10% slippage

    reason = ""
    if filled < shares * 0.5:
        reason = f"Insufficient liquidity: only {filled:.0f} of {shares:.0f} shares available"
    elif slippage > 5:
        reason = f"High slippage: {slippage:.1f}% above best ask"
    elif prob < 0.5:
        reason = f"Low fill probability: {prob:.0%}"
    else:
        reason = f"Good fill: {filled:.0f} shares at ${fill_price:.4f} ({slippage:.2f}% slippage)"

    return RealisticFillEstimate(
        signal_price=signal_price,
        best_ask=book.best_ask,
        estimated_fill_price=fill_price if fill_price > 0 else signal_price,
        slippage_pct=slippage,
        available_liquidity=book.ask_depth_10 if side == "BUY" else book.bid_depth_10,
        fill_probability=prob,
        levels_consumed=levels,
        would_fill=would_fill,
        reason=reason,
    )


# =============================================================================
# Latency Tracking (from latency_monitor.py patterns)
# =============================================================================

@dataclass
class LatencyMetrics:
    """Track latency at each pipeline stage."""
    detection_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    validation_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    order_build_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    submission_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    e2e_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Counters
    signals_detected: int = 0
    orders_attempted: int = 0
    orders_rejected: int = 0
    api_errors: int = 0

    def record_detection(self, ms: float):
        self.detection_ms.append(ms)
        self.signals_detected += 1

    def record_validation(self, ms: float):
        self.validation_ms.append(ms)

    def record_order_build(self, ms: float):
        self.order_build_ms.append(ms)

    def record_submission(self, ms: float, success: bool, is_api_error: bool = False):
        self.submission_ms.append(ms)
        self.orders_attempted += 1
        if not success:
            self.orders_rejected += 1
        if is_api_error:
            self.api_errors += 1

    def record_e2e(self, ms: float):
        self.e2e_ms.append(ms)

    def _calc_stats(self, data: deque) -> dict:
        if not data:
            return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "count": 0}
        sorted_data = sorted(data)
        return {
            "avg": round(mean(data), 2),
            "min": round(min(data), 2),
            "max": round(max(data), 2),
            "p50": round(median(data), 2),
            "p95": round(sorted_data[int(len(sorted_data) * 0.95)] if len(sorted_data) >= 20 else max(data), 2),
            "count": len(data),
        }

    def to_dict(self) -> dict:
        return {
            "detection": self._calc_stats(self.detection_ms),
            "validation": self._calc_stats(self.validation_ms),
            "order_build": self._calc_stats(self.order_build_ms),
            "submission": self._calc_stats(self.submission_ms),
            "e2e": self._calc_stats(self.e2e_ms),
            "signals_detected": self.signals_detected,
            "orders_attempted": self.orders_attempted,
            "orders_rejected": self.orders_rejected,
            "api_errors": self.api_errors,
            "success_rate": round((self.orders_attempted - self.orders_rejected) / max(1, self.orders_attempted) * 100, 1),
        }

LATENCY = LatencyMetrics()

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class MarketPrice:
    """Current prices for a market."""
    condition_id: str
    slug: str
    title: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    up_liquidity: float
    down_liquidity: float
    pair_cost: float
    timestamp: datetime
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "title": self.title,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "up_price": self.up_price,
            "down_price": self.down_price,
            "up_liquidity": self.up_liquidity,
            "down_liquidity": self.down_liquidity,
            "pair_cost": self.pair_cost,
            "timestamp": self.timestamp.isoformat(),
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
        }


@dataclass
class Signal:
    """Trading signal."""
    timestamp: datetime
    condition_id: str
    market_title: str
    outcome: str  # "Up" or "Down"
    token_id: str
    price: float
    signal_type: str  # "AGGRESSIVE" or "STANDARD"
    pair_cost: float
    action: str  # "BUY" or "SKIP"
    reason: str
    latency_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "condition_id": self.condition_id,
            "market_title": self.market_title,
            "outcome": self.outcome,
            "token_id": self.token_id,
            "price": self.price,
            "signal_type": self.signal_type,
            "pair_cost": self.pair_cost,
            "action": self.action,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class OrderAttempt:
    """Record of an order submission attempt."""
    timestamp: datetime
    condition_id: str
    market_title: str
    token_id: str
    outcome: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    order_type: str  # "FOK", "GTC", etc.

    # Timing
    detection_ms: float = 0
    validation_ms: float = 0
    order_build_ms: float = 0
    submission_ms: float = 0
    e2e_ms: float = 0

    # Result
    success: bool = False
    clob_order_id: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "condition_id": self.condition_id,
            "market_title": self.market_title,
            "token_id": self.token_id[:16] + "..." if len(self.token_id) > 16 else self.token_id,
            "outcome": self.outcome,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "order_type": self.order_type,
            "detection_ms": round(self.detection_ms, 2),
            "validation_ms": round(self.validation_ms, 2),
            "order_build_ms": round(self.order_build_ms, 2),
            "submission_ms": round(self.submission_ms, 2),
            "e2e_ms": round(self.e2e_ms, 2),
            "success": self.success,
            "clob_order_id": self.clob_order_id,
            "error_message": self.error_message,
        }


@dataclass
class Position:
    """Tracked position in a market (paper trading)."""
    condition_id: str
    market_title: str
    up_shares: float = 0
    up_cost: float = 0
    down_shares: float = 0
    down_cost: float = 0
    trades_count: int = 0
    opened_at: Optional[datetime] = None

    @property
    def up_avg_price(self) -> float:
        return self.up_cost / self.up_shares if self.up_shares > 0 else 0

    @property
    def down_avg_price(self) -> float:
        return self.down_cost / self.down_shares if self.down_shares > 0 else 0

    @property
    def pair_cost(self) -> float:
        return self.up_avg_price + self.down_avg_price

    @property
    def total_cost(self) -> float:
        return self.up_cost + self.down_cost

    @property
    def hedge_ratio(self) -> float:
        if self.up_shares == 0 or self.down_shares == 0:
            return 0
        return min(self.up_shares, self.down_shares) / max(self.up_shares, self.down_shares)

    @property
    def guaranteed_payout(self) -> float:
        return min(self.up_shares, self.down_shares)

    @property
    def theoretical_profit(self) -> float:
        return self.guaranteed_payout - self.total_cost

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "market_title": self.market_title,
            "up_shares": self.up_shares,
            "up_cost": self.up_cost,
            "up_avg_price": round(self.up_avg_price, 4),
            "down_shares": self.down_shares,
            "down_cost": self.down_cost,
            "down_avg_price": round(self.down_avg_price, 4),
            "pair_cost": round(self.pair_cost, 4),
            "total_cost": round(self.total_cost, 2),
            "hedge_ratio": round(self.hedge_ratio, 4),
            "guaranteed_payout": round(self.guaranteed_payout, 2),
            "theoretical_profit": round(self.theoretical_profit, 2),
            "trades_count": self.trades_count,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


@dataclass
class PaperAccount:
    """Paper trading account."""
    balance: float = 1000.0
    starting_balance: float = 1000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    order_attempts: List[OrderAttempt] = field(default_factory=list)

    @property
    def open_exposure(self) -> float:
        return sum(p.total_cost for p in self.positions.values())

    @property
    def open_positions_count(self) -> int:
        return len([p for p in self.positions.values() if p.total_cost > 0])

    def to_dict(self) -> dict:
        return {
            "balance": round(self.balance, 2),
            "starting_balance": self.starting_balance,
            "open_exposure": round(self.open_exposure, 2),
            "open_positions": self.open_positions_count,
            "total_order_attempts": len(self.order_attempts),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }


@dataclass
class AppState:
    """Application state."""
    is_running: bool = False
    markets: Dict[str, MarketPrice] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)
    account: PaperAccount = field(default_factory=PaperAccount)
    connected_clients: Set[WebSocket] = field(default_factory=set)
    last_price_update: Optional[datetime] = None

STATE = AppState()

# =============================================================================
# CLOB API Client (simplified, using existing patterns)
# =============================================================================

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

async def fetch_crypto_markets() -> List[dict]:
    """Fetch active 15-minute crypto markets from Gamma API using series endpoint.

    IMPORTANT: We find the market CLOSEST TO EXPIRY for each asset, as that's
    where the price action happens. Markets far from expiry sit at $0.99/$0.99.
    """
    markets = []
    # Known 15m series slugs
    all_series = {
        "btc": "btc-up-or-down-15m",
        "eth": "eth-up-or-down-15m",
        "sol": "sol-up-or-down-15m",
        "xrp": "xrp-up-or-down-15m",
    }

    # Filter by enabled assets
    if CONFIG.ENABLED_ASSETS:
        series_slugs = [all_series[a.lower()] for a in CONFIG.ENABLED_ASSETS if a.lower() in all_series]
        logger.debug(f"Checking enabled assets: {CONFIG.ENABLED_ASSETS}")
    else:
        series_slugs = list(all_series.values())
        logger.debug("Checking all assets")

    async with httpx.AsyncClient(timeout=15.0) as client:
        for series_slug in series_slugs:
            try:
                # Fetch series to get all active events
                resp = await client.get(f"{GAMMA_API}/series", params={"slug": series_slug})
                if resp.status_code != 200:
                    continue
                series_data = resp.json()
                if not series_data:
                    continue

                series = series_data[0] if isinstance(series_data, list) else series_data
                events = series.get("events", [])

                # Get ALL active events, then find the one closest to expiry
                all_active = [e for e in events if e.get("active") and not e.get("closed")]

                # Parse end timestamp from slug (embedded like btc-updown-15m-1769499900)
                # The number is the START timestamp - so end time is start + 900 (15 min)
                def get_end_timestamp(e):
                    slug = e.get("slug", "")
                    parts = slug.split("-")
                    if parts:
                        try:
                            start_ts = int(parts[-1])
                            return start_ts + 900  # End time = start + 15 minutes
                        except:
                            return float('inf')
                    return float('inf')

                # CRITICAL: Filter out expired markets FIRST, then sort
                # NOTE: Use time.time() NOT datetime.utcnow().timestamp() - the latter
                # is wrong for naive datetimes (adds local TZ offset!)
                now_ts = int(time.time())
                active_events = [e for e in all_active if get_end_timestamp(e) > now_ts]

                # Sort by end timestamp (soonest first)
                active_events.sort(key=get_end_timestamp)

                logger.debug(f"{series_slug}: {len(all_active)} total active, {len(active_events)} not expired")

                # Log what we found (reuse now_ts from above)
                for e in active_events[:3]:
                    slug = e.get("slug", "")
                    end_ts = get_end_timestamp(e)
                    mins_left = (end_ts - now_ts) / 60
                    logger.debug(f"  {series_slug}: {slug} expires in {mins_left:.0f}min")

                # Get the market closest to expiry (already filtered to non-expired)
                for i, event in enumerate(active_events[:5]):  # Check first 5 closest
                    event_slug = event.get("slug", "")
                    end_ts = get_end_timestamp(event)
                    mins_left = (end_ts - now_ts) / 60

                    logger.debug(f"Checking {event_slug}: {mins_left:.1f}min left")

                    market_resp = await client.get(
                        f"{GAMMA_API}/markets",
                        params={"slug": event_slug}
                    )
                    if market_resp.status_code == 200:
                        market_data = market_resp.json()
                        if market_data and len(market_data) > 0:
                            m = market_data[0]
                            if m.get("acceptingOrders", False):
                                markets.append(m)
                                end_date = m.get("endDate", "")
                                mins_left = (end_ts - now_ts) / 60
                                logger.info(f"Found ACTIVE 15m market: {m.get('question', '')[:50]} - {mins_left:.1f}min left")
                                break  # Found the closest accepting-orders market that hasn't expired

            except Exception as e:
                logger.warning(f"Error fetching {series_slug}: {e}")

    return markets


async def fetch_order_book(token_id: str) -> Optional[dict]:
    """Fetch order book for a token from CLOB API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{CLOB_API}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"Error fetching order book for {token_id[:16]}...: {e}")
            return None


async def submit_order_simulation(token_id: str, side: str, price: float, size: float, fill_estimate: RealisticFillEstimate) -> dict:
    """
    SIMULATION MODE: Simulate order submission with realistic latency.

    Uses order book data to estimate whether the order would fill.
    """
    t_start = time.perf_counter()

    # Simulate network latency (50-150ms typical for API calls)
    simulated_latency_ms = random.uniform(50, 150)
    await asyncio.sleep(simulated_latency_ms / 1000)

    submission_ms = (time.perf_counter() - t_start) * 1000

    # Determine simulated success based on fill estimate
    would_fill = fill_estimate.would_fill if fill_estimate else True
    fill_prob = fill_estimate.fill_probability if fill_estimate else 0.9

    # Roll the dice based on fill probability
    success = random.random() < fill_prob if would_fill else False

    return {
        "success": success,
        "submission_ms": submission_ms,
        "status_code": 200 if success else 400,
        "error_message": None if success else f"SIMULATION: Fill prob {fill_prob:.0%}, would_fill={would_fill}",
        "clob_order_id": f"SIM-{int(time.time()*1000)}" if success else None,
        "is_api_error": False,
        "is_simulation": True,
    }


async def submit_order_live(token_id: str, side: str, price: float, size: float, order_type: str = "FOK") -> dict:
    """
    LIVE MODE: Submit actual order to CLOB API with authentication.

    Requires POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and POLYMARKET_PRIVATE_KEY
    environment variables to be set.
    """
    t_start = time.perf_counter()

    # Check for credentials
    if not CONFIG.API_KEY or not CONFIG.PRIVATE_KEY:
        return {
            "success": False,
            "submission_ms": (time.perf_counter() - t_start) * 1000,
            "error_message": "LIVE MODE ERROR: Missing API credentials. Set POLYMARKET_API_KEY and POLYMARKET_PRIVATE_KEY",
            "is_api_error": True,
            "is_simulation": False,
        }

    # Build order payload
    order_data = {
        "tokenID": token_id,
        "side": side,
        "price": str(price),
        "size": str(size),
        "type": order_type,
    }

    # TODO: Add proper CLOB authentication headers here
    # This requires py-clob-client or manual signature generation
    # For now, we attempt unauthenticated to track latency

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # In full live mode, you'd use py-clob-client here:
            # from py_clob_client.client import ClobClient
            # client = ClobClient(host=CLOB_API, key=CONFIG.API_KEY, ...)
            # response = client.create_order(...)

            resp = await client.post(
                f"{CLOB_API}/order",
                json=order_data,
                headers={
                    "Content-Type": "application/json",
                    # Add auth headers here when implementing full live mode
                }
            )
            submission_ms = (time.perf_counter() - t_start) * 1000

            result = {
                "success": False,
                "submission_ms": submission_ms,
                "status_code": resp.status_code,
                "error_message": None,
                "clob_order_id": None,
                "is_api_error": False,
                "is_simulation": False,
            }

            if resp.status_code == 200:
                data = resp.json()
                result["success"] = data.get("success", False)
                result["clob_order_id"] = data.get("orderID") or data.get("order_id")
                if not result["success"]:
                    result["error_message"] = data.get("error", "Unknown error")
            elif resp.status_code == 401:
                result["error_message"] = "Authentication required - check API credentials"
            elif resp.status_code == 403:
                result["error_message"] = "Forbidden - API key may lack permissions"
            elif resp.status_code == 400:
                try:
                    data = resp.json()
                    result["error_message"] = data.get("error", f"Bad request: {resp.text[:100]}")
                except:
                    result["error_message"] = f"Bad request: {resp.text[:100]}"
            else:
                result["error_message"] = f"HTTP {resp.status_code}: {resp.text[:100]}"
                result["is_api_error"] = True

            return result

        except httpx.TimeoutException:
            return {
                "success": False,
                "submission_ms": (time.perf_counter() - t_start) * 1000,
                "error_message": "Request timeout",
                "is_api_error": True,
                "is_simulation": False,
            }
        except Exception as e:
            return {
                "success": False,
                "submission_ms": (time.perf_counter() - t_start) * 1000,
                "error_message": str(e),
                "is_api_error": True,
                "is_simulation": False,
            }


async def submit_order(token_id: str, side: str, price: float, size: float, order_type: str = "FOK", fill_estimate: RealisticFillEstimate = None) -> dict:
    """
    Submit order - routes to simulation or live mode based on CONFIG.LIVE_MODE.

    This is the main entry point for order submission. Toggle CONFIG.LIVE_MODE
    at the top of this file to switch between simulation and live trading.
    """
    if CONFIG.LIVE_MODE:
        logger.info(f"LIVE ORDER: {side} {size:.2f} shares of {token_id[:16]}... @ ${price:.4f}")
        return await submit_order_live(token_id, side, price, size, order_type)
    else:
        logger.debug(f"SIM ORDER: {side} {size:.2f} shares @ ${price:.4f}")
        return await submit_order_simulation(token_id, side, price, size, fill_estimate)


def parse_market_with_tokens(market_data: dict) -> Optional[dict]:
    """Parse market data including token IDs."""
    condition_id = market_data.get("conditionId") or market_data.get("condition_id")
    if not condition_id:
        return None

    # Get token IDs from market data
    tokens = market_data.get("tokens", [])
    up_token_id = None
    down_token_id = None

    for token in tokens:
        outcome = (token.get("outcome", "") or "").lower()
        token_id = token.get("token_id") or token.get("tokenId")
        if outcome in ("yes", "up"):
            up_token_id = token_id
        elif outcome in ("no", "down"):
            down_token_id = token_id

    # Also check clobTokenIds format (may be JSON string or list)
    clob_token_ids = market_data.get("clobTokenIds", [])
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            clob_token_ids = []
    if len(clob_token_ids) >= 2 and not up_token_id:
        up_token_id = clob_token_ids[0]
        down_token_id = clob_token_ids[1]

    return {
        "condition_id": condition_id,
        "slug": market_data.get("slug", ""),
        "title": market_data.get("question", "") or market_data.get("title", "Unknown"),
        "up_token_id": up_token_id,
        "down_token_id": down_token_id,
        "end_date": market_data.get("endDate"),
    }


# =============================================================================
# Trading Logic
# =============================================================================

def calculate_trade_size(price: float) -> float:
    """Calculate trade size based on price (smaller at higher prices)."""
    if price < 0.15:
        return CONFIG.TRADE_SIZE_BASE * 2.0
    elif price < 0.25:
        return CONFIG.TRADE_SIZE_BASE * 1.5
    elif price < 0.35:
        return CONFIG.TRADE_SIZE_BASE
    else:
        return CONFIG.TRADE_SIZE_BASE * 0.5


def get_dynamic_thresholds(time_to_resolution: float) -> tuple:
    """Get aggressive and standard thresholds based on time to resolution.

    Returns: (aggressive_threshold, standard_threshold)
    """
    if not CONFIG.DYNAMIC_THRESHOLDS:
        return (CONFIG.AGGRESSIVE_BUY_THRESHOLD, CONFIG.STANDARD_BUY_THRESHOLD)

    for (min_sec, max_sec), (agg, std) in CONFIG.THRESHOLDS_BY_TIME.items():
        if min_sec <= time_to_resolution < max_sec:
            return (agg, std)

    # Default to base thresholds if outside defined ranges
    return (CONFIG.AGGRESSIVE_BUY_THRESHOLD, CONFIG.STANDARD_BUY_THRESHOLD)


async def process_market_signal(market: MarketPrice):
    """Process a market tick and generate/execute signals."""
    t_detection_start = time.perf_counter()

    # DEBUG: Log every market check
    asset = market.slug.split('-')[0].upper()[:3]
    logger.debug(f"SIGNAL CHECK {asset}: Up=${market.up_price:.3f} Down=${market.down_price:.3f} Pair=${market.pair_cost:.3f}")

    # Calculate time to resolution
    time_to_resolution = 900  # Default to full 15 min if unknown
    if market.resolution_time:
        time_to_resolution = (market.resolution_time - datetime.utcnow()).total_seconds()

    # SAFETY CHECK 1: Resolution time - don't buy too close to resolution
    if time_to_resolution < CONFIG.MIN_TIME_TO_RESOLUTION_SEC:
        logger.info(f"BLOCKED {asset}: Too close to resolution ({time_to_resolution:.0f}s)")
        return  # Too close to resolution, skip

    # Get dynamic thresholds based on time window
    aggressive_threshold, standard_threshold = get_dynamic_thresholds(time_to_resolution)

    # Log when thresholds change from default
    if aggressive_threshold != CONFIG.AGGRESSIVE_BUY_THRESHOLD:
        logger.debug(f"{asset}: Using dynamic thresholds for {time_to_resolution:.0f}s window: agg<${aggressive_threshold:.2f}, std<${standard_threshold:.2f}")

    # SAFETY CHECK 2: Maximum pair cost - never buy if implied pair cost > MAX_PAIR_COST
    # This prevents losses even on aggressive buys
    if market.pair_cost > CONFIG.MAX_PAIR_COST:
        logger.info(f"BLOCKED {asset}: Pair cost {market.pair_cost:.3f} > MAX {CONFIG.MAX_PAIR_COST}")
        return  # Market pair cost too high, no arbitrage opportunity

    # Get existing position for hedge ratio and pair cost checks
    position = STATE.account.positions.get(market.condition_id)

    # Check Up side - using dynamic thresholds
    up_signal_type = None
    if market.up_price < aggressive_threshold:
        # SAFETY CHECK 3: Even aggressive buys must have reasonable pair cost
        projected_pair = market.up_price + market.down_price
        if projected_pair < CONFIG.MAX_PAIR_COST:
            # SAFETY CHECK 4: Check hedge ratio - don't go too unbalanced
            if position and position.up_shares > 0 and position.down_shares == 0:
                # Already holding Up only - need to buy Down first or skip
                pass  # Skip this Up buy
            else:
                up_signal_type = "AGGRESSIVE"
    elif market.up_price < standard_threshold:
        # Check if pair cost would be acceptable
        current_pair = position.pair_cost if position and position.pair_cost > 0 else market.pair_cost
        if current_pair < CONFIG.TARGET_PAIR_COST:
            # Check hedge ratio
            if position:
                hedge = position.hedge_ratio
                if hedge == 0 or hedge >= CONFIG.MIN_HEDGE_RATIO:
                    up_signal_type = "STANDARD"
            else:
                up_signal_type = "STANDARD"

    # Check Down side - using dynamic thresholds
    down_signal_type = None
    if market.down_price < aggressive_threshold:
        # SAFETY CHECK 3: Even aggressive buys must have reasonable pair cost
        projected_pair = market.up_price + market.down_price
        if projected_pair < CONFIG.MAX_PAIR_COST:
            # SAFETY CHECK 4: Check hedge ratio - don't go too unbalanced
            if position and position.down_shares > 0 and position.up_shares == 0:
                # Already holding Down only - need to buy Up first or skip
                pass  # Skip this Down buy
            else:
                down_signal_type = "AGGRESSIVE"
    elif market.down_price < standard_threshold:
        current_pair = position.pair_cost if position and position.pair_cost > 0 else market.pair_cost
        if current_pair < CONFIG.TARGET_PAIR_COST:
            # Check hedge ratio
            if position:
                hedge = position.hedge_ratio
                if hedge == 0 or hedge >= CONFIG.MIN_HEDGE_RATIO:
                    down_signal_type = "STANDARD"
            else:
                down_signal_type = "STANDARD"

    detection_ms = (time.perf_counter() - t_detection_start) * 1000

    # DEBUG: Log signal types
    if up_signal_type or down_signal_type:
        logger.info(f"SIGNAL {asset}: up_type={up_signal_type} down_type={down_signal_type} up_token={bool(market.up_token_id)} down_token={bool(market.down_token_id)}")

    # Process Up signal
    if up_signal_type and market.up_token_id:
        await execute_signal(
            market=market,
            outcome="Up",
            token_id=market.up_token_id,
            price=market.up_price,
            signal_type=up_signal_type,
            detection_ms=detection_ms,
        )

    # Process Down signal
    if down_signal_type and market.down_token_id:
        await execute_signal(
            market=market,
            outcome="Down",
            token_id=market.down_token_id,
            price=market.down_price,
            signal_type=down_signal_type,
            detection_ms=detection_ms,
        )


async def execute_signal(
    market: MarketPrice,
    outcome: str,
    token_id: str,
    price: float,
    signal_type: str,
    detection_ms: float,
):
    """Execute a trading signal with full order book analysis for realistic simulation."""
    t_e2e_start = time.perf_counter()

    # =========================================================================
    # STEP 1: Fetch real-time order book for accurate fill estimation
    # =========================================================================
    order_book = await fetch_order_book_detailed(token_id)
    liquidity_before = order_book.ask_depth_10 if order_book else 0

    # Validation phase
    t_validation_start = time.perf_counter()

    # Check position limits
    position = STATE.account.positions.get(market.condition_id)
    if position and position.total_cost >= CONFIG.MAX_POSITION_PER_MARKET:
        signal = Signal(
            timestamp=datetime.utcnow(),
            condition_id=market.condition_id,
            market_title=market.title,
            outcome=outcome,
            token_id=token_id,
            price=price,
            signal_type=signal_type,
            pair_cost=market.pair_cost,
            action="SKIP",
            reason="Position limit reached",
            latency_ms=detection_ms,
        )
        STATE.signals.append(signal)
        return

    # Check total exposure
    if STATE.account.open_exposure >= CONFIG.MAX_TOTAL_EXPOSURE:
        signal = Signal(
            timestamp=datetime.utcnow(),
            condition_id=market.condition_id,
            market_title=market.title,
            outcome=outcome,
            token_id=token_id,
            price=price,
            signal_type=signal_type,
            pair_cost=market.pair_cost,
            action="SKIP",
            reason="Total exposure limit reached",
            latency_ms=detection_ms,
        )
        STATE.signals.append(signal)
        return

    validation_ms = (time.perf_counter() - t_validation_start) * 1000
    LATENCY.record_validation(validation_ms)

    # Order building phase
    t_build_start = time.perf_counter()

    trade_size_usd = calculate_trade_size(price)
    shares = trade_size_usd / price

    # =========================================================================
    # STEP 2: Calculate realistic fill estimate based on order book
    # =========================================================================
    fill_estimate = estimate_fill(order_book, "BUY", trade_size_usd, price)

    # Use estimated fill price instead of signal price for paper trading accuracy
    realistic_fill_price = fill_estimate.estimated_fill_price if fill_estimate.would_fill else price

    order_build_ms = (time.perf_counter() - t_build_start) * 1000
    LATENCY.record_order_build(order_build_ms)

    # Create signal record with fill estimate info
    fill_info = f"Fill: ${realistic_fill_price:.4f} ({fill_estimate.slippage_pct:.1f}% slip, {fill_estimate.fill_probability:.0%} prob)"
    signal = Signal(
        timestamp=datetime.utcnow(),
        condition_id=market.condition_id,
        market_title=market.title,
        outcome=outcome,
        token_id=token_id,
        price=price,
        signal_type=signal_type,
        pair_cost=market.pair_cost,
        action="BUY",
        reason=f"Price ${price:.3f} under {signal_type.lower()} | {fill_info}",
        latency_ms=detection_ms,
    )
    STATE.signals.append(signal)
    LATENCY.record_detection(detection_ms)

    # Submit order - routes to simulation or live based on CONFIG.LIVE_MODE
    result = await submit_order(
        token_id=token_id,
        side="BUY",
        price=realistic_fill_price,  # Use realistic price
        size=shares,
        order_type="FOK",
        fill_estimate=fill_estimate,  # Pass for simulation accuracy
    )

    e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
    LATENCY.record_e2e(e2e_ms)
    LATENCY.record_submission(result["submission_ms"], result["success"], result.get("is_api_error", False))

    # Record order attempt
    order_attempt = OrderAttempt(
        timestamp=datetime.utcnow(),
        condition_id=market.condition_id,
        market_title=market.title,
        token_id=token_id,
        outcome=outcome,
        side="BUY",
        price=realistic_fill_price,  # Use realistic price
        size=shares,
        order_type="FOK",
        detection_ms=detection_ms,
        validation_ms=validation_ms,
        order_build_ms=order_build_ms,
        submission_ms=result["submission_ms"],
        e2e_ms=e2e_ms,
        success=result["success"],
        clob_order_id=result.get("clob_order_id"),
        error_message=result.get("error_message"),
    )
    STATE.account.order_attempts.append(order_attempt)

    # =========================================================================
    # STEP 3: Create detailed trade log for post-analysis
    # =========================================================================
    is_simulation = result.get("is_simulation", not CONFIG.LIVE_MODE)
    order_status = "SIMULATED_FILL" if is_simulation and result["success"] else \
                   "SIMULATED_MISS" if is_simulation and not result["success"] else \
                   "LIVE_FILL" if result["success"] else "LIVE_FAIL"

    trade_log = DetailedTradeLog(
        timestamp=datetime.utcnow().isoformat(),
        market_slug=market.slug,
        condition_id=market.condition_id,
        outcome=outcome,
        signal_type=signal_type,
        signal_price=price,
        pair_cost=market.pair_cost,
        order_book=order_book.to_dict() if order_book else {},
        fill_estimate=fill_estimate.to_dict(),
        order_size_usd=trade_size_usd,
        order_size_shares=shares,
        order_type="FOK",
        detection_ms=detection_ms,
        submission_ms=result["submission_ms"],
        e2e_ms=e2e_ms,
        order_status=order_status,
        error_message=result.get("error_message"),
        liquidity_before=liquidity_before,
        liquidity_after=0,  # Will be updated on next check
        was_front_run=False,  # Will be determined later
    )
    log_trade(trade_log)

    # =========================================================================
    # STEP 4: Intense audit logging
    # =========================================================================
    mode_tag = "LIVE" if CONFIG.LIVE_MODE else "SIM"
    logger.info(
        f"[{mode_tag}] TRADE #{len(TRADE_LOGS):04d} | "
        f"{market.slug} {outcome} | "
        f"Signal ${price:.4f} -> Fill ${realistic_fill_price:.4f} | "
        f"Slip {fill_estimate.slippage_pct:.2f}% | "
        f"Status: {order_status} | "
        f"E2E {e2e_ms:.0f}ms"
    )

    # Update paper trading position (simulate fill for paper trading)
    if market.condition_id not in STATE.account.positions:
        STATE.account.positions[market.condition_id] = Position(
            condition_id=market.condition_id,
            market_title=market.title,
            opened_at=datetime.utcnow(),
        )

    position = STATE.account.positions[market.condition_id]
    if outcome == "Up":
        position.up_shares += shares
        position.up_cost += trade_size_usd
    else:
        position.down_shares += shares
        position.down_cost += trade_size_usd
    position.trades_count += 1

    STATE.account.balance -= trade_size_usd

    logger.info(
        f"ORDER: {outcome} @ ${price:.3f} | "
        f"E2E: {e2e_ms:.1f}ms | "
        f"Submit: {result['submission_ms']:.1f}ms | "
        f"{'OK' if result['success'] else result.get('error_message', 'Failed')[:30]}"
    )


# =============================================================================
# Price Loop
# =============================================================================

async def check_competition_tracking():
    """Check recent trades for front-running (liquidity disappeared)."""
    if not TRADE_LOGS:
        return

    # Check last 10 trades that haven't been analyzed yet
    for trade in TRADE_LOGS[-10:]:
        if trade.liquidity_after > 0:
            continue  # Already analyzed

        token_id = None
        # Get token ID from the trade
        for m in STATE.markets.values():
            if m.condition_id == trade.condition_id:
                token_id = m.up_token_id if trade.outcome == "Up" else m.down_token_id
                break

        if not token_id:
            continue

        # Fetch current order book
        current_book = await fetch_order_book_detailed(token_id)
        if not current_book:
            continue

        current_liquidity = current_book.ask_depth_10

        # Update trade log
        trade.liquidity_after = current_liquidity

        # Determine if front-run (liquidity dropped by > 50%)
        if trade.liquidity_before > 0:
            liquidity_ratio = current_liquidity / trade.liquidity_before
            if liquidity_ratio < 0.5:
                trade.was_front_run = True
                logger.warning(
                    f"FRONT-RUN DETECTED: {trade.market_slug} {trade.outcome} - "
                    f"Liquidity dropped {trade.liquidity_before:.0f} → {current_liquidity:.0f}"
                )


async def price_loop():
    """Main loop to fetch real market data and process signals."""
    logger.info("Starting price loop with LIVE market data...")

    while STATE.is_running:
        try:
            t_loop_start = time.perf_counter()

            # Fetch active markets
            markets = await fetch_crypto_markets()

            for market_data in markets:
                parsed = parse_market_with_tokens(market_data)
                if not parsed or not parsed.get("up_token_id") or not parsed.get("down_token_id"):
                    continue

                # PRIMARY: Use outcomePrices from Gamma API (this is the REAL market price!)
                outcome_prices = market_data.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except:
                        outcome_prices = []

                if outcome_prices and len(outcome_prices) >= 2:
                    # outcomePrices[0] = Up/Yes, outcomePrices[1] = Down/No
                    up_price = float(outcome_prices[0])
                    down_price = float(outcome_prices[1])
                    logger.debug(f"Using outcomePrices: Up ${up_price:.3f} Down ${down_price:.3f}")
                else:
                    # FALLBACK: Use CLOB order book if outcomePrices not available
                    up_book = await fetch_order_book(parsed["up_token_id"])
                    down_book = await fetch_order_book(parsed["down_token_id"])

                    if not up_book or not down_book:
                        continue

                    up_asks = up_book.get("asks", [])
                    down_asks = down_book.get("asks", [])

                    if not up_asks or not down_asks:
                        continue

                    if isinstance(up_asks[0], dict):
                        up_price = float(up_asks[0]["price"])
                    else:
                        up_price = float(up_asks[0][0])

                    if isinstance(down_asks[0], dict):
                        down_price = float(down_asks[0]["price"])
                    else:
                        down_price = float(down_asks[0][0])

                    logger.debug(f"Using CLOB order book: Up ${up_price:.3f} Down ${down_price:.3f}")

                # Liquidity from Gamma API or estimate
                up_liquidity = float(market_data.get("volume", 0)) / 2  # Rough estimate
                down_liquidity = float(market_data.get("volume", 0)) / 2

                # Parse resolution time
                # NOTE: Timezone-safe implementation:
                # - end_date from API is ISO 8601 UTC (with Z suffix)
                # - We parse as UTC and make naive (no tzinfo)
                # - Compare with datetime.utcnow() which is also naive UTC
                # - This works regardless of VPS local timezone
                resolution_time = None
                end_date_str = parsed.get("end_date")
                if end_date_str:
                    try:
                        resolution_time = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        # Make naive for comparison with utcnow()
                        resolution_time = resolution_time.replace(tzinfo=None)
                    except (ValueError, TypeError):
                        pass

                # Create market price object
                market_price = MarketPrice(
                    condition_id=parsed["condition_id"],
                    slug=parsed["slug"],
                    title=parsed["title"],
                    up_token_id=parsed["up_token_id"],
                    down_token_id=parsed["down_token_id"],
                    up_price=up_price,
                    down_price=down_price,
                    up_liquidity=up_liquidity,
                    down_liquidity=down_liquidity,
                    pair_cost=up_price + down_price,
                    timestamp=datetime.utcnow(),
                    resolution_time=resolution_time,
                )

                STATE.markets[parsed["condition_id"]] = market_price

                # Process for signals
                await process_market_signal(market_price)

            STATE.last_price_update = datetime.utcnow()

            # Check for front-running on recent trades
            await check_competition_tracking()

            loop_time_ms = (time.perf_counter() - t_loop_start) * 1000
            logger.debug(f"Price loop: {len(STATE.markets)} markets, {loop_time_ms:.0f}ms, {len(TRADE_LOGS)} trades logged")

            # Broadcast to clients
            await broadcast_state()

        except Exception as e:
            import traceback
            logger.error(f"Error in price loop: {e}")
            logger.error(traceback.format_exc())

        await asyncio.sleep(CONFIG.PRICE_CHECK_INTERVAL_SEC)


# =============================================================================
# WebSocket
# =============================================================================

async def broadcast_state():
    """Broadcast state to all connected WebSocket clients."""
    if not STATE.connected_clients:
        return

    message = {
        "type": "state_update",
        "timestamp": datetime.utcnow().isoformat(),
        "account": STATE.account.to_dict(),
        "markets": {k: v.to_dict() for k, v in STATE.markets.items()},
        "config": {
            "AGGRESSIVE_BUY_THRESHOLD": CONFIG.AGGRESSIVE_BUY_THRESHOLD,
            "STANDARD_BUY_THRESHOLD": CONFIG.STANDARD_BUY_THRESHOLD,
            "TARGET_PAIR_COST": CONFIG.TARGET_PAIR_COST,
            "MAX_PAIR_COST": CONFIG.MAX_PAIR_COST,
        },
        "latency": LATENCY.to_dict(),
        "recent_signals": [s.to_dict() for s in STATE.signals[-20:]],
        "recent_orders": [o.to_dict() for o in STATE.account.order_attempts[-20:]],
    }

    disconnected = set()
    for ws in STATE.connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.add(ws)

    for ws in disconnected:
        STATE.connected_clients.discard(ws)


# =============================================================================
# API Routes
# =============================================================================

@app.get("/arbitrage")
@app.get("/arbitrage/")
async def index():
    """Serve the arbitrage dashboard."""
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/arbitrage/api/status")
async def get_status():
    """Get current status."""
    return {
        "running": STATE.is_running,
        "account": STATE.account.to_dict(),
        "markets_count": len(STATE.markets),
        "signals_count": len(STATE.signals),
        "latency": LATENCY.to_dict(),
        "last_update": STATE.last_price_update.isoformat() if STATE.last_price_update else None,
    }


@app.get("/arbitrage/api/markets")
async def get_markets():
    """Get all current market prices."""
    return {k: v.to_dict() for k, v in STATE.markets.items()}


@app.get("/arbitrage/api/positions")
async def get_positions():
    """Get all positions."""
    return {k: v.to_dict() for k, v in STATE.account.positions.items()}


@app.get("/arbitrage/api/signals")
async def get_signals():
    """Get recent signals."""
    return [s.to_dict() for s in STATE.signals[-100:]]


@app.get("/arbitrage/api/orders")
async def get_orders():
    """Get recent order attempts."""
    return [o.to_dict() for o in STATE.account.order_attempts[-100:]]


@app.get("/arbitrage/api/latency")
async def get_latency():
    """Get latency metrics."""
    return LATENCY.to_dict()


@app.get("/arbitrage/api/trade-logs")
async def get_trade_logs(limit: int = 100):
    """Get detailed trade logs for analysis."""
    return [t.to_dict() for t in TRADE_LOGS[-limit:]]


def calculate_pl_stats() -> dict:
    """Calculate P/L statistics from trade logs and positions.

    PAIR_ACCUMULATION strategy P/L:
    - Each complete pair (1 Up + 1 Down share) pays out $1.00 at resolution
    - Profit = $1.00 - (avg_up_price + avg_down_price) per pair
    - Unhedged shares are at-risk (either $0 or $1 payout)
    """
    if not STATE.account.positions:
        return {
            "total_pairs_completed": 0,
            "total_pair_cost": 0,
            "total_pair_payout": 0,
            "realized_profit": 0,
            "unrealized_exposure": 0,
            "positions": [],
        }

    positions_pl = []
    total_pairs = 0
    total_pair_cost = 0
    total_unhedged_cost = 0

    for cid, pos in STATE.account.positions.items():
        # Pairs = minimum of up and down shares
        pairs = min(pos.up_shares, pos.down_shares)
        total_pairs += pairs

        # Cost of the paired shares
        if pos.up_shares > 0 and pos.down_shares > 0:
            pair_avg_cost = pos.up_avg_price + pos.down_avg_price
            pair_total_cost = pairs * pair_avg_cost
            pair_profit = pairs * 1.0 - pair_total_cost  # $1 payout per pair
        else:
            pair_avg_cost = 0
            pair_total_cost = 0
            pair_profit = 0

        total_pair_cost += pair_total_cost

        # Unhedged exposure (shares at risk)
        unhedged_up = max(0, pos.up_shares - pos.down_shares)
        unhedged_down = max(0, pos.down_shares - pos.up_shares)
        unhedged_cost = unhedged_up * pos.up_avg_price + unhedged_down * pos.down_avg_price
        total_unhedged_cost += unhedged_cost

        positions_pl.append({
            "market": pos.market_title[:40],
            "pairs": round(pairs, 2),
            "pair_cost": round(pair_avg_cost, 4),
            "pair_profit": round(pair_profit, 2),
            "unhedged_up": round(unhedged_up, 2),
            "unhedged_down": round(unhedged_down, 2),
            "unhedged_cost": round(unhedged_cost, 2),
        })

    # Total P/L assuming pairs resolve (guaranteed profit)
    guaranteed_profit = total_pairs * 1.0 - total_pair_cost

    return {
        "total_pairs_completed": round(total_pairs, 2),
        "total_pair_cost": round(total_pair_cost, 2),
        "total_pair_payout": round(total_pairs * 1.0, 2),
        "guaranteed_profit": round(guaranteed_profit, 2),
        "unrealized_exposure": round(total_unhedged_cost, 2),
        "profit_pct": round((guaranteed_profit / total_pair_cost * 100) if total_pair_cost > 0 else 0, 2),
        "positions": positions_pl,
    }


@app.get("/arbitrage/api/pl")
async def get_pl():
    """Get P/L statistics."""
    return calculate_pl_stats()


@app.get("/arbitrage/api/simulation-stats")
async def get_simulation_stats():
    """Get comprehensive simulation statistics for analysis."""
    if not TRADE_LOGS:
        return {"error": "No trades logged yet"}

    # Calculate statistics
    total_trades = len(TRADE_LOGS)
    total_volume_usd = sum(t.order_size_usd for t in TRADE_LOGS)

    # Fill statistics
    successful_fills = [t for t in TRADE_LOGS if t.order_status in ("SIMULATED_FILL", "LIVE_FILL")]
    fill_rate = len(successful_fills) / total_trades if total_trades > 0 else 0

    # Slippage statistics
    slippages = [t.fill_estimate.get("slippage_pct", 0) for t in TRADE_LOGS]
    avg_slippage = sum(slippages) / len(slippages) if slippages else 0
    max_slippage = max(slippages) if slippages else 0

    # Latency statistics
    detection_times = [t.detection_ms for t in TRADE_LOGS]
    e2e_times = [t.e2e_ms for t in TRADE_LOGS]
    avg_detection = sum(detection_times) / len(detection_times) if detection_times else 0
    avg_e2e = sum(e2e_times) / len(e2e_times) if e2e_times else 0

    # Price statistics
    signal_prices = [t.signal_price for t in TRADE_LOGS]
    fill_prices = [t.fill_estimate.get("estimated_fill_price", 0) for t in TRADE_LOGS]

    # Profitability estimate (assuming all trades resolve correctly)
    # For pairs, profit = payout ($1) - cost (sum of prices paid)
    total_signal_cost = sum(signal_prices)
    total_fill_cost = sum(fill_prices)

    # By outcome
    up_trades = [t for t in TRADE_LOGS if t.outcome == "Up"]
    down_trades = [t for t in TRADE_LOGS if t.outcome == "Down"]

    # By signal type
    aggressive = [t for t in TRADE_LOGS if t.signal_type == "AGGRESSIVE"]
    standard = [t for t in TRADE_LOGS if t.signal_type == "STANDARD"]

    # By order status
    sim_fills = len([t for t in TRADE_LOGS if t.order_status == "SIMULATED_FILL"])
    sim_misses = len([t for t in TRADE_LOGS if t.order_status == "SIMULATED_MISS"])
    live_fills = len([t for t in TRADE_LOGS if t.order_status == "LIVE_FILL"])
    live_fails = len([t for t in TRADE_LOGS if t.order_status == "LIVE_FAIL"])

    # Liquidity analysis
    avg_liquidity = sum(t.liquidity_before for t in TRADE_LOGS) / total_trades if total_trades > 0 else 0

    # Competition analysis
    front_run_count = sum(1 for t in TRADE_LOGS if t.was_front_run)

    # P/L stats
    pl_stats = calculate_pl_stats()

    return {
        "mode": "LIVE" if CONFIG.LIVE_MODE else "SIMULATION",
        "first_trade": TRADE_LOGS[0].timestamp if TRADE_LOGS else None,
        "last_trade": TRADE_LOGS[-1].timestamp if TRADE_LOGS else None,
        "summary": {
            "total_trades": total_trades,
            "total_volume_usd": round(total_volume_usd, 2),
            "fill_rate": round(fill_rate, 4),
            "avg_slippage_pct": round(avg_slippage, 4),
            "max_slippage_pct": round(max_slippage, 4),
        },
        "order_status": {
            "simulated_fills": sim_fills,
            "simulated_misses": sim_misses,
            "live_fills": live_fills,
            "live_fails": live_fails,
        },
        "latency": {
            "avg_detection_ms": round(avg_detection, 2),
            "avg_e2e_ms": round(avg_e2e, 2),
            "p95_e2e_ms": round(sorted(e2e_times)[int(len(e2e_times) * 0.95)] if len(e2e_times) >= 20 else max(e2e_times, default=0), 2),
        },
        "pricing": {
            "avg_signal_price": round(sum(signal_prices) / len(signal_prices) if signal_prices else 0, 4),
            "avg_fill_price": round(sum(fill_prices) / len(fill_prices) if fill_prices else 0, 4),
            "total_signal_cost": round(total_signal_cost, 4),
            "total_estimated_fill_cost": round(total_fill_cost, 4),
            "slippage_cost": round(total_fill_cost - total_signal_cost, 4),
        },
        "profitability": pl_stats,
        "breakdown": {
            "up_trades": len(up_trades),
            "down_trades": len(down_trades),
            "aggressive_signals": len(aggressive),
            "standard_signals": len(standard),
        },
        "liquidity": {
            "avg_available_shares": round(avg_liquidity, 2),
            "front_run_count": front_run_count,
            "front_run_rate": round(front_run_count / total_trades if total_trades > 0 else 0, 4),
        },
        "log_file": str(TRADE_LOG_FILE),
        "log_size_kb": round(TRADE_LOG_FILE.stat().st_size / 1024, 2) if TRADE_LOG_FILE.exists() else 0,
    }


@app.get("/arbitrage/api/export-logs")
async def export_logs():
    """Export all trade logs as JSON for external analysis."""
    return {
        "trades": [t.to_dict() for t in TRADE_LOGS],
        "stats": await get_simulation_stats(),
        "config": {
            "AGGRESSIVE_BUY_THRESHOLD": CONFIG.AGGRESSIVE_BUY_THRESHOLD,
            "STANDARD_BUY_THRESHOLD": CONFIG.STANDARD_BUY_THRESHOLD,
            "TARGET_PAIR_COST": CONFIG.TARGET_PAIR_COST,
            "MAX_PAIR_COST": CONFIG.MAX_PAIR_COST,
            "TRADE_SIZE_BASE": CONFIG.TRADE_SIZE_BASE,
        },
        "exported_at": datetime.utcnow().isoformat(),
    }


@app.post("/arbitrage/api/start")
async def start_trading():
    """Start live testing."""
    if STATE.is_running:
        return {"status": "already_running"}

    STATE.is_running = True
    asyncio.create_task(price_loop())
    return {"status": "started"}


@app.post("/arbitrage/api/stop")
async def stop_trading():
    """Stop live testing."""
    STATE.is_running = False
    return {"status": "stopped"}


@app.post("/arbitrage/api/reset")
async def reset_account():
    """Reset everything."""
    global LATENCY
    STATE.account = PaperAccount(
        balance=CONFIG.STARTING_BALANCE,
        starting_balance=CONFIG.STARTING_BALANCE,
    )
    STATE.signals.clear()
    STATE.markets.clear()
    LATENCY = LatencyMetrics()
    return {"status": "reset"}


@app.post("/arbitrage/api/config")
async def update_config(config: dict):
    """Update configuration."""
    if "AGGRESSIVE_BUY_THRESHOLD" in config:
        CONFIG.AGGRESSIVE_BUY_THRESHOLD = float(config["AGGRESSIVE_BUY_THRESHOLD"])
    if "STANDARD_BUY_THRESHOLD" in config:
        CONFIG.STANDARD_BUY_THRESHOLD = float(config["STANDARD_BUY_THRESHOLD"])
    if "TARGET_PAIR_COST" in config:
        CONFIG.TARGET_PAIR_COST = float(config["TARGET_PAIR_COST"])
    if "MAX_PAIR_COST" in config:
        CONFIG.MAX_PAIR_COST = float(config["MAX_PAIR_COST"])

    return {"status": "updated", "config": {
        "AGGRESSIVE_BUY_THRESHOLD": CONFIG.AGGRESSIVE_BUY_THRESHOLD,
        "STANDARD_BUY_THRESHOLD": CONFIG.STANDARD_BUY_THRESHOLD,
        "TARGET_PAIR_COST": CONFIG.TARGET_PAIR_COST,
        "MAX_PAIR_COST": CONFIG.MAX_PAIR_COST,
    }}


@app.post("/arbitrage/api/test-signal")
async def test_signal():
    """
    Test the signal detection and order submission pipeline.

    Creates a fake market with a price below threshold to trigger a signal,
    then attempts an order submission to verify latency tracking.
    """
    # Create a fake market for testing
    test_market = MarketPrice(
        condition_id="test-signal-verification",
        slug="test-signal",
        title="TEST: Signal Verification",
        up_token_id="test-up-token-12345",
        down_token_id="test-down-token-67890",
        up_price=0.20,  # Below AGGRESSIVE_BUY_THRESHOLD (0.25)
        down_price=0.25,  # Below STANDARD_BUY_THRESHOLD (0.35)
        up_liquidity=1000.0,
        down_liquidity=1000.0,
        pair_cost=0.45,  # Well below TARGET_PAIR_COST (0.95)
        timestamp=datetime.utcnow(),
        resolution_time=datetime.utcnow() + timedelta(minutes=10),  # 10 mins to resolution
    )

    logger.info("=== TEST SIGNAL: Starting signal verification ===")
    logger.info(f"Test market: Up=${test_market.up_price}, Down=${test_market.down_price}, Pair=${test_market.pair_cost}")

    # Process the signal (this will trigger order submission)
    await process_market_signal(test_market)

    # Get latest metrics
    latest_signals = [s for s in STATE.signals if s.condition_id == "test-signal-verification"]
    latest_orders = [o for o in STATE.account.order_attempts if o.condition_id == "test-signal-verification"]

    return {
        "status": "test_complete",
        "signals_generated": len(latest_signals),
        "orders_attempted": len(latest_orders),
        "signals": [s.to_dict() for s in latest_signals],
        "orders": [o.to_dict() for o in latest_orders],
        "latency": LATENCY.to_dict(),
    }


@app.websocket("/arbitrage/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live updates."""
    await websocket.accept()
    STATE.connected_clients.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(STATE.connected_clients)}")

    # Send initial state
    await websocket.send_json({
        "type": "connected",
        "timestamp": datetime.utcnow().isoformat(),
        "account": STATE.account.to_dict(),
        "markets": {k: v.to_dict() for k, v in STATE.markets.items()},
        "config": {
            "AGGRESSIVE_BUY_THRESHOLD": CONFIG.AGGRESSIVE_BUY_THRESHOLD,
            "STANDARD_BUY_THRESHOLD": CONFIG.STANDARD_BUY_THRESHOLD,
            "TARGET_PAIR_COST": CONFIG.TARGET_PAIR_COST,
            "MAX_PAIR_COST": CONFIG.MAX_PAIR_COST,
        },
        "latency": LATENCY.to_dict(),
    })

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        STATE.connected_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(STATE.connected_clients)}")


# Mount static files under /arbitrage/static
app.mount("/arbitrage/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static")


@app.on_event("startup")
async def startup_event():
    """Start price loop if auto-start is enabled."""
    if STATE.is_running:
        logger.info("Auto-starting price loop...")
        asyncio.create_task(price_loop())


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start on launch")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading mode (real orders)")

    args = parser.parse_args()

    if args.live:
        CONFIG.LIVE_MODE = True

    if args.auto_start:
        STATE.is_running = True

    # Clear startup banner
    print("\n" + "=" * 70)
    print(" POLYMARKET ARBITRAGE BOT")
    print("=" * 70)

    if CONFIG.LIVE_MODE:
        print(" MODE: *** LIVE TRADING *** (real orders will be submitted)")
        print(f" API Key: {'Configured' if CONFIG.API_KEY else 'MISSING!'}")
        print(f" Private Key: {'Configured' if CONFIG.PRIVATE_KEY else 'MISSING!'}")
    else:
        print(" MODE: SIMULATION (paper trading with realistic fill estimates)")
        print(" To enable live mode: --live flag or set LIVE_MODE=True in code")

    print(f"\n Server: http://{args.host}:{args.port}/arbitrage/")
    print(f" Trade logs: {TRADE_LOG_FILE}")
    print(f" Thresholds: Aggressive <${CONFIG.AGGRESSIVE_BUY_THRESHOLD:.2f}, Standard <${CONFIG.STANDARD_BUY_THRESHOLD:.2f}")
    print("=" * 70 + "\n")

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
