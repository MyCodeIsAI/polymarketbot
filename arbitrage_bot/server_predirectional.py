#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot - PREDIRECTIONAL VERSION

LIVE REF MONITORING + REAL API CALLS

Strategy:
1. Poll ref account trades every 0.5s (fast mode)
2. Detect bias from first expensive trade (>=0.70)
3. WAIT until bias established, then pile on aggressively
4. Fire real API calls (will fail with 401 but tracks latency)

Based on exhaustive research of 56,584 historical trades:
- 76.3% of windows: First trade IS expensive (instant bias)
- 93%+ accuracy at 0.70, 94.5% at 0.80
- 84.3% post-bias: pure pile-on (no minority trades)
- 96.5% of profit in bias-established windows
- 8.4x more capital deployed when bias is established
"""

import asyncio
import json
import os
import sys
import time
import ssl
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import httpx
import uvicorn


# =============================================================================
# PERSISTENT HTTP CLIENT (connection pooling for ~200ms vs ~600ms latency)
# =============================================================================
_HTTP_CLIENT = None

async def get_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=30,
            keepalive_expiry=30.0,
        )
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            limits=limits,
            http2=True,
        )
    return _HTTP_CLIENT


# Simulation trade log file
TRADE_LOG_FILE = "/root/arbitrage_bot/predirectional_trades.jsonl"

def log_trade_to_file(trade_data: dict):
    """Log trade to JSONL file for comparison with reference."""
    import json
    from datetime import datetime
    trade_data["timestamp"] = datetime.utcnow().isoformat()
    try:
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(json.dumps(trade_data) + "\n")
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/root/predirectional.log") if os.path.exists("/root") else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="Polymarket PREDIRECTIONAL Bot")

# API Endpoints
CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Reference wallet
REF_WALLET = "0x93c22116e4402c9332ee6db578050e688934c072"

# SSL context for urllib
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# Bot startup time (for failsafe)
BOT_START_TIME: float = 0


# =============================================================================
# BIAS STATE MACHINE
# =============================================================================

class BiasState(Enum):
    WAITING = "waiting"              # No expensive trade - DON'T TRADE
    BIAS_UP = "bias_up"              # First expensive was UP (high conf >=0.80)
    BIAS_DOWN = "bias_down"          # First expensive was DOWN (high conf)
    TENTATIVE_UP = "tentative_up"    # First expensive UP (low conf 0.70-0.80)
    TENTATIVE_DOWN = "tentative_down"
    PIVOT = "pivot"                  # Ref changing direction - STOP


@dataclass
class ConditionBias:
    """Bias tracking for a single condition/market."""
    condition_id: str
    state: BiasState = BiasState.WAITING
    first_expensive_outcome: Optional[str] = None
    first_expensive_price: float = 0
    first_expensive_time: float = 0
    second_expensive_confirms: bool = False
    trades_seen: int = 0
    last_update: float = 0
    # Failsafe: track when we first saw this condition
    first_seen_time: float = 0
    window_start_time: float = 0  # Parsed from slug if possible

    @property
    def bias_direction(self) -> Optional[str]:
        if self.state in [BiasState.BIAS_UP, BiasState.TENTATIVE_UP]:
            return "Up"
        elif self.state in [BiasState.BIAS_DOWN, BiasState.TENTATIVE_DOWN]:
            return "Down"
        return None

    @property
    def is_high_confidence(self) -> bool:
        return self.state in [BiasState.BIAS_UP, BiasState.BIAS_DOWN]

    @property
    def should_trade(self) -> bool:
        return self.state not in [BiasState.WAITING, BiasState.PIVOT]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PredirectionalConfig:
    # Ref monitoring
    REF_WALLET: str = REF_WALLET
    REF_POLL_INTERVAL: float = 0.25  # OPTIMIZED: 250ms polling

    # Bias detection thresholds
    EXPENSIVE_HIGH: float = 0.80   # High confidence
    EXPENSIVE_LOW: float = 0.70    # Low confidence
    PIVOT_THRESHOLD: float = 0.60  # Expensive on minority = pivot

    # Strategy
    SKIP_NO_BIAS: bool = True      # Don't trade until bias detected
    HEDGE_ENABLED: bool = True    # DEFINITIVE: Skip minority (rebate farming)

    # FAILSAFE: Only trade windows that opened AFTER bot started
    # When enabled, we won't trade on historical biases that happened before we were running
    REQUIRE_BOT_RUNNING_BEFORE_WINDOW: bool = False  # DISABLED for now (set True to enable)

    # Sizing multipliers - MATCHED TO REF
    BIAS_SIZE_MULT: float = 12.8   # Validated from clean 15-min TAKER data    # Signal trades 11.6x larger than hedges (clean TAKER audit)
    TENTATIVE_SIZE_MULT: float = 1.0
    MID_SIZE_MULT: float = 5.5         # Mid-range trades 4.9x base (clean TAKER audit)
    HEDGE_SIZE_MULT: float = 1.0    # Hedges at base size (clean TAKER audit)

    # Price limits
    DOMINANT_MAX_PRICE: float = 0.90
    MINORITY_MAX_PRICE: float = 0.30
    # Phase 2: Buy opposite expensive when it reaches this price
    OPPOSITE_TRIGGER_PRICE: float = 0.65
    # Phase 3: End-of-window aggressive trading (>10min)
    PHASE3_START_SEC: float = 600.0
    PHASE3_HEDGE_SIZE_MULT: float = 0.66  # Phase 3 hedges 0.66x smaller (validated)   # Hedges SMALLER late, not larger (clean TAKER audit)
    MAX_PAIR_COST: float = 1.02

    # Timing - PRICE-BASED TRIGGERS (from REF analysis of 106,124 trades)
    # REF has a HARD 2s MINIMUM - zero 1-second gaps in 52,707 consecutive trade pairs
    # 77.6% of trades are same-second bursts, 13.9% are exactly 2s apart
    # 59.7% of 2s trades have price OR outcome change, 40.3% are position building
    PRICE_CHANGE_TRIGGER: float = 0.01  # Trade when price changes by this much
    MIN_TRADE_INTERVAL: float = 2.0     # HARD 2s MINIMUM (confirmed from data)
    MIN_TIME_TO_RESOLUTION: int = 5
    MAX_TIME_TO_RESOLUTION: int = 900

    # Sizing - MATCHED TO REF (from analysis)
    # REF: Median individual trade $5.44, P75 $17.90
    # REF: Median burst $13.26, P75 $41.61
    TRADE_SIZE_BASE: float = 1.14   # Validated from clean 15-min TAKER data   # Base = cheap hedge median $1.35 (clean TAKER audit)
    STARTING_BALANCE: float = 200.0

    # Assets
    ENABLED_ASSETS: List[str] = field(default_factory=lambda: ["btc", "sol", "eth", "xrp"])


CONFIG = PredirectionalConfig()


# =============================================================================
# REF BIAS TRACKER
# =============================================================================

class RefBiasTracker:
    """Tracks ref account's trades to detect directional bias in real-time."""

    def __init__(self):
        self.conditions: Dict[str, ConditionBias] = {}
        self.processed_trade_keys: Set[str] = set()
        self.trades_processed = 0
        self.biases_detected = 0
        self.pivots_detected = 0
        self.skipped_historical_biases = 0  # Failsafe counter

    def _trade_key(self, trade: dict) -> str:
        """Create unique key for deduplication."""
        return f"{trade.get('transactionHash', '')}-{trade.get('timestamp', '')}-{trade.get('outcome', '')}-{trade.get('price', '')}"

    def _parse_window_start_from_slug(self, slug: str) -> float:
        """
        Parse window start time from slug.
        Example: btc-up-or-down-15m-jan-27-2025-6-15pm-et-1738016100
        The last number is the window start epoch.
        """
        try:
            parts = slug.split("-")
            epoch = int(parts[-1])
            return float(epoch)
        except:
            return 0

    def process_trade(self, trade: dict) -> Optional[str]:
        """
        Process a ref trade and update bias state.

        KEY FIX: Instead of PIVOT (stop trading), we now FOLLOW the most recent
        expensive trade direction. This matches REF's actual behavior of switching
        directions within windows.

        Returns event: 'BIAS_DETECTED', 'BIAS_UPDATED', 'BIAS_CONFIRMED', or None
        """
        # Deduplicate
        key = self._trade_key(trade)
        if key in self.processed_trade_keys:
            return None
        self.processed_trade_keys.add(key)

        # Only process BUY trades for 15min crypto
        if trade.get("_our_side", trade.get("side")) != "BUY":
            return None

        slug = trade.get("slug", "")
        if "-15m-" not in slug:
            return None

        condition_id = trade.get("conditionId", "")
        outcome = trade.get("outcome", "")
        price = float(trade.get("price", 0))
        timestamp = trade.get("timestamp", time.time())

        if not condition_id or not outcome:
            return None

        # Initialize condition if needed
        if condition_id not in self.conditions:
            window_start = self._parse_window_start_from_slug(slug)
            self.conditions[condition_id] = ConditionBias(
                condition_id=condition_id,
                first_seen_time=time.time(),
                window_start_time=window_start,
            )

        cond = self.conditions[condition_id]
        cond.trades_seen += 1
        cond.last_update = time.time()
        self.trades_processed += 1

        # Only expensive trades affect bias direction
        if price < CONFIG.EXPENSIVE_LOW:
            return None

        # Get current bias direction (if any)
        current_bias_dir = cond.bias_direction
        is_high_conf = price >= CONFIG.EXPENSIVE_HIGH
        new_state = BiasState.BIAS_UP if outcome == "Up" else BiasState.BIAS_DOWN
        if not is_high_conf:
            new_state = BiasState.TENTATIVE_UP if outcome == "Up" else BiasState.TENTATIVE_DOWN

        # First expensive trade - sets initial bias
        if cond.first_expensive_outcome is None:
            cond.first_expensive_outcome = outcome
            cond.first_expensive_price = price
            cond.first_expensive_time = timestamp
            cond.state = new_state
            self.biases_detected += 1

            # Check failsafe: was this window already open before we started?
            is_historical = False
            if CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW and BOT_START_TIME > 0:
                if cond.window_start_time > 0 and cond.window_start_time < BOT_START_TIME:
                    is_historical = True
                    self.skipped_historical_biases += 1
                    logger.warning(f"HISTORICAL BIAS (skipped): {slug} started before bot")

            conf_tag = "HIGH" if is_high_conf else "TENT"
            if is_historical:
                conf_tag = "HISTORICAL-SKIP"

            logger.info(f"BIAS DETECTED ({conf_tag}): {slug} -> {outcome} @ ${price:.2f}")
            return "BIAS_DETECTED"

        # Subsequent expensive trades - follow the direction!
        if current_bias_dir == outcome:
            # Same direction - confirm/strengthen bias
            cond.state = BiasState.BIAS_UP if outcome == "Up" else BiasState.BIAS_DOWN
            cond.second_expensive_confirms = True
            logger.debug(f"BIAS REINFORCED: {slug} -> {outcome} @ ${price:.2f}")
            return "BIAS_CONFIRMED"
        else:
            # Different direction - UPDATE bias to follow (not PIVOT!)
            # This is the key fix: REF switches directions, so we follow
            old_dir = current_bias_dir
            cond.state = new_state
            self.pivots_detected += 1  # Track direction changes for stats
            logger.info(f"BIAS UPDATED: {slug} {old_dir}->{outcome} @ ${price:.2f}")
            return "BIAS_UPDATED"

        return None

    def get_bias(self, condition_id: str) -> ConditionBias:
        if condition_id not in self.conditions:
            self.conditions[condition_id] = ConditionBias(
                condition_id=condition_id,
                first_seen_time=time.time(),
            )
        return self.conditions[condition_id]

    def should_trade(self, condition_id: str) -> bool:
        """Check if we should trade this condition (including failsafe check)."""
        bias = self.get_bias(condition_id)

        # Basic check: must have established bias
        if not bias.should_trade:
            return False

        # Failsafe check: was the bot running before this window opened?
        if CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW and BOT_START_TIME > 0:
            if bias.window_start_time > 0 and bias.window_start_time < BOT_START_TIME:
                # Window started before bot - don't trade historical biases
                return False

        return True

    def get_direction(self, condition_id: str) -> Optional[str]:
        return self.get_bias(condition_id).bias_direction

    def reset_condition(self, condition_id: str):
        if condition_id in self.conditions:
            del self.conditions[condition_id]

    def is_historical_window(self, condition_id: str) -> bool:
        """Check if this window started before the bot started (failsafe check)."""
        if BOT_START_TIME <= 0:
            return False
        bias = self.conditions.get(condition_id)
        if not bias:
            return False
        if bias.window_start_time > 0 and bias.window_start_time < BOT_START_TIME:
            return True
        return False


# Global tracker
REF_TRACKER = RefBiasTracker()


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MarketPrice:
    condition_id: str
    slug: str
    title: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    pair_cost: float
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id[:16] + "...",
            "slug": self.slug,
            "up_price": self.up_price,
            "down_price": self.down_price,
            "pair_cost": self.pair_cost,
        }


@dataclass
class Signal:
    timestamp: datetime
    condition_id: str
    slug: str
    outcome: str
    price: float
    signal_type: str
    bias_state: str
    pair_cost: float
    size_usd: float
    shares: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "slug": self.slug,
            "outcome": self.outcome,
            "price": self.price,
            "signal_type": self.signal_type,
            "bias_state": self.bias_state,
            "size_usd": round(self.size_usd, 2),
            "shares": round(self.shares, 2),
        }


@dataclass
class OrderAttempt:
    timestamp: datetime
    slug: str
    outcome: str
    side: str
    price: float
    size: float
    signal_type: str
    submission_ms: float
    status_code: int
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "slug": self.slug,
            "outcome": self.outcome,
            "price": self.price,
            "size": round(self.size, 2),
            "signal_type": self.signal_type,
            "submission_ms": round(self.submission_ms, 1),
            "status_code": self.status_code,
            "success": self.success,
            "error": self.error_message,
        }


@dataclass
class State:
    balance: float = 200.0
    signals: List[Signal] = field(default_factory=list)
    order_attempts: List[OrderAttempt] = field(default_factory=list)
    last_trade_time: Dict[str, float] = field(default_factory=dict)      # key = "condition_id:outcome"
    last_trade_price: Dict[str, float] = field(default_factory=dict)     # key = "condition_id:outcome" - for price-based triggers
    ref_trades_fetched: int = 0
    is_running: bool = False


STATE = State(balance=CONFIG.STARTING_BALANCE)


# =============================================================================
# REF TRADE FETCHING (Real-time)
# =============================================================================

def fetch_json_sync(url: str, timeout: int = 10) -> Optional[dict]:
    """Synchronous JSON fetch (for ref trades)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.debug(f"Fetch error: {e}")
        return None


async def fetch_recent_ref_trades() -> List[dict]:
    """Fetch recent trades from ref account (both taker and maker)."""
    all_trades = []

    try:
        # Activity endpoint - where ref was TAKER
        activity_url = f"{DATA_API}/activity?user={REF_WALLET}&limit=100"
        activity = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_json_sync(activity_url)
        )
        if activity:
            for t in activity:
                if t.get("type") == "TRADE":
                    t["_our_side"] = t.get("side")
                    all_trades.append(t)

        # Maker endpoint - where ref was MAKER
        maker_url = f"{DATA_API}/trades?maker={REF_WALLET}&limit=100"
        maker_trades = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_json_sync(maker_url)
        )
        if maker_trades:
            for t in maker_trades:
                taker_side = t.get("side")
                if taker_side == "BUY":
                    t["_our_side"] = "SELL"
                elif taker_side == "SELL":
                    t["_our_side"] = "BUY"
                all_trades.append(t)

        STATE.ref_trades_fetched = len(all_trades)
        return all_trades

    except Exception as e:
        logger.error(f"Error fetching ref trades: {e}")
        return []


# =============================================================================
# MARKET FETCHING
# =============================================================================

async def fetch_crypto_markets() -> List[MarketPrice]:
    """Fetch active 15-min crypto markets closest to expiry."""
    markets = []

    series_map = {
        "btc": "btc-up-or-down-15m",
        "eth": "eth-up-or-down-15m",
        "sol": "sol-up-or-down-15m",
        "xrp": "xrp-up-or-down-15m",
    }

    series_slugs = [series_map[a.lower()] for a in CONFIG.ENABLED_ASSETS if a.lower() in series_map]

    async with httpx.AsyncClient(timeout=15.0) as client:
        for series_slug in series_slugs:
            try:
                # Step 1: Get series to find active events
                resp = await client.get(f"{GAMMA_API}/series", params={"slug": series_slug})
                if resp.status_code != 200:
                    continue

                series_data = resp.json()
                if not series_data:
                    continue

                series = series_data[0] if isinstance(series_data, list) else series_data
                events = series.get("events", [])

                # Get active events
                active_events = [e for e in events if e.get("active") and not e.get("closed")]

                def get_end_ts(e):
                    slug = e.get("slug", "")
                    parts = slug.split("-")
                    try:
                        return int(parts[-1]) + 900
                    except:
                        return float("inf")

                now = time.time()
                valid_events = [e for e in active_events if get_end_ts(e) > now]

                if not valid_events:
                    continue

                # Get closest to expiry
                valid_events.sort(key=get_end_ts)
                event = valid_events[0]
                event_slug = event.get("slug", "")

                # Step 2: Fetch full event data (includes markets)
                event_resp = await client.get(f"{GAMMA_API}/events", params={"slug": event_slug})
                if event_resp.status_code != 200:
                    logger.debug(f"Failed to fetch event {event_slug}: {event_resp.status_code}")
                    continue

                event_data = event_resp.json()
                if isinstance(event_data, list):
                    event_data = event_data[0] if event_data else {}

                # Get market details from the full event data
                markets_data = event_data.get("markets", [])
                if not markets_data:
                    logger.debug(f"No markets in event {event_slug}")
                    continue

                market_data = markets_data[0]
                condition_id = market_data.get("conditionId", "")

                # Fetch current prices
                tokens = market_data.get("clobTokenIds", [])
                if len(tokens) < 2:
                    continue

                # Use /markets endpoint for prices (not /book which has different format)
                market_resp = await client.get(f"{CLOB_API}/markets/{condition_id}")
                if market_resp.status_code != 200:
                    logger.debug(f"Failed to fetch market {condition_id[:16]}: {market_resp.status_code}")
                    continue

                market = market_resp.json()

                # Parse prices from tokens array
                tokens_list = market.get("tokens", [])
                up_token = next((t for t in tokens_list if t.get("outcome") == "Up"), None)
                down_token = next((t for t in tokens_list if t.get("outcome") == "Down"), None)

                if not up_token or not down_token:
                    continue

                up_price = float(up_token.get("price", 0))
                down_price = float(down_token.get("price", 0))

                end_ts = get_end_ts(event)
                resolution_time = datetime.utcfromtimestamp(end_ts)

                markets.append(MarketPrice(
                    condition_id=condition_id,
                    slug=event_slug,
                    title=event_data.get("title", ""),
                    up_token_id=up_token.get("token_id", ""),
                    down_token_id=down_token.get("token_id", ""),
                    up_price=up_price,
                    down_price=down_price,
                    pair_cost=up_price + down_price,
                    resolution_time=resolution_time,
                ))

                logger.debug(f"Fetched {event_slug}: cid={condition_id[:16]}... up={up_price:.2f} down={down_price:.2f}")

            except Exception as e:
                logger.error(f"Error fetching {series_slug}: {e}")

    return markets


# =============================================================================
# ORDER SUBMISSION (Real API calls that will fail with 401)
# =============================================================================

async def submit_order_real(
    token_id: str,
    side: str,
    price: float,
    size: float,
    order_type: str = "FOK",
) -> dict:
    """
    Submit REAL order to CLOB API with connection pooling.
    Uses persistent HTTP client for ~200ms latency vs ~600ms.
    Will fail with 401 (no auth) but gives us real latency data.
    """
    t_start = time.perf_counter()

    order_data = {
        "tokenID": token_id,
        "side": side,
        "price": str(price),
        "size": str(size),
        "type": order_type,
    }

    try:
        client = await get_http_client()
        resp = await client.post(
            f"{CLOB_API}/order",
            json=order_data,
            headers={"Content-Type": "application/json"}
        )
        submission_ms = (time.perf_counter() - t_start) * 1000

        return {
            "success": resp.status_code == 200,
            "submission_ms": submission_ms,
            "status_code": resp.status_code,
            "error_message": f"HTTP {resp.status_code}" if resp.status_code != 200 else None,
            "order_type": order_type,
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "submission_ms": (time.perf_counter() - t_start) * 1000,
            "status_code": 0,
            "error_message": "Timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "submission_ms": (time.perf_counter() - t_start) * 1000,
            "status_code": 0,
            "error_message": str(e),
        }


# =============================================================================
# SIGNAL GENERATION & EXECUTION
# =============================================================================

def should_trade_price_trigger(condition_id: str, outcome: str, current_price: float) -> bool:
    """
    Trading trigger based on REF analysis of 52,707 consecutive trades.

    REF pattern:
    - 77.6% same-second bursts
    - 0% 1-second gaps (ZERO!)
    - 13.9% 2-second gaps
    - 40.3% of 2s trades have NO price change (position building/rebate farming)

    Logic: 2s HARD MINIMUM, then trade is allowed (price change NOT required)
    Price change can optionally accelerate/prioritize trading but is not mandatory.
    """
    key = f"{condition_id}:{outcome}"
    now = time.time()

    # HARD 2s MINIMUM - no trading within 2s of last trade on this outcome
    last_time = STATE.last_trade_time.get(key, 0)
    if last_time > 0 and (now - last_time) < CONFIG.MIN_TRADE_INTERVAL:
        return False

    # After 2s, trading is ALLOWED (price change NOT required per REF analysis)
    # Optional: price change detection for logging/prioritization
    last_price = STATE.last_trade_price.get(key)
    if last_price is not None:
        price_change = abs(current_price - last_price)
        if price_change >= CONFIG.PRICE_CHANGE_TRIGGER:
            logger.debug(f"Price change trigger: {price_change:.3f} >= {CONFIG.PRICE_CHANGE_TRIGGER}")

    return True  # Allow trading after 2s minimum regardless of price change


def record_trade_for_trigger(condition_id: str, outcome: str, price: float):
    """Record a trade for price-based trigger tracking."""
    key = f"{condition_id}:{outcome}"
    STATE.last_trade_time[key] = time.time()
    STATE.last_trade_price[key] = price


def calculate_size(price: float, signal_type: str, is_high_conf: bool, is_phase3: bool = False) -> float:
    """Calculate trade size based on PRICE TIER (validated from clean 15-min data).

    Size scaling:
    - Cheap (<$0.30): 1.0x base (hedges)
    - Mid ($0.30-0.70): 5.5x base
    - Expensive (>=$0.70): 12.8x base (signals)

    Phase 3 hedges are 0.66x smaller (validated).
    """
    base = CONFIG.TRADE_SIZE_BASE

    # Price-tier based sizing (validated Pattern 3)
    if price >= CONFIG.EXPENSIVE_LOW:  # >=$0.70 = expensive
        if signal_type == "BIAS_DOMINANT" and not is_high_conf:
            # Tentative signal - lower confidence
            mult = CONFIG.TENTATIVE_SIZE_MULT
        else:
            # High confidence expensive trade
            mult = CONFIG.BIAS_SIZE_MULT
    elif price < CONFIG.MINORITY_MAX_PRICE:  # <$0.30 = cheap hedge
        if is_phase3:
            # Phase 3 hedges are smaller (validated Pattern 5)
            mult = CONFIG.HEDGE_SIZE_MULT * CONFIG.PHASE3_HEDGE_SIZE_MULT
        else:
            mult = CONFIG.HEDGE_SIZE_MULT
    else:  # $0.30-$0.70 = mid-range
        mult = CONFIG.MID_SIZE_MULT

    return base * mult


async def process_market(market: MarketPrice):
    """Process a market and generate signals based on ref's bias.

    Uses PRICE-BASED TRIGGERS instead of time-based cooldown.
    Each outcome (Up/Down) is tracked independently.
    Trading happens when price changes by >= $0.01 from last trade.
    """
    condition_id = market.condition_id
    asset = market.slug.split("-")[0].upper()

    # Check resolution time
    if market.resolution_time:
        time_to_res = (market.resolution_time - datetime.utcnow()).total_seconds()
        if time_to_res < CONFIG.MIN_TIME_TO_RESOLUTION:
            return
        if time_to_res > CONFIG.MAX_TIME_TO_RESOLUTION:
            return

    # Get bias state
    bias = REF_TRACKER.get_bias(condition_id)

    # WAITING - don't trade
    if bias.state == BiasState.WAITING:
        if CONFIG.SKIP_NO_BIAS:
            logger.debug(f"{asset}: WAITING - no bias yet")
            return

    # PIVOT state is no longer used - we follow direction changes instead
    # (Keeping for backwards compatibility, but state should never be PIVOT now)
    if bias.state == BiasState.PIVOT:
        logger.debug(f"{asset}: Legacy PIVOT state detected, continuing anyway")
        # Don't return - allow trading to continue

    # FAILSAFE: Check if this is a historical window (started before bot)
    if REF_TRACKER.is_historical_window(condition_id):
        if CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW:
            logger.debug(f"{asset}: HISTORICAL - window started before bot, skipping")
            return
        # If failsafe disabled, log but continue
        logger.debug(f"{asset}: HISTORICAL (failsafe disabled) - would skip if enabled")

    # Determine signals based on bias direction
    bias_dir = bias.bias_direction
    is_high_conf = bias.is_high_confidence

    if not bias_dir:
        return

    # PHASE 3: Calculate time into window
    time_into_window = 0
    is_phase3 = False
    if bias.window_start_time > 0:
        time_into_window = time.time() - bias.window_start_time
        is_phase3 = time_into_window >= CONFIG.PHASE3_START_SEC
        if is_phase3:
            logger.debug(f"{asset}: PHASE 3 active ({time_into_window:.0f}s into window)")

    # Get dominant and minority prices
    if bias_dir == "Up":
        dom_price, dom_token = market.up_price, market.up_token_id
        min_price, min_token = market.down_price, market.down_token_id
        dom_outcome, min_outcome = "Up", "Down"
    else:
        dom_price, dom_token = market.down_price, market.down_token_id
        min_price, min_token = market.up_price, market.up_token_id
        dom_outcome, min_outcome = "Down", "Up"

    # DOMINANT SIDE - aggressive buying (with price-based trigger)
    if dom_price <= CONFIG.DOMINANT_MAX_PRICE and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for dominant outcome
        if not should_trade_price_trigger(condition_id, dom_outcome, dom_price):
            logger.debug(f"{asset}: SKIP {dom_outcome} - price unchanged from ${STATE.last_trade_price.get(f'{condition_id}:{dom_outcome}', 0):.3f}")
        else:
            size_usd = calculate_size(dom_price, "BIAS_DOMINANT", is_high_conf)
            shares = size_usd / dom_price if dom_price > 0 else 0

            signal = Signal(
                timestamp=datetime.utcnow(),
                condition_id=condition_id,
                slug=market.slug,
                outcome=dom_outcome,
                price=dom_price,
                signal_type="BIAS_DOMINANT",
                bias_state=bias.state.value,
                pair_cost=market.pair_cost,
                size_usd=size_usd,
                shares=shares,
            )
            STATE.signals.append(signal)

            # Submit real order (will fail with 401)
            conf_tag = "HIGH" if is_high_conf else "TENT"
            logger.info(f"SIGNAL {asset}: {dom_outcome} @ ${dom_price:.3f} | BIAS_DOMINANT ({conf_tag}) | ${size_usd:.2f}")

            result = await submit_order_real(dom_token, "BUY", dom_price, shares)

            # Log trade for comparison with reference
            log_trade_to_file({
                "market_slug": market.slug,
                "condition_id": condition_id,
                "outcome": dom_outcome,
                "signal_type": "BIAS_DOMINANT",
                "signal_price": dom_price,
                "pair_cost": market.pair_cost,
                "order_size_usd": size_usd,
                "order_size_shares": shares,
                "order_type": "FOK",
                "submission_ms": result.get("submission_ms", 0),
                "status_code": result.get("status_code", 0),
                "error_message": result.get("error_message"),
                "bias_direction": bias.bias_direction,
                "bias_confidence": "HIGH" if is_high_conf else "TENTATIVE",
            })

            order = OrderAttempt(
                timestamp=datetime.utcnow(),
                slug=market.slug,
                outcome=dom_outcome,
                side="BUY",
                price=dom_price,
                size=shares,
                signal_type="BIAS_DOMINANT",
                submission_ms=result["submission_ms"],
                status_code=result["status_code"],
                success=result["success"],
                error_message=result.get("error_message"),
            )
            STATE.order_attempts.append(order)

            logger.info(f"  ORDER: {result['status_code']} in {result['submission_ms']:.0f}ms | {result.get('error_message', 'OK')}")

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, dom_outcome, dom_price)

    # PHASE 2: OPPOSITE SIDE EXPENSIVE - when minority reaches trigger price
    # This implements the secondary signal: buy opposite when it becomes attractive
    if min_price >= CONFIG.OPPOSITE_TRIGGER_PRICE and min_price <= CONFIG.DOMINANT_MAX_PRICE and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for minority outcome
        if not should_trade_price_trigger(condition_id, min_outcome, min_price):
            logger.debug(f"{asset}: SKIP {min_outcome} SECONDARY - price unchanged")
        else:
            size_usd = calculate_size(min_price, "BIAS_DOMINANT", is_high_conf)  # Same size as dominant
            shares = size_usd / min_price if min_price > 0 else 0

            signal = Signal(
                timestamp=datetime.utcnow(),
                condition_id=condition_id,
                slug=market.slug,
                outcome=min_outcome,
                price=min_price,
                signal_type="BIAS_SECONDARY",
                bias_state=bias.state.value,
                pair_cost=market.pair_cost,
                size_usd=size_usd,
                shares=shares,
            )
            STATE.signals.append(signal)

            logger.info(f"SIGNAL {asset}: {min_outcome} @ ${min_price:.3f} | BIAS_SECONDARY (opp>=0.65) | ${size_usd:.2f}")
            result = await submit_order_real(min_token, "BUY", min_price, shares)
            order = OrderAttempt(
                timestamp=datetime.utcnow(),
                slug=market.slug,
                outcome=min_outcome,
                side="BUY",
                price=min_price,
                size=shares,
                signal_type="BIAS_SECONDARY",
                submission_ms=result["submission_ms"],
                status_code=result["status_code"],
                success=result["success"],
                error_message=result.get("error_message"),
            )
            STATE.order_attempts.append(order)
            logger.info(f"  ORDER: {result['status_code']} in {result['submission_ms']:.0f}ms | {result.get('error_message', 'OK')}")

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, min_outcome, min_price)

    # PHASE 3: Both directions expensive trading (when >10min into window)
    if is_phase3 and min_price >= CONFIG.EXPENSIVE_LOW and min_price <= CONFIG.DOMINANT_MAX_PRICE and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for minority outcome (Phase 3)
        if not should_trade_price_trigger(condition_id, min_outcome, min_price):
            logger.debug(f"{asset}: SKIP {min_outcome} PHASE3 - price unchanged")
        else:
            size_usd = calculate_size(min_price, "BIAS_DOMINANT", is_high_conf)
            shares = size_usd / min_price if min_price > 0 else 0

            signal = Signal(
                timestamp=datetime.utcnow(),
                condition_id=condition_id,
                slug=market.slug,
                outcome=min_outcome,
                price=min_price,
                signal_type="PHASE3_BOTH",
                bias_state=bias.state.value,
                pair_cost=market.pair_cost,
                size_usd=size_usd,
                shares=shares,
            )
            STATE.signals.append(signal)

            logger.info(f"SIGNAL {asset}: {min_outcome} @ ${min_price:.3f} | PHASE3_BOTH ({time_into_window:.0f}s) | ${size_usd:.2f}")
            result = await submit_order_real(min_token, "BUY", min_price, shares)
            order = OrderAttempt(
                timestamp=datetime.utcnow(),
                slug=market.slug,
                outcome=min_outcome,
                side="BUY",
                price=min_price,
                size=shares,
                signal_type="PHASE3_BOTH",
                submission_ms=result["submission_ms"],
                status_code=result["status_code"],
                success=result["success"],
                error_message=result.get("error_message"),
            )
            STATE.order_attempts.append(order)
            logger.info(f"  ORDER: {result['status_code']} in {result['submission_ms']:.0f}ms | {result.get('error_message', 'OK')}")

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, min_outcome, min_price)

    # MINORITY SIDE - optional tiny hedge
    if CONFIG.HEDGE_ENABLED and min_price <= CONFIG.MINORITY_MAX_PRICE and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for minority hedge
        if not should_trade_price_trigger(condition_id, min_outcome, min_price):
            logger.debug(f"{asset}: SKIP {min_outcome} HEDGE - price unchanged")
        else:
            # Size based on price tier and phase (calculate_size handles phase3)
            size_usd = calculate_size(min_price, "BIAS_HEDGE", is_high_conf, is_phase3)
            shares = size_usd / min_price if min_price > 0 else 0

            signal = Signal(
                timestamp=datetime.utcnow(),
                condition_id=condition_id,
                slug=market.slug,
                outcome=min_outcome,
                price=min_price,
                signal_type="BIAS_HEDGE",
                bias_state=bias.state.value,
                pair_cost=market.pair_cost,
                size_usd=size_usd,
                shares=shares,
            )
            STATE.signals.append(signal)

            result = await submit_order_real(min_token, "BUY", min_price, shares)

            order = OrderAttempt(
                timestamp=datetime.utcnow(),
                slug=market.slug,
                outcome=min_outcome,
                side="BUY",
                price=min_price,
                size=shares,
                signal_type="BIAS_HEDGE",
                submission_ms=result["submission_ms"],
                status_code=result["status_code"],
                success=result["success"],
                error_message=result.get("error_message"),
            )
            STATE.order_attempts.append(order)

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, min_outcome, min_price)


# =============================================================================
# MAIN LOOPS
# =============================================================================

async def ref_monitor_loop():
    """Poll ref account for new trades and update bias state."""
    logger.info(f"Starting ref monitor loop (every {CONFIG.REF_POLL_INTERVAL}s)")

    while STATE.is_running:
        try:
            trades = await fetch_recent_ref_trades()

            for trade in trades:
                event = REF_TRACKER.process_trade(trade)
                # Events are logged in process_trade

            logger.debug(f"Ref poll: {len(trades)} trades, {REF_TRACKER.trades_processed} processed, {REF_TRACKER.biases_detected} biases")

        except Exception as e:
            logger.error(f"Ref monitor error: {e}")

        await asyncio.sleep(CONFIG.REF_POLL_INTERVAL)


async def trading_loop():
    """Main trading loop - check markets and generate signals."""
    logger.info("Starting trading loop")

    while STATE.is_running:
        try:
            markets = await fetch_crypto_markets()

            for market in markets:
                await process_market(market)

        except Exception as e:
            logger.error(f"Trading loop error: {e}")

        await asyncio.sleep(1.0)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "name": "PREDIRECTIONAL Bot",
        "ref_wallet": REF_WALLET,
        "is_running": STATE.is_running,
        "poll_interval": CONFIG.REF_POLL_INTERVAL,
        "failsafe_enabled": CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW,
        "bot_start_time": BOT_START_TIME,
    }


@app.get("/api/status")
async def get_status():
    return {
        "is_running": STATE.is_running,
        "balance": STATE.balance,
        "signals_count": len(STATE.signals),
        "orders_count": len(STATE.order_attempts),
        "ref_trades_fetched": STATE.ref_trades_fetched,
        "ref_trades_processed": REF_TRACKER.trades_processed,
        "biases_detected": REF_TRACKER.biases_detected,
        "pivots_detected": REF_TRACKER.pivots_detected,
        "skipped_historical": REF_TRACKER.skipped_historical_biases,
        "poll_interval_sec": CONFIG.REF_POLL_INTERVAL,
        "failsafe_enabled": CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW,
        "bot_start_time": datetime.utcfromtimestamp(BOT_START_TIME).isoformat() if BOT_START_TIME > 0 else None,
        "active_biases": {
            cid[:16]: {
                "state": bias.state.value,
                "direction": bias.bias_direction,
                "high_conf": bias.is_high_confidence,
                "trades": bias.trades_seen,
                "is_historical": REF_TRACKER.is_historical_window(cid),
            }
            for cid, bias in REF_TRACKER.conditions.items()
            if bias.state != BiasState.WAITING
        },
    }


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    return {"signals": [s.to_dict() for s in STATE.signals[-limit:]]}


@app.get("/api/orders")
async def get_orders(limit: int = 50):
    return {"orders": [o.to_dict() for o in STATE.order_attempts[-limit:]]}


@app.post("/api/start")
async def start_bot():
    global BOT_START_TIME
    if not STATE.is_running:
        STATE.is_running = True
        BOT_START_TIME = time.time()
        asyncio.create_task(ref_monitor_loop())
        asyncio.create_task(trading_loop())
        logger.info(f"Bot STARTED at {datetime.utcfromtimestamp(BOT_START_TIME).isoformat()}")
    return {"status": "running", "bot_start_time": BOT_START_TIME}


@app.post("/api/stop")
async def stop_bot():
    STATE.is_running = False
    logger.info("Bot STOPPED")
    return {"status": "stopped"}


@app.post("/api/enable_failsafe")
async def enable_failsafe():
    """Enable the failsafe that prevents trading on historical windows."""
    CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW = True
    logger.info("FAILSAFE ENABLED: Will only trade windows that opened after bot started")
    return {"failsafe_enabled": True, "bot_start_time": BOT_START_TIME}


@app.post("/api/disable_failsafe")
async def disable_failsafe():
    """Disable the failsafe (allow trading on any window with bias)."""
    CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW = False
    logger.info("FAILSAFE DISABLED: Will trade any window with bias (including historical)")
    return {"failsafe_enabled": False}


# =============================================================================
# MAIN
# =============================================================================

@app.on_event("startup")
async def on_startup():
    global BOT_START_TIME
    BOT_START_TIME = time.time()

    logger.info("=" * 70)
    logger.info("PREDIRECTIONAL BOT - LIVE REF MONITORING")
    logger.info("=" * 70)
    logger.info(f"Ref wallet: {REF_WALLET}")
    logger.info(f"Poll interval: {CONFIG.REF_POLL_INTERVAL}s (FAST)")
    logger.info(f"Enabled assets: {CONFIG.ENABLED_ASSETS}")
    logger.info(f"Skip no-bias: {CONFIG.SKIP_NO_BIAS}")
    logger.info(f"Hedge enabled: {CONFIG.HEDGE_ENABLED}")
    logger.info(f"Bias size mult: {CONFIG.BIAS_SIZE_MULT}x")
    logger.info(f"")
    logger.info(f"FAILSAFE: {CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW}")
    if CONFIG.REQUIRE_BOT_RUNNING_BEFORE_WINDOW:
        logger.info(f"  -> Only trading windows that open AFTER bot starts")
    else:
        logger.info(f"  -> Will trade ANY window with bias (historical allowed)")
    logger.info(f"Bot start time: {datetime.utcfromtimestamp(BOT_START_TIME).isoformat()}")
    logger.info("")
    logger.info("Strategy: WAIT for ref's first expensive trade, then PILE ON")
    logger.info("Orders will fail with 401 (no auth) - tracking latency only")
    logger.info("=" * 70)

    # Auto-start
    STATE.is_running = True
    asyncio.create_task(ref_monitor_loop())
    asyncio.create_task(trading_loop())
    logger.info("Bot auto-started")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
