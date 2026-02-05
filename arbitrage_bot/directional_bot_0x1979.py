#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot - PREDIRECTIONAL VERSION + SAFETY LIMITS

LIVE REF MONITORING + REAL API CALLS + PIVOT DETECTION

Strategy:
1. Poll ref account trades every 0.5s (fast mode)
2. Detect bias from first expensive trade (>=0.70)
3. WAIT until bias established, then pile on aggressively
4. MONITOR for opposite expensive (pivot signal at >=0.75)
5. REVERSE position on pivot with reduced sizing
6. Fire real API calls (will fail with 401 but tracks latency)

V2 SAFETY DESIGN (from comprehensive backtest of 23,337 trades):
- Pivot detection at $0.80 threshold (74.4% accuracy, EV $1,021)
- FULL position sizing on pivots (backtest shows better than reduced)
- NO hard caps on pivots or exposure (hard caps are dangerous)
- Warnings only for monitoring - never block critical actions
- EV of reversing on opposite expensive: +$1,021 per window at $0.80
- Win rate: 95.5% with pivot handling vs 86.1% without
"""

import asyncio
import json
import os
import sys
import time
import gc
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
TRADE_LOG_FILE = "/root/arbitrage_bot/directional_bot_0x1979_trades.jsonl"

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

# Add polymarketbot src for blockchain monitoring
POLYMARKET_SRC = Path("/root/polymarketbot/src")
sys.path.insert(0, str(POLYMARKET_SRC))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/root/directional_bot_0x1979.log") if os.path.exists("/root") else logging.StreamHandler()
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
REF_WALLET = "0x1979ae6b7e6534de9c4539d0c205e582ca637c9d"
# =============================================================================
# WEBSOCKET REF MONITORING (for <1s latency)
# =============================================================================

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, ConnectionClosedError
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not available - using polling only")

# Fast blockchain monitoring for ~2-3s REF trade detection
try:
    from blockchain_fast_detect import FastBlockchainDetector, DetectedTrade, TokenCache
    FAST_BLOCKCHAIN_AVAILABLE = True
    logger.info("FastBlockchainDetector loaded successfully")
except ImportError as e:
    FAST_BLOCKCHAIN_AVAILABLE = False
    logger.warning(f"Fast blockchain module not available: {e}")

# Smart Alchemy wrapper for credit-conserving detection
try:
    from smart_alchemy_wrapper import SmartAlchemyWrapper, AlchemyMode
    SMART_ALCHEMY_AVAILABLE = True
    logger.info("SmartAlchemyWrapper loaded successfully")
except ImportError as e:
    SMART_ALCHEMY_AVAILABLE = False
    logger.warning(f"Smart Alchemy module not available: {e}")

# Legacy import (fallback)
BLOCKCHAIN_AVAILABLE = False
try:
    from blockchain.polygon_monitor import PolygonMonitor, BlockchainTrade
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    pass

# RPC URLs for blockchain monitoring (used by legacy monitor)
POLYGON_RPC_URL = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
ALCHEMY_POLYGON_RPC = os.environ.get("ALCHEMY_POLYGON_RPC", "")  # Set this for fast detection
ALCHEMY_RPC_URL = os.environ.get("POLYGON_WS_URL", "wss://rpc-mainnet.matic.quiknode.pro")

# Reference account proxy wallet (trades route through this)
REF_PROXY_WALLET = "0x1979ae6b7e6534de9c4539d0c205e582ca637c9d"

# Trades file for token cache initialization
REF_TRADES_FILE = "/root/reference_collector/ref_0x1979_trades.json" 

# Global blockchain monitor
BLOCKCHAIN_MONITOR = None
FAST_DETECTOR = None
SMART_ALCHEMY = None


class RefWebSocketMonitor:
    """Real-time REF trade monitoring via WebSocket.
    
    Subscribes to Polymarket market websocket and filters for REF trades.
    Provides <1s latency vs 10-16s from polling API.
    """
    
    WS_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    GAMMA_API = "https://gamma-api.polymarket.com"
    
    # 15m crypto series slugs
    CRYPTO_SERIES = ["btc-up-or-down-15m", "eth-up-or-down-15m", "sol-up-or-down-15m", "xrp-up-or-down-15m"]
    
    def __init__(self, ref_wallet: str, on_ref_trade):
        self.ref_wallet = ref_wallet.lower()
        self.on_ref_trade = on_ref_trade  # Callback: async def(trade_dict)
        self._ws = None
        self._should_run = False
        self._subscribed_tokens = set()
        self._last_token_refresh = 0
        self._connection_task = None
        self._receive_task = None
        self._token_refresh_task = None
        
        # Stats
        self.trades_received = 0
        self.ref_trades_detected = 0
        self.connection_count = 0
        self.last_ref_trade_time = None
        self.is_connected = False
    
    async def start(self):
        """Start websocket monitoring."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start WS monitor - websockets library not installed")
            return False
        
        self._should_run = True
        self._connection_task = asyncio.create_task(self._connection_loop())
        self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())
        logger.info("RefWebSocketMonitor started")
        return True
    
    async def stop(self):
        """Stop websocket monitoring."""
        self._should_run = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        if self._token_refresh_task:
            self._token_refresh_task.cancel()
            try:
                await self._token_refresh_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        self.is_connected = False
        logger.info("RefWebSocketMonitor stopped")
    
    async def _connection_loop(self):
        """Main connection loop with auto-reconnect."""
        reconnect_delay = 1.0
        max_delay = 60.0
        
        while self._should_run:
            try:
                await self._connect()
                reconnect_delay = 1.0  # Reset on successful connect
                
            except Exception as e:
                logger.error(f"WS connection error: {e}")
                self.is_connected = False
                
                # Exponential backoff
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)
    
    async def _connect(self):
        """Establish websocket connection and subscribe."""
        logger.info(f"Connecting to WS: {self.WS_ENDPOINT}")
        
        async with websockets.connect(
            self.WS_ENDPOINT,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self.is_connected = True
            self.connection_count += 1
            logger.info("WS connected")
            
            # Subscribe to current window tokens
            await self._subscribe_current_tokens()
            
            # Receive loop
            await self._receive_loop()
    
    async def _subscribe_current_tokens(self):
        """Subscribe to current 15m crypto window token IDs."""
        token_ids = await self._get_current_token_ids()
        
        if not token_ids:
            logger.warning("No token IDs found for subscription")
            return
        
        # Unsubscribe from old tokens if needed
        if self._subscribed_tokens:
            old_tokens = list(self._subscribed_tokens - set(token_ids))
            if old_tokens:
                unsub_msg = json.dumps({
                    "type": "unsubscribe",
                    "channel": "market",
                    "assets_ids": old_tokens,
                })
                await self._ws.send(unsub_msg)
                logger.debug(f"Unsubscribed from {len(old_tokens)} old tokens")
        
        # Subscribe to new tokens
        sub_msg = json.dumps({
            "type": "subscribe",
            "channel": "market",
            "assets_ids": token_ids,
        })
        await self._ws.send(sub_msg)
        
        self._subscribed_tokens = set(token_ids)
        self._last_token_refresh = time.time()
        logger.info(f"Subscribed to {len(token_ids)} token IDs for 15m crypto")
    
    async def _get_current_token_ids(self) -> List[str]:
        """Get token IDs for current 15m crypto windows."""
        token_ids = []

        try:
            client = await get_http_client()
            now = time.time()
            current_window = int((now // 900) * 900)
            next_window = current_window + 900

            # Build event slugs for current and next windows
            assets = ["btc", "eth", "sol", "xrp"]
            event_slugs = []
            for asset in assets:
                event_slugs.append(f"{asset}-updown-15m-{current_window}")
                # Also get next window if within 2 minutes
                if next_window - now < 120:
                    event_slugs.append(f"{asset}-updown-15m-{next_window}")

            for event_slug in event_slugs:
                try:
                    resp = await client.get(
                        f"{self.GAMMA_API}/events",
                        params={"slug": event_slug},
                        timeout=10.0,
                    )

                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    if not data:
                        continue

                    event = data[0] if isinstance(data, list) else data

                    # Get markets from event
                    markets = event.get("markets", [])
                    for market in markets:
                        # Token IDs are in clobTokenIds field
                        clob_ids = market.get("clobTokenIds", [])
                        clob_ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                        if isinstance(clob_ids, list):
                            token_ids.extend(clob_ids)

                except Exception as e:
                    logger.debug(f"Error fetching {event_slug}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error getting token IDs: {e}")

        logger.info(f"Found {len(token_ids)} token IDs for WS subscription")
        return token_ids
    async def _token_refresh_loop(self):
        """Periodically refresh token subscriptions for new windows."""
        while self._should_run:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self._ws and self.is_connected:
                    # Check if we need to refresh (new window coming)
                    now = time.time()
                    current_window = (int(now) // 900) * 900
                    next_window = current_window + 900
                    time_to_next = next_window - now
                    
                    # Refresh 30s before new window starts
                    if time_to_next < 30 or time.time() - self._last_token_refresh > 120:
                        logger.info("Refreshing token subscriptions for new window")
                        await self._subscribe_current_tokens()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token refresh error: {e}")
    
    async def _receive_loop(self):
        """Receive and process websocket messages."""
        logger.info("[WS] Starting receive loop...")
        msg_count = 0
        while self._should_run and self._ws:
            try:
                message = await self._ws.recv()
                msg_count += 1
                if msg_count <= 5 or msg_count % 100 == 0:
                    logger.info(f"[WS] Received message #{msg_count}, len={len(message)}")
                await self._handle_message(message)
                
            except ConnectionClosed as e:
                logger.warning(f"WS connection closed: {e}")
                self.is_connected = False
                break
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"WS receive error: {e}")
    
    async def _handle_message(self, raw_message: str):
        """Parse and filter websocket messages for REF trades."""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        
        # Handle list messages (initial book snapshot)
        if isinstance(data, list):
            return  # Skip list messages for now
        
        msg_type = data.get("type", "")
        
        # DEBUG: (disabled - too noisy)
        # if msg_type not in ["book", "price_change"]:
        #     keys = list(data.keys())[:8]
        #     sample = str(data)[:200]
        #     logger.info(f"[WS-DEBUG] type={msg_type}, keys={keys}, sample={sample}")
        
        # Handle trade messages
        if msg_type == "trade":
            await self._process_trade(data)
        elif msg_type == "trades":
            for trade in data.get("trades", []):
                await self._process_trade(trade)
        elif msg_type == "price_change":
            # Batch of price updates, may contain trades
            pass
        elif msg_type == "subscribed":
            logger.debug(f"Subscription confirmed: {data}")
        elif msg_type == "error":
            logger.warning(f"WS error message: {data}")
    
    async def _process_trade(self, trade: dict):
        """Check if trade is from REF and trigger callback."""
        self.trades_received += 1
        
        # Get maker/taker addresses
        maker = (trade.get("maker") or "").lower()
        taker = (trade.get("taker") or "").lower()
        
        # Check if REF is involved
        is_ref_maker = maker == self.ref_wallet
        is_ref_taker = taker == self.ref_wallet
        
        if not is_ref_maker and not is_ref_taker:
            return
        
        # REF trade detected!
        self.ref_trades_detected += 1
        self.last_ref_trade_time = time.time()
        
        # For TAKER trades (directional signals), REF is the taker buying
        # We care most about taker trades
        side = trade.get("side", "").upper()
        price = float(trade.get("price", 0))
        size = float(trade.get("size", 0))
        
        # Determine if this is a meaningful signal
        is_taker = is_ref_taker and side == "BUY"
        
        logger.info(f"[WS] REF trade detected: {'TAKER' if is_taker else 'MAKER'} {side} @ ${price:.2f} size={size:.2f}")
        
        # Convert to format expected by process_trade
        trade_dict = {
            "transactionHash": trade.get("id", ""),
            "timestamp": trade.get("timestamp", time.time() * 1000),
            "price": price,
            "size": size,
            "side": side,
            "outcome": trade.get("outcome", ""),
            "slug": "",  # Need to look up from token_id
            "conditionId": trade.get("market", trade.get("condition_id", "")),
            "type": "TRADE",
            "_our_side": "BUY" if is_ref_taker else "SELL",
            "usdcSize": size * price if is_taker else 0,  # Only TAKER has usdcSize > 0
        }
        
        # Look up slug from token_id if possible
        token_id = trade.get("asset_id", trade.get("token_id", ""))
        if token_id:
            trade_dict["token_id"] = token_id
        
        # Trigger callback immediately
        try:
            await self.on_ref_trade(trade_dict)
        except Exception as e:
            logger.error(f"Error in ref trade callback: {e}")
    
    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            "is_connected": self.is_connected,
            "connection_count": self.connection_count,
            "trades_received": self.trades_received,
            "ref_trades_detected": self.ref_trades_detected,
            "subscribed_tokens": len(self._subscribed_tokens),
            "last_ref_trade": self.last_ref_trade_time,
        }


# Global websocket monitor instance
WS_MONITOR: Optional[RefWebSocketMonitor] = None

# =============================================================================
# WEBSOCKET TRADE CALLBACK (Primary detection - <1s latency)
# =============================================================================

async def on_websocket_ref_trade(trade_dict: dict):
    """Process trade from WebSocket - called within <1s of trade execution.
    
    This is the PRIMARY detection method for fastest latency.
    """
    global BOT_START_TIME
    detection_time = time.time()
    
    # CRITICAL: Ignore trades from before bot started (no historical catchup)
    trade_ts = trade_dict.get("timestamp", 0)
    if isinstance(trade_ts, (int, float)) and trade_ts > 1e12:
        trade_ts = trade_ts / 1000  # Convert ms to seconds
    
    if trade_ts < BOT_START_TIME:
        logger.debug(f"[WS] Ignoring historical trade (ts={trade_ts:.0f} < start={BOT_START_TIME:.0f})")
        return
    
    # Calculate real latency
    latency_ms = (detection_time - trade_ts) * 1000 if trade_ts > 0 else 0
    
    # Get token info for slug lookup
    token_id = trade_dict.get("token_id", "")
    price = float(trade_dict.get("price", 0))
    size = float(trade_dict.get("size", 0))
    usdc_size = price * size
    
    # Only process meaningful trades
    if usdc_size < 0.10:
        return
    
    # Try to get slug from token_id  
    slug = trade_dict.get("slug", "")
    if not slug and token_id and SMART_ALCHEMY and SMART_ALCHEMY.token_cache:
        token_info = SMART_ALCHEMY.token_cache.get_token_info(token_id)
        if token_info:
            slug = token_info.slug
            trade_dict["slug"] = slug
            trade_dict["outcome"] = token_info.outcome
            trade_dict["condition_id"] = token_info.condition_id
    
    # Only process 15m crypto markets
    if not slug or "-15m-" not in slug:
        return
    
    # Determine asset from slug
    asset = "?"
    for a in ["btc", "eth", "sol", "xrp"]:
        if slug.startswith(a):
            asset = a.upper()
            break
    
    logger.info(f"[WS] {asset} REF trade: {trade_dict.get('outcome', '?')} @ ${price:.2f} (${usdc_size:.2f}) | latency: {latency_ms:.0f}ms")
    
    # Convert to format expected by REF_TRACKER
    formatted_trade = {
        "side": trade_dict.get("side", "BUY"),
        "size": usdc_size,
        "price": price,
        "token_id": token_id,
        "wallet": REF_WALLET,
        "tx_hash": trade_dict.get("transactionHash", ""),
        "source": "websocket",
        "latency_ms": latency_ms,
        "timestamp": trade_ts,
        "asset": asset,
        "outcome": trade_dict.get("outcome", ""),
        "slug": slug,
        "condition_id": trade_dict.get("conditionId", trade_dict.get("condition_id", "")),
    }
    
    # Process through REF_TRACKER
    event = REF_TRACKER.process_trade(formatted_trade)
    
    if event:
        logger.info(f"[WS] BIAS EVENT: {event.event_type} for {slug}")




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
    """Bias tracking for a single condition/market with safety limits."""
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

    # SAFETY: Pivot tracking for tiered position sizing
    pivot_count: int = 0                    # Number of direction changes
    last_pivot_time: float = 0              # When last pivot occurred
    total_traded_usd: float = 0             # Running total for window exposure limit
    has_position: bool = False              # True after first order attempt in this window

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
        # V2: Never block on pivot count, just warn
        if hasattr(CONFIG, 'WARN_PIVOT_COUNT') and self.pivot_count > CONFIG.WARN_PIVOT_COUNT:
            logger.warning(f"HIGH PIVOT COUNT WARNING: {self.pivot_count} (not blocking)")
        return self.state not in [BiasState.WAITING, BiasState.PIVOT]

    @property
    def position_size_multiplier(self) -> float:
        """V2: Always return 1.0 (full position) - backtest shows this is better."""
        # Backtest results: Full position gives $90.98 avg PnL vs $84.30 with tiered
        return 1.0

    def can_trade_more(self, proposed_usd: float) -> bool:
        """V2: Always allow trading (warn but don't block)."""
        # V2: Log warning but never block
        if hasattr(CONFIG, 'WARN_EXPOSURE_USD') and (self.total_traded_usd + proposed_usd) > CONFIG.WARN_EXPOSURE_USD:
            logger.warning(f"HIGH EXPOSURE WARNING: ${self.total_traded_usd + proposed_usd:.2f} (not blocking)")
        return True  # V2: Never block

    def record_trade(self, usd_amount: float):
        """Record a trade for exposure tracking."""
        self.total_traded_usd += usd_amount


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
    # REMOVED: Old PIVOT_THRESHOLD replaced by safety version below

    # Strategy
    SKIP_NO_BIAS: bool = True      # Don't trade until bias detected
    HEDGE_ENABLED: bool = True    # DEFINITIVE: Skip minority (rebate farming)

    # FAILSAFE: Only trade windows that opened AFTER bot started
    # When enabled, we won't trade on historical biases that happened before we were running
    REQUIRE_BOT_RUNNING_BEFORE_WINDOW: bool = False  # DISABLED for now (set True to enable)

    # Sizing multipliers - MATCHED TO REF
    BIAS_SIZE_MULT: float = 13.5   # Matched to REF actual 13.5x (backtest of 23,343 trades)
    TENTATIVE_SIZE_MULT: float = 1.0
    MID_SIZE_MULT: float = 5.6         # Matched to REF actual 5.6x (backtest of 23,343 trades)
    HEDGE_SIZE_MULT: float = 1.0    # Hedges at base size (clean TAKER audit)

    # Price limits
    DOMINANT_MAX_PRICE: float = 0.95      # Breakeven at 95.6% accuracy is $0.956; margin of safety
    MINORITY_MAX_PRICE: float = 0.30
    # Phase 2: Buy opposite expensive when it reaches this price
    OPPOSITE_TRIGGER_PRICE: float = 0.65
    # Phase 3: End-of-window aggressive trading (>10min)
    PHASE3_START_SEC: float = 600.0
    PHASE3_HEDGE_SIZE_MULT: float = 0.66  # Phase 3 hedges 0.66x smaller (validated)   # Hedges SMALLER late, not larger (clean TAKER audit)
    MAX_PAIR_COST: float = 1.02

    # SLIPPAGE PROTECTION (Validated from 21k+ REF trades backtest)
    # Detection delay thresholds
    MAX_DETECTION_DELAY_S: float = 30.0      # API polling = ~16-17s typical; 5s was for broken blockchain detector
    LATE_ENTRY_THRESHOLD_S: float = 3.0      # Reduce size if >3s delay
    LATE_ENTRY_SIZE_MULT: float = 0.5        # 50% size on late entries
    
    # Slippage tolerance (based on P95 from backtest)
    # <1s: 0%, 1-3s: 7%, 3-5s: 14%, 5-10s: 15%
    BASE_SLIPPAGE_PCT: float = 2.0           # Conservative base
    SLIPPAGE_PER_SECOND: float = 3.0         # ~3% per second of delay
    MAX_SLIPPAGE_PCT: float = 15.0           # P95 ceiling at 5-10s
    
    # Skip if price moved against us by this much
    MAX_ADVERSE_MOVE_PCT: float = 10.0       # Skip if >10% adverse move

    # Timing - PRICE-BASED TRIGGERS (from REF analysis of 106,124 trades)
    # REF has a HARD 2s MINIMUM - zero 1-second gaps in 52,707 consecutive trade pairs
    # 77.6% of trades are same-second bursts, 13.9% are exactly 2s apart
    # 59.7% of 2s trades have price OR outcome change, 40.3% are position building
    PRICE_CHANGE_TRIGGER: float = 0.01  # Trade when price changes by this much
    MIN_TRADE_INTERVAL: float = 2.0     # HARD 2s MINIMUM (confirmed from data)
    MIN_TIME_TO_RESOLUTION: int = 5       # Hard stop for ALL signals (existing)
    MIN_TTR_NEW_ENTRY: int = 30            # Don't enter NEW windows with <30s remaining (backtested)
    MAX_TIME_TO_RESOLUTION: int = 900

    # Sizing - MATCHED TO REF (from analysis)
    # REF: Median individual trade $5.44, P75 $17.90
    # REF: Median burst $13.26, P75 $41.61
    TRADE_SIZE_BASE: float = 1.14   # Validated from clean 15-min TAKER data   # Base = cheap hedge median $1.35 (clean TAKER audit)
    STARTING_BALANCE: float = 200.0


    # ===== SAFETY LIMITS (validated from safety_analysis.py) =====

    # Pivot detection threshold - higher for better accuracy
    # $0.75 gives 67.3% accuracy vs 64.8% at $0.70
    PIVOT_THRESHOLD: float = 0.73      # Grid search: 95.6% at $0.73 vs 95.2% at $0.80 (285 windows)

    # Tiered position sizing after pivots
    # Proven to increase win rate from 78.5% to 85.8%
    INITIAL_POSITION_PCT: float = 1.00      # 100% on first signal
    AFTER_PIVOT_1_PCT: float = 1.00         # V2: Full position (backtest shows better)
    AFTER_PIVOT_2_PCT: float = 1.00         # V2: Full position (backtest shows better)
    MAX_PIVOTS_ALLOWED: int = 99            # V2: Never block pivots

    # Risk controls
    # MAX_SINGLE_TRADE_USD: float = 50.0   # V2: REMOVED - hard caps are dangerous
    # MAX_WINDOW_EXPOSURE_USD: float = 200.0  # V2: REMOVED - hard caps are dangerous
    SLIPPAGE_OVERRIDE_ON_PIVOT: bool = True # Execute pivot signals even at poor price
    # V2: WARNING THRESHOLDS (not blocking)
    WARN_EXPOSURE_USD: float = 500.0       # Log warning when exceeded (don't block)
    WARN_PIVOT_COUNT: int = 3              # Log warning when exceeded (don't block)
    ALWAYS_EXECUTE_PIVOTS: bool = True     # CRITICAL: Never block pivot signals

    # DAILY LOSS LIMIT (entry gate only - never blocks in-window actions)
    # Not validated from backtest (no losing days in dataset) - safety net only
    DAILY_LOSS_LIMIT_USD: float = 50.0     # Stop new entries if daily loss exceeds this
    DAILY_LOSS_LIMIT_ENABLED: bool = True  # Toggle for testing

    # Timing
    PIVOT_RESPONSE_TARGET_SEC: float = 8.0  # REF median response time

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
        # Prevent unbounded growth - keep only recent trade keys
        if len(self.processed_trade_keys) > 5000:
            # Keep a random subset (can't easily get 'recent' from a set)
            self.processed_trade_keys = set(list(self.processed_trade_keys)[-2500:])

        # Only process BUY trades for 15min crypto
        if trade.get("_our_side", trade.get("side")) != "BUY":
            return None

        slug = trade.get("slug", "")
        if "-15m-" not in slug:
            return None
        
        # CRITICAL: Validate window has not expired
        # Parse window end time from slug (format: xxx-updown-15m-TIMESTAMP)
        try:
            parts = slug.split("-")
            window_start_ts = int(parts[-1])
            window_end_ts = window_start_ts + 900  # 15 minutes
            if time.time() > window_end_ts:
                # Window has expired - REJECT this trade
                logger.debug(f"EXPIRED WINDOW: {slug} ended at {window_end_ts}, now {time.time():.0f}")
                return None
        except (ValueError, IndexError):
            pass  # If we cant parse, allow it through

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
            record_ref_trade_time(condition_id, timestamp)  # Track actual trade time for slippage
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
            # Same direction - confirm/strengthen bias (any expensive price)
            cond.state = BiasState.BIAS_UP if outcome == "Up" else BiasState.BIAS_DOWN
            cond.second_expensive_confirms = True
            logger.debug(f"BIAS REINFORCED: {slug} -> {outcome} @ ${price:.2f}")
            return "BIAS_CONFIRMED"
        else:
            # Different direction - PIVOT DETECTION
            # SAFETY: Use higher threshold ($0.75) for better pivot accuracy (67.3% vs 64.8%)
            if price < CONFIG.PIVOT_THRESHOLD:
                # Below pivot threshold - not a confirmed pivot signal
                logger.debug(f"OPPOSITE (below pivot thresh): {slug} {outcome} @ ${price:.2f} < ${CONFIG.PIVOT_THRESHOLD:.2f}")
                return None

            # V2: Log warning but NEVER block pivots
            if hasattr(CONFIG, 'WARN_PIVOT_COUNT') and cond.pivot_count >= CONFIG.WARN_PIVOT_COUNT:
                logger.warning(f"HIGH PIVOTS: {slug} has {cond.pivot_count} pivots (continuing anyway - V2)")
            # CRITICAL: Never return here - pivots must execute

            # CONFIRMED PIVOT - opposite expensive at/above pivot threshold
            old_dir = current_bias_dir
            cond.pivot_count += 1
            cond.last_pivot_time = time.time()
            cond.state = new_state
            self.pivots_detected += 1

            pos_mult = cond.position_size_multiplier * 100
            logger.info(f"PIVOT #{cond.pivot_count}: {slug} {old_dir}->{outcome} @ ${price:.2f} | Position now {pos_mult:.0f}%")
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

    def clear_expired_windows(self):
        """Clear all biases for windows that have ended (15-min windows)."""
        now = time.time()
        expired = []

        for cid, bias in self.conditions.items():
            if bias.window_start_time > 0:
                window_end = bias.window_start_time + 900  # 15 minutes
                if now > window_end:
                    expired.append(cid)

        if expired:
            for cid in expired:
                del self.conditions[cid]
            logger.info(f"[WINDOW-RESET] Cleared {len(expired)} expired window biases")
            # Reset counters for new window
            self.biases_detected = len(self.conditions)  # Remaining active biases
            self.pivots_detected = sum(c.pivot_count for c in self.conditions.values())

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
    # Daily loss tracking (for entry gate failsafe)
    daily_pnl: float = 0.0                  # Running P&L for current day
    daily_pnl_date: str = ""                # Current tracking date (YYYY-MM-DD)
    daily_windows_entered: int = 0          # Windows entered today
    daily_windows_blocked: int = 0          # Windows blocked by daily limit


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
                    # CRITICAL: Filter to TAKER trades only (usdcSize > 0)
                    usdc_size = float(t.get("usdcSize", 0) or 0)
                    if usdc_size <= 0:
                        continue  # Skip MAKER trades - not directional signals
                    t["_our_side"] = t.get("side")
                    all_trades.append(t)

        # DISABLED: Maker endpoint - MAKER trades are noise, not directional signals
        # Only TAKER trades (from activity endpoint) are used for bias detection
        # maker_url = f"{DATA_API}/trades?maker={REF_WALLET}&limit=100"
        # maker_trades = await asyncio.get_event_loop().run_in_executor(
        # None, lambda: fetch_json_sync(maker_url)
        # )
        # if maker_trades:
        # for t in maker_trades:
        # taker_side = t.get("side")
        # if taker_side == "BUY":
        # t["_our_side"] = "SELL"
        # elif taker_side == "SELL":
        # t["_our_side"] = "BUY"
        # all_trades.append(t)
        #         STATE.ref_trades_fetched = len(all_trades)
        return all_trades

    except Exception as e:
        logger.error(f"Error fetching ref trades: {e}")
        return []


# =============================================================================
# MARKET FETCHING
# =============================================================================

async def fetch_single_market(client: httpx.AsyncClient, series_slug: str) -> Optional[MarketPrice]:
    """Fetch a single market - called in parallel for speed."""
    try:
        # Step 1: Get series to find active events
        resp = await client.get(f"{GAMMA_API}/series", params={"slug": series_slug})
        if resp.status_code != 200:
            return None

        series_data = resp.json()
        if not series_data:
            return None

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
            return None

        # Get closest to expiry
        valid_events.sort(key=get_end_ts)
        event = valid_events[0]
        event_slug = event.get("slug", "")

        # Step 2: Fetch full event data (includes markets)
        event_resp = await client.get(f"{GAMMA_API}/events", params={"slug": event_slug})
        if event_resp.status_code != 200:
            return None

        event_data = event_resp.json()
        if isinstance(event_data, list):
            event_data = event_data[0] if event_data else {}

        markets_data = event_data.get("markets", [])
        if not markets_data:
            return None

        market_data = markets_data[0]
        condition_id = market_data.get("conditionId", "")

        tokens = market_data.get("clobTokenIds", [])
        if len(tokens) < 2:
            return None

        # Step 3: Fetch current prices
        market_resp = await client.get(f"{CLOB_API}/markets/{condition_id}")
        if market_resp.status_code != 200:
            return None

        market = market_resp.json()
        tokens_list = market.get("tokens", [])
        up_token = next((t for t in tokens_list if t.get("outcome") == "Up"), None)
        down_token = next((t for t in tokens_list if t.get("outcome") == "Down"), None)

        if not up_token or not down_token:
            return None

        up_price = float(up_token.get("price", 0))
        down_price = float(down_token.get("price", 0))
        end_ts = get_end_ts(event)
        resolution_time = datetime.utcfromtimestamp(end_ts)

        return MarketPrice(
            condition_id=condition_id,
            slug=event_slug,
            title=event_data.get("title", ""),
            up_token_id=up_token.get("token_id", ""),
            down_token_id=down_token.get("token_id", ""),
            up_price=up_price,
            down_price=down_price,
            pair_cost=up_price + down_price,
            resolution_time=resolution_time,
        )
    except Exception as e:
        logger.debug(f"Error fetching {series_slug}: {e}")
        return None


async def fetch_crypto_markets() -> List[MarketPrice]:
    """Fetch active 15-min crypto markets - ALL IN PARALLEL for speed."""
    series_map = {
        "btc": "btc-up-or-down-15m",
        "eth": "eth-up-or-down-15m",
        "sol": "sol-up-or-down-15m",
        "xrp": "xrp-up-or-down-15m",
    }

    series_slugs = [series_map[a.lower()] for a in CONFIG.ENABLED_ASSETS if a.lower() in series_map]

    async with httpx.AsyncClient(timeout=2.0) as client:  # Reduced from 5.0s
        # PARALLEL fetch all markets at once
        tasks = [fetch_single_market(client, slug) for slug in series_slugs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        markets = []
        for r in results:
            if isinstance(r, MarketPrice):
                markets.append(r)
        
        return markets


async def _old_fetch_crypto_markets() -> List[MarketPrice]:
    """OLD SEQUENTIAL VERSION - KEPT FOR REFERENCE."""
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
# SLIPPAGE PROTECTION (Validated from backtest of 21,513 15m crypto trades)
# =============================================================================

def check_slippage_safety(
    ref_entry_price: float,
    current_price: float,
    detection_delay_s: float,
    trade_direction: str,
) -> tuple:
    """
    Check if trade is safe given detection delay and price movement.
    
    Based on backtest:
    - <1s: 0% typical slippage, 14.5% adverse
    - 1-3s: 7% P95, 26.5% adverse
    - 3-5s: 14% P95, 29.8% adverse  
    - 5-10s: 15% P95, 34.9% adverse
    - >10s: 18% P95, 48.9% adverse (coin flip!)
    
    Returns: (should_trade, reason, adjustments)
    """
    # Price movement calculation
    if ref_entry_price > 0:
        price_change_pct = ((current_price - ref_entry_price) / ref_entry_price) * 100
    else:
        price_change_pct = 0.0
    
    # Adverse = price went up (worse entry for us as buyer)
    is_adverse = price_change_pct > 0
    
    # SKIP CONDITIONS
    if detection_delay_s > CONFIG.MAX_DETECTION_DELAY_S:
        return False, f"Detection {detection_delay_s:.1f}s > {CONFIG.MAX_DETECTION_DELAY_S}s max", {}
    
    if is_adverse and abs(price_change_pct) > CONFIG.MAX_ADVERSE_MOVE_PCT:
        return False, f"Adverse move {price_change_pct:+.1f}% > {CONFIG.MAX_ADVERSE_MOVE_PCT}% max", {}
    
    # Calculate allowed slippage based on delay
    allowed = CONFIG.BASE_SLIPPAGE_PCT + (detection_delay_s * CONFIG.SLIPPAGE_PER_SECOND)
    allowed = min(allowed, CONFIG.MAX_SLIPPAGE_PCT)
    
    if abs(price_change_pct) > allowed:
        return False, f"Slippage {abs(price_change_pct):.1f}% > {allowed:.1f}% allowed", {}
    
    # Size adjustment for late entries
    size_mult = 1.0
    if detection_delay_s > CONFIG.LATE_ENTRY_THRESHOLD_S:
        size_mult = CONFIG.LATE_ENTRY_SIZE_MULT
    
    return True, "OK", {
        "max_slippage_pct": allowed,
        "size_multiplier": size_mult,
        "detection_delay_s": detection_delay_s,
        "price_change_pct": price_change_pct,
    }


# Track last ref trade timestamp per market for slippage calc
_REF_TRADE_TIMESTAMPS = {}

def record_ref_trade_time(condition_id: str, trade_timestamp: float = None):
    """Record actual time of REFs trade for slippage calculation."""
    _REF_TRADE_TIMESTAMPS[condition_id] = trade_timestamp if trade_timestamp else time.time()

def get_detection_delay(condition_id: str) -> float:
    """Get seconds since we detected REFs trade for this market."""
    if condition_id in _REF_TRADE_TIMESTAMPS:
        return time.time() - _REF_TRADE_TIMESTAMPS[condition_id]
    return 2.0  # Default assumption if unknown


def _slippage_blocks_entry(condition_id: str, current_price: float, outcome: str, asset: str, bias) -> bool:
    """
    Check if slippage should BLOCK a first-entry trade.

    ONLY called when we're FLAT (bias.has_position is False).
    Once we have any position in the window, this is NEVER called -
    all signals (pivots, confirmations, secondary) execute unconditionally
    because being wrong-direction is worse than bad slippage.

    Returns True if entry should be BLOCKED, False if OK to proceed.
    """
    detection_delay = get_detection_delay(condition_id)
    ref_price = bias.first_expensive_price if bias.first_expensive_price > 0 else current_price
    safe, reason, adj = check_slippage_safety(ref_price, current_price, detection_delay, outcome)
    if not safe:
        logger.info(f"SLIPPAGE BLOCK {asset}: {outcome} @ ${current_price:.3f} | {reason} (flat, safe to skip)")
        return True  # BLOCK the entry
    # Log slippage info even when OK
    if adj.get("price_change_pct", 0) != 0:
        logger.debug(f"SLIPPAGE OK {asset}: {adj.get('price_change_pct', 0):+.1f}% move, {detection_delay:.1f}s delay")
    return False  # OK to proceed


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
    # Prevent unbounded dict growth - keep only recent entries
    if len(STATE.last_trade_time) > 500:
        # Keep entries from last hour
        cutoff = time.time() - 3600
        STATE.last_trade_time = {k: v for k, v in STATE.last_trade_time.items() if v > cutoff}
        STATE.last_trade_price = {k: STATE.last_trade_price[k] for k in STATE.last_trade_time.keys() if k in STATE.last_trade_price}


def calculate_size(price: float, signal_type: str, is_high_conf: bool, is_phase3: bool = False, bias: ConditionBias = None) -> float:
    """Calculate trade size with SAFETY LIMITS applied.

    Size scaling:
    - Cheap (<$0.30): 1.0x base (hedges)
    - Mid ($0.30-0.70): 5.5x base
    - Expensive (>=$0.70): 12.8x base (signals)

    SAFETY LIMITS:
    - Tiered position sizing after pivots (100% -> 50% -> 25%)
    - Max single trade limit ($50)
    - Window exposure limit ($200)
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

    raw_size = base * mult

    # ================================================================
    # APPLY SAFETY LIMITS
    # ================================================================

    # V2: No position reduction - backtest shows full position is better
    # if bias:
    #     position_mult = bias.position_size_multiplier  # Always 1.0 in V2

    # V2: No hard caps - just log for monitoring
    # The backtest shows hard caps can prevent critical pivot responses
    # which is MORE dangerous than allowing larger positions

    return raw_size


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

    # V2: Log warnings but never block
    if hasattr(CONFIG, 'WARN_PIVOT_COUNT') and bias.pivot_count >= CONFIG.WARN_PIVOT_COUNT:
        logger.debug(f"{asset}: HIGH PIVOTS ({bias.pivot_count}) - warning only, continuing")

    if hasattr(CONFIG, 'WARN_EXPOSURE_USD') and bias.total_traded_usd >= CONFIG.WARN_EXPOSURE_USD:
        logger.debug(f"{asset}: HIGH EXPOSURE (${bias.total_traded_usd:.2f}) - warning only, continuing")

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

    # ===== ENTRY GATES (new windows only - never block in-window actions) =====
    if not bias.has_position:
        # Gate 1: Min time-to-resolution for NEW entries (backtested: 30s optimal)
        if market.resolution_time:
            ttr = (market.resolution_time - datetime.utcnow()).total_seconds()
            if ttr < CONFIG.MIN_TTR_NEW_ENTRY:
                logger.debug(f"{asset}: SKIP NEW ENTRY - only {ttr:.0f}s remaining (min {CONFIG.MIN_TTR_NEW_ENTRY}s)")
                return

        # Gate 2: Daily loss limit (stop entering new windows if daily loss exceeded)
        if CONFIG.DAILY_LOSS_LIMIT_ENABLED:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if STATE.daily_pnl_date != today:
                # New day - reset counters
                if STATE.daily_pnl_date:
                    logger.info(f"DAILY RESET: {STATE.daily_pnl_date} P&L ${STATE.daily_pnl:+.2f} | "
                                f"Entered {STATE.daily_windows_entered} | Blocked {STATE.daily_windows_blocked}")
                STATE.daily_pnl = 0.0
                STATE.daily_pnl_date = today
                STATE.daily_windows_entered = 0
                STATE.daily_windows_blocked = 0
            if STATE.daily_pnl <= -CONFIG.DAILY_LOSS_LIMIT_USD:
                STATE.daily_windows_blocked += 1
                logger.warning(f"{asset}: DAILY LIMIT HIT - P&L ${STATE.daily_pnl:+.2f} <= "
                               f"-${CONFIG.DAILY_LOSS_LIMIT_USD} | Blocking new entries")
                return

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

    # FIX: Use bias-detected price if higher than stale CLOB price
    # This ensures we trade at prices comparable to REF
    if bias.first_expensive_price > 0 and bias.first_expensive_outcome == dom_outcome:
        if bias.first_expensive_price > dom_price:
            logger.debug(f"{asset}: Using bias price ${bias.first_expensive_price:.3f} > CLOB ${dom_price:.3f}")
            dom_price = bias.first_expensive_price

    # DOMINANT SIDE - aggressive buying (with price-based trigger)
    # Max price only applies to first entry; once positioned, execute at any price (pivots etc.)
    price_ok = dom_price <= CONFIG.DOMINANT_MAX_PRICE or bias.has_position
    if price_ok and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for dominant outcome
        if not should_trade_price_trigger(condition_id, dom_outcome, dom_price):
            logger.debug(f"{asset}: SKIP {dom_outcome} - price unchanged from ${STATE.last_trade_price.get(f'{condition_id}:{dom_outcome}', 0):.3f}")
        # SLIPPAGE CHECK DISABLED: Backtested against 271 windows - would block
        # 110 entries (107 correct, 3 wrong). NET NEGATIVE by 104 windows.
        # 15min binary markets are too volatile for price-based entry gating.
        # elif not bias.has_position and _slippage_blocks_entry(condition_id, dom_price, dom_outcome, asset, bias):
        #     pass  # Slippage too high for first entry - we're flat, safe to skip
        else:
            size_usd = calculate_size(dom_price, "BIAS_DOMINANT", is_high_conf, is_phase3, bias)
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
            if len(STATE.signals) > 200:
                STATE.signals = STATE.signals[-100:]

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
            if len(STATE.order_attempts) > 200:
                STATE.order_attempts = STATE.order_attempts[-100:]

            logger.info(f"  ORDER: {result['status_code']} in {result['submission_ms']:.0f}ms | {result.get('error_message', 'OK')}")

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, dom_outcome, dom_price)

            # SAFETY: Record trade for window exposure tracking
            bias.record_trade(size_usd)
            bias.has_position = True  # Committed to this window - no more slippage checks

    # PHASE 2: OPPOSITE SIDE EXPENSIVE - when minority reaches trigger price
    # This implements the secondary signal: buy opposite when it becomes attractive
    min_price_ok = min_price <= CONFIG.DOMINANT_MAX_PRICE or bias.has_position
    if min_price >= CONFIG.OPPOSITE_TRIGGER_PRICE and min_price_ok and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for minority outcome
        if not should_trade_price_trigger(condition_id, min_outcome, min_price):
            logger.debug(f"{asset}: SKIP {min_outcome} SECONDARY - price unchanged")
        else:
            size_usd = calculate_size(min_price, "BIAS_DOMINANT", is_high_conf, is_phase3, bias)  # Same size as dominant
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
            if len(STATE.signals) > 200:
                STATE.signals = STATE.signals[-100:]

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
            if len(STATE.order_attempts) > 200:
                STATE.order_attempts = STATE.order_attempts[-100:]
            logger.info(f"  ORDER: {result['status_code']} in {result['submission_ms']:.0f}ms | {result.get('error_message', 'OK')}")

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, min_outcome, min_price)
            bias.has_position = True

    # PHASE 3: Both directions expensive trading (when >10min into window)
    if is_phase3 and min_price >= CONFIG.EXPENSIVE_LOW and min_price_ok and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for minority outcome (Phase 3)
        if not should_trade_price_trigger(condition_id, min_outcome, min_price):
            logger.debug(f"{asset}: SKIP {min_outcome} PHASE3 - price unchanged")
        else:
            size_usd = calculate_size(min_price, "BIAS_DOMINANT", is_high_conf, is_phase3, bias)
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
            if len(STATE.signals) > 200:
                STATE.signals = STATE.signals[-100:]

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
            if len(STATE.order_attempts) > 200:
                STATE.order_attempts = STATE.order_attempts[-100:]
            logger.info(f"  ORDER: {result['status_code']} in {result['submission_ms']:.0f}ms | {result.get('error_message', 'OK')}")

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, min_outcome, min_price)
            bias.has_position = True

    # MINORITY SIDE - optional tiny hedge
    if CONFIG.HEDGE_ENABLED and min_price <= CONFIG.MINORITY_MAX_PRICE and market.pair_cost <= CONFIG.MAX_PAIR_COST:
        # Check price-based trigger for minority hedge
        if not should_trade_price_trigger(condition_id, min_outcome, min_price):
            logger.debug(f"{asset}: SKIP {min_outcome} HEDGE - price unchanged")
        else:
            # Size based on price tier and phase (calculate_size handles phase3)
            size_usd = calculate_size(min_price, "BIAS_HEDGE", is_high_conf, is_phase3, bias)
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
            if len(STATE.signals) > 200:
                STATE.signals = STATE.signals[-100:]

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
            if len(STATE.order_attempts) > 200:
                STATE.order_attempts = STATE.order_attempts[-100:]

            # Record trade for price-based trigger tracking
            record_trade_for_trigger(condition_id, min_outcome, min_price)
            bias.has_position = True


# =============================================================================
# MAIN LOOPS
# =============================================================================

async def ref_monitor_loop():
    """Monitor ref account via FAST blockchain detection (primary) with API fallback.

    FastBlockchainDetector monitors the REF proxy wallet directly on blockchain.
    Expected latency: ~2-3 seconds (vs 10-16s from data-api polling).

    Priority order:
    1. FastBlockchainDetector polling (fast: ~2-3s)
    2. Data-API polling (slow: ~10-16s) - fallback only
    """
    global BLOCKCHAIN_MONITOR, WS_MONITOR, FAST_DETECTOR, SMART_ALCHEMY

    # Track detection timestamps for latency measurement
    blockchain_detections = []

    async def on_fast_trade(trade):
        """Process trade from FastBlockchainDetector - called within ~2-3s of on-chain."""
        detection_time = time.time()

        # Only process 15m crypto markets (our target)
        if not trade.slug or "-15m-" not in trade.slug:
            return

        # Convert DetectedTrade to dict format expected by REF_TRACKER
        trade_dict = {
            "side": trade.side,
            "size": float(trade.usdc_amount),
            "price": float(trade.price) if trade.price else 0.0,
            "token_id": trade.token_id,
            "wallet": trade.wallet,
            "tx_hash": trade.tx_hash,
            "source": "fast_blockchain",
            "latency_ms": trade.detection_latency_ms,
            "timestamp": trade.timestamp.timestamp() if hasattr(trade.timestamp, "timestamp") else time.time(),
            "asset": trade.asset.upper(),
            "outcome": trade.outcome,
            "slug": trade.slug,
            "conditionId": trade.condition_id,
        }

        event = REF_TRACKER.process_trade(trade_dict)

        # Log every trade detection (for visibility)
        lat_ms = trade.detection_latency_ms
        logger.info(f"[FAST-BC] {trade.asset.upper()} {trade.side} {trade.outcome} @ ${float(trade.price):.3f} (${float(trade.usdc_amount):.2f}) | latency: {lat_ms:.0f}ms")

        if event:
            logger.info(f"[FAST-BC] >>> BIAS SIGNAL: {event}")
            blockchain_detections.append({
                "time": detection_time,
                "latency_ms": lat_ms,
                "timestamp": trade.timestamp.timestamp() if hasattr(trade.timestamp, "timestamp") else time.time(),
                "side": trade.side,
                "size": float(trade.usdc_amount),
                "asset": trade.asset,
            })

    # Start blockchain monitoring
    # Priority: SmartAlchemy (if Alchemy RPC configured) > FastDetector > API polling
    fast_started = False

    # Option 1: SmartAlchemyWrapper - uses Alchemy smartly to save credits
    if SMART_ALCHEMY_AVAILABLE and ALCHEMY_POLYGON_RPC:
        try:
            logger.info("[SmartAlchemy] Initializing SmartAlchemyWrapper...")
            logger.info(f"[SmartAlchemy] Alchemy RPC configured: {ALCHEMY_POLYGON_RPC[:40]}...")

            # Create token cache
            token_cache = TokenCache(trades_file=REF_TRADES_FILE)
            await token_cache.refresh()

            SMART_ALCHEMY = SmartAlchemyWrapper(
                ref_proxy_wallet=REF_PROXY_WALLET,
                on_trade_callback=on_fast_trade,
                token_cache=token_cache,
                alchemy_rpc=ALCHEMY_POLYGON_RPC,
                cheap_rpc=ALCHEMY_POLYGON_RPC,  # Use QuickNode for cheap RPC too
            )

            if await SMART_ALCHEMY.start():
                logger.info("[SmartAlchemy] Started! Window-aware detection active")
                logger.info("[SmartAlchemy] Strategy: Sleep 25s -> Alchemy -> Cutoff on expensive")
                fast_started = True
            else:
                logger.warning("[SmartAlchemy] Failed - falling back to basic detector")
                SMART_ALCHEMY = None
        except Exception as e:
            logger.error(f"[SmartAlchemy] Init error: {e}")
            SMART_ALCHEMY = None

    # Option 2: Basic FastBlockchainDetector (no Alchemy, free RPC only)
    if not fast_started and FAST_BLOCKCHAIN_AVAILABLE:
        try:
            logger.info("[FAST-BC] Initializing FastBlockchainDetector (no Alchemy)...")

            FAST_DETECTOR = FastBlockchainDetector(
                ref_proxy_wallet=REF_PROXY_WALLET,
                on_trade_callback=on_fast_trade,
                poll_interval=2.0,
                trades_file=REF_TRADES_FILE,
            )

            if await FAST_DETECTOR.start():
                logger.info("[FAST-BC] REF proxy monitoring started!")
                logger.info("[FAST-BC] Expected latency: ~2-3 seconds")
                fast_started = True
            else:
                logger.warning("[FAST-BC] Failed to start")
                FAST_DETECTOR = None
        except Exception as e:
            logger.error(f"[FAST-BC] Init error: {e}")
            FAST_DETECTOR = None

    if not fast_started:
        logger.warning("[MONITOR] No fast blockchain detection - using API polling (10-16s latency)")

    # Polling loop (API fallback for any missed trades)
    logger.info(f"Starting data-api polling (every {CONFIG.REF_POLL_INTERVAL}s) as backup")
    last_status_log = 0
    last_api_poll = 0

    while STATE.is_running:

        try:
            # Clear expired window biases to prevent stale data
            REF_TRACKER.clear_expired_windows()
            # Force garbage collection to prevent memory buildup
            gc.collect()

            now = time.time()

            # Always do API polling as backup, but less frequently if fast detection active
            poll_interval = 5.0 if fast_started else CONFIG.REF_POLL_INTERVAL

            if now - last_api_poll >= poll_interval:
                trades = await fetch_recent_ref_trades()
                for trade in trades:
                    REF_TRACKER.process_trade(trade)
                last_api_poll = now

            # Status logging every 30s
            if now - last_status_log > 30:
                if SMART_ALCHEMY:
                    stats = SMART_ALCHEMY.get_stats()
                    logger.info(f"[STATUS] SmartAlchemy: mode={stats.mode.value} | {stats.seconds_into_window}s into window | Expensive: {stats.expensive_detected} | Biases: {REF_TRACKER.biases_detected}")
                elif FAST_DETECTOR:
                    logger.info(f"[STATUS] Fast-BC: {FAST_DETECTOR.trades_detected} trades | Avg latency: {FAST_DETECTOR.avg_latency_ms:.0f}ms | Biases: {REF_TRACKER.biases_detected}")
                else:
                    logger.info(f"[STATUS] API polling | Trades: {REF_TRACKER.trades_processed} | Biases: {REF_TRACKER.biases_detected}")
                last_status_log = now

        except Exception as e:
            logger.error(f"Ref monitor error: {e}")

        await asyncio.sleep(0.5)  # Check frequently but let async tasks run

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

        await asyncio.sleep(1.0)  # Reduced from 0.1s to prevent event loop contention


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
                "pivots": bias.pivot_count,
                "position_pct": int(bias.position_size_multiplier * 100),
                "exposure_usd": round(bias.total_traded_usd, 2),
            }
            for cid, bias in REF_TRACKER.conditions.items()
            if bias.state != BiasState.WAITING
        },
        "websocket": WS_MONITOR.get_stats() if WS_MONITOR else None,
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
    logger.info(f"=== V2 SAFETY DESIGN (no hard caps) ===")
    logger.info(f"Pivot threshold: ${CONFIG.PIVOT_THRESHOLD:.2f} (better EV at 0.80)")
    logger.info(f"Position on pivot: FULL (backtest shows better than reduced)")
    logger.info(f"Hard caps: DISABLED (warnings only)")
    if hasattr(CONFIG, 'WARN_EXPOSURE_USD'):
        logger.info(f"Warn exposure: ${CONFIG.WARN_EXPOSURE_USD:.2f}")
    if hasattr(CONFIG, 'WARN_PIVOT_COUNT'):
        logger.info(f"Warn pivots: {CONFIG.WARN_PIVOT_COUNT}")
    logger.info(f"ALWAYS_EXECUTE_PIVOTS: True (critical)")
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
    
    
    # Initialize WebSocket monitor for <1s latency detection
    global WS_MONITOR
    if WEBSOCKETS_AVAILABLE:
        try:
            WS_MONITOR = RefWebSocketMonitor(
                ref_wallet=REF_WALLET,
                on_ref_trade=on_websocket_ref_trade
            )
            await WS_MONITOR.start()
            logger.info("[WS] WebSocket monitor ENABLED - <1s latency detection active")
        except Exception as e:
            logger.error(f"[WS] Failed to start WebSocket monitor: {e}")
            WS_MONITOR = None
    else:
        logger.warning("[WS] websockets library not available - using polling only")
    
    logger.info("Bot auto-started")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
