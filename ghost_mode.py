#!/usr/bin/env python3
"""
Ghost Mode - Real-time copy trade simulation without live execution.

This monitors REAL trades from tracked accounts on Polymarket and simulates
exactly what the copy trading bot would do - including sending API requests
(which will fail without a connected wallet, but shows timing).

Features:
- Real-time monitoring of tracked accounts via Polymarket API
- Keyword filtering for specific markets
- Drawdown-based stoploss protection
- Full API request simulation (ghost execution)
- NO retroactive position mirroring - only new trades

NOTE: This file contains ONLY ghost mode specific logic.
Core copy trading infrastructure (accounts, slippage tiers) lives in src/copytrade/
"""

import asyncio
import sys
import os
import time
import json
import uuid
import requests
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Cross-platform path handling - get the directory where this script lives
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import core copy trading infrastructure
from src.copytrade import CopyTradeAccount, DEFAULT_SLIPPAGE_TIERS, AccountManager

# Try to import blockchain monitoring (optional - requires RPC URL)
try:
    from src.blockchain import PolygonMonitor, BlockchainTrade
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    PolygonMonitor = None
    BlockchainTrade = None

# Try to import live execution modules (optional - requires PRIVATE_KEY)
try:
    from src.api.clob import CLOBClient
    from src.api.auth import PolymarketAuth
    LIVE_EXECUTION_AVAILABLE = True
except ImportError:
    LIVE_EXECUTION_AVAILABLE = False
    CLOBClient = None
    PolymarketAuth = None

# State persistence file (cross-platform)
STATE_FILE = PROJECT_ROOT / "ghost_state.json"


@dataclass
class GhostTrade:
    """A simulated trade that would have been executed in ghost mode."""

    id: str
    timestamp: datetime
    account_name: str
    market_id: str
    market_name: str
    outcome: str
    side: str
    target_size: Decimal
    target_price: Decimal
    our_size: Decimal
    our_price: Decimal
    status: str  # 'would_execute', 'filtered_keyword', 'filtered_stoploss', 'filtered_slippage', 'api_simulated', 'api_error'

    # Timing
    detection_ms: float = 0  # API call time
    true_latency_ms: float = 0  # Time from trade execution to detection (the important one!)
    api_call_ms: float = 0
    total_ms: float = 0

    # Slippage info
    actual_slippage_pct: Optional[Decimal] = None
    max_allowed_slippage_pct: Optional[Decimal] = None

    # Order routing info
    order_type: str = "market"  # 'market' or 'limit'
    order_params: Optional[dict] = None  # Full order parameters that would be sent

    # API simulation result
    api_response: Optional[dict] = None
    api_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "account_name": self.account_name,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "outcome": self.outcome,
            "side": self.side,
            "target_size": float(self.target_size),
            "target_price": float(self.target_price),
            "our_size": float(self.our_size),
            "our_price": float(self.our_price),
            "status": self.status,
            "detection_ms": round(self.detection_ms, 2),
            "true_latency_ms": round(self.true_latency_ms, 2),  # Time from trade to detection
            "api_call_ms": round(self.api_call_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "actual_slippage_pct": float(self.actual_slippage_pct * 100) if self.actual_slippage_pct is not None else None,
            "max_allowed_slippage_pct": float(self.max_allowed_slippage_pct * 100) if self.max_allowed_slippage_pct is not None else None,
            "order_type": self.order_type,
            "order_params": self.order_params,
            "api_response": self.api_response,
            "api_error": self.api_error,
        }


class GhostModeState:
    """
    State manager for ghost mode simulation.

    Uses AccountManager from src/copytrade for account operations,
    adding ghost-mode specific tracking (simulated trades, positions, stats).
    """

    def __init__(self, load_from_file: bool = True):
        self.enabled = False
        self.is_live_mode = False  # False = ghost mode, True = live mode (with wallet)
        self.started_at: Optional[datetime] = None

        # Blockchain monitoring configuration
        # Set this to a Polygon RPC URL for ~2-5s latency vs ~15-21s with API polling
        # Example: "https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY"
        self.polygon_rpc_url: Optional[str] = os.environ.get("POLYGON_RPC_URL")
        self._blockchain_monitor: Optional[PolygonMonitor] = None
        self.blockchain_enabled = False
        self.blockchain_trades_detected = 0

        # Use AccountManager for account operations
        self._account_manager = AccountManager(STATE_FILE, load_from_file=load_from_file)

        # Ghost-mode specific state
        self.ghost_trades: List[GhostTrade] = []
        self.positions: Dict[str, dict] = {}  # market_id:outcome -> position
        self.missed_trades: List[dict] = []

        # Stats (ghost mode specific)
        self.trades_detected = 0
        self.trades_would_execute = 0
        self.trades_filtered_keyword = 0
        self.trades_filtered_stoploss = 0
        self.trades_filtered_slippage = 0
        self.trades_filtered_limit = 0  # Trades skipped due to limit order price check
        self.trades_missed_offline = 0
        self.api_calls_simulated = 0
        self.api_errors = 0

        # Latency tracking
        self.detection_latencies: List[float] = []
        self.api_latencies: List[float] = []

        # P&L tracking (simulated)
        self.simulated_balance = Decimal("10000")
        self.simulated_pnl = Decimal("0")

        # Last shutdown timestamp (for detecting missed trades)
        self.last_shutdown: Optional[datetime] = None

        # Live trading infrastructure (initialized when entering live mode)
        self._clob_client: Optional[CLOBClient] = None
        self._auth: Optional[PolymarketAuth] = None
        self._private_key: Optional[str] = os.environ.get("PRIVATE_KEY")
        # Proxy/funder wallet address (where funds are held on Polymarket)
        # This is different from the signer address derived from private key
        self._funder_address: Optional[str] = os.environ.get("FUNDER_ADDRESS")

        # Add default account if none loaded
        if not self.accounts:
            self.add_account(
                name="automatedaitradingbot",
                wallet="0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11",
                keywords=["temperature", "tempature", "weather", "celsius", "fahrenheit", "°f", "°c"],
                max_drawdown_percent=Decimal("15"),
            )

    # Delegate account operations to AccountManager
    @property
    def accounts(self) -> Dict[int, CopyTradeAccount]:
        return self._account_manager.accounts

    @property
    def seen_trade_hashes(self) -> Set[str]:
        return self._account_manager.seen_trade_hashes

    def save_state(self):
        """Save state to persistence file."""
        self._account_manager.save_state()

    def add_account(self, **kwargs) -> CopyTradeAccount:
        """Add an account to track."""
        return self._account_manager.add_account(**kwargs)

    def update_account(self, account_id: int, **kwargs) -> Optional[CopyTradeAccount]:
        """Update an existing account's settings."""
        return self._account_manager.update_account(account_id, **kwargs)

    def delete_account(self, account_id: int) -> bool:
        """Delete an account."""
        return self._account_manager.delete_account(account_id)

    # Ghost mode specific methods
    def clear_trading_state(self):
        """Clear all trading state (positions, trades, stats) but keep accounts."""
        self.ghost_trades = []
        self.positions = {}
        self.missed_trades = []

        # Reset stats
        self.trades_detected = 0
        self.trades_would_execute = 0
        self.trades_filtered_keyword = 0
        self.trades_filtered_stoploss = 0
        self.trades_filtered_slippage = 0
        self.trades_filtered_limit = 0
        self.trades_missed_offline = 0
        self.api_calls_simulated = 0
        self.api_errors = 0

        # Reset latency tracking
        self.detection_latencies = []
        self.api_latencies = []

        # Reset P&L
        self.simulated_balance = Decimal("10000")
        self.simulated_pnl = Decimal("0")

        # DON'T clear seen_trade_hashes - we want to remember what we've processed
        # to avoid duplicates after restart

        print("  [State] Trading state cleared (accounts preserved)")

    def init_live_trading(self) -> bool:
        """Initialize live trading infrastructure.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not LIVE_EXECUTION_AVAILABLE:
            print("  [Live] Execution modules not available")
            return False

        if not self._private_key:
            print("  [Live] PRIVATE_KEY not set in environment")
            return False

        try:
            # Initialize auth with funder address if set (proxy wallet)
            self._auth = PolymarketAuth(
                private_key=self._private_key,
                funder_address=self._funder_address,
            )
            signer = self._auth.signer_address[:10]
            funder = self._funder_address[:10] if self._funder_address else signer
            print(f"  [Live] Auth initialized - Signer: {signer}... Funder: {funder}...")
            return True
        except Exception as e:
            print(f"  [Live] Failed to initialize auth: {e}")
            return False

    def is_live_ready(self) -> bool:
        """Check if live trading is ready."""
        return (
            LIVE_EXECUTION_AVAILABLE and
            self._private_key is not None and
            self._auth is not None
        )

    async def check_missed_trades(self, ws_broadcast):
        """Check for trades that occurred while system was offline."""
        if not self.last_shutdown:
            return

        shutdown_ts = self.last_shutdown.timestamp()
        current_ts = datetime.utcnow().timestamp()

        for account in self.accounts.values():
            if not account.enabled:
                continue

            try:
                url = f"https://data-api.polymarket.com/activity?user={account.wallet}&type=TRADE&limit=50"
                resp = requests.get(url, timeout=10)

                if resp.status_code != 200:
                    continue

                trades = resp.json()

                for trade in trades:
                    trade_ts = trade.get('timestamp', 0)
                    trade_hash = f"{trade.get('transactionHash')}_{trade.get('timestamp')}_{trade.get('asset')}"

                    # Trade happened while we were offline
                    if shutdown_ts < trade_ts < current_ts:
                        if trade_hash not in self.seen_trade_hashes:
                            market_name = trade.get('title', 'Unknown')

                            # Check if it matches keywords
                            if account.matches_keywords(market_name):
                                self.missed_trades.append({
                                    'account_name': account.name,
                                    'market_name': market_name,
                                    'side': trade.get('side', 'BUY'),
                                    'price': trade.get('price', 0),
                                    'size': trade.get('usdcSize', 0),
                                    'timestamp': trade_ts,
                                    'reason': 'System offline',
                                })
                                self.trades_missed_offline += 1
                                self._account_manager.add_seen_trade_hash(trade_hash)

                                print(f"  [MISSED] {trade.get('side')} ${trade.get('usdcSize', 0):.2f} - {market_name[:40]} (system offline)")

            except Exception as e:
                print(f"  [Error] Checking missed trades for {account.name}: {e}")

        if self.missed_trades:
            await ws_broadcast({
                "type": "missed_trades",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "count": len(self.missed_trades),
                    "trades": self.missed_trades[:20],
                },
            })

    def start(self):
        """Start ghost mode monitoring."""
        self.enabled = True
        self.started_at = datetime.utcnow()
        # Mark current positions as baseline - don't copy existing positions
        for account in self.accounts.values():
            account.last_seen_trade_id = None

    def stop(self):
        """Stop ghost mode monitoring."""
        self.enabled = False
        self.save_state()

    def set_polygon_rpc(self, rpc_url: str) -> None:
        """
        Configure Polygon RPC URL for blockchain monitoring.

        This enables ~2-5s detection latency vs ~15-21s with API polling.

        Args:
            rpc_url: Polygon RPC URL (HTTP or WebSocket)
                     Example: "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"
        """
        self.polygon_rpc_url = rpc_url
        print(f"  [Blockchain] RPC URL configured: {rpc_url[:40]}...")

    def get_live_wallet_address(self) -> Optional[str]:
        """Get our live trading wallet address (funder/proxy wallet for portfolio lookups)."""
        # Use funder address if set (this is the proxy wallet where funds are held)
        if self._funder_address:
            return self._funder_address
        # Fall back to signer address if no funder set
        if self._auth and hasattr(self._auth, 'signer_address'):
            return self._auth.signer_address
        return None

    def _get_onchain_usdc_balance(self, wallet: str) -> float:
        """Fetch USDC balance directly from Polygon blockchain.

        Args:
            wallet: Ethereum wallet address

        Returns:
            USDC balance as float (already divided by 10^6 for decimals)
        """
        if not self.polygon_rpc_url:
            return 0.0

        try:
            # USDC contract on Polygon: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
            # Method: balanceOf(address) = 0x70a08231
            wallet_padded = wallet.lower().replace("0x", "").zfill(64)
            data = f"0x70a08231{wallet_padded}"

            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_call",
                "params": [{
                    "to": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                    "data": data
                }, "latest"]
            }

            resp = requests.post(self.polygon_rpc_url, json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json().get("result", "0x0")
                balance_raw = int(result, 16) if result else 0
                # USDC has 6 decimals
                return balance_raw / 1_000_000
        except Exception as e:
            print(f"  [Blockchain] Error fetching USDC balance: {e}")

        return 0.0

    def get_live_portfolio(self) -> dict:
        """Fetch live portfolio data for our trading wallet.

        Uses on-chain USDC balance (accurate) combined with Polymarket position data.
        """
        wallet = self.get_live_wallet_address()
        if not wallet:
            return {"balance": 0, "total_value": 0, "realized_pnl": 0, "unrealized_pnl": 0}

        # Get actual on-chain USDC balance (the REAL balance)
        usdc_balance = self._get_onchain_usdc_balance(wallet)

        # Also try Polymarket API for position data (may return 0 for fresh accounts)
        position_value = 0.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        positions_count = 0

        try:
            # Fetch positions from Polymarket Data API
            url = f"https://data-api.polymarket.com/positions?user={wallet.lower()}"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                positions = resp.json()
                positions_count = len(positions)

                for pos in positions:
                    size = float(pos.get("size", 0))
                    current_price = float(pos.get("curPrice", pos.get("currentPrice", 0)))
                    avg_price = float(pos.get("avgPrice", pos.get("averagePrice", 0)))

                    # Position value = size * current price
                    position_value += size * current_price

                    # Unrealized P&L = (current_price - avg_price) * size
                    if avg_price > 0:
                        unrealized_pnl += (current_price - avg_price) * size
        except Exception as e:
            print(f"  [Live Portfolio] Error fetching positions: {e}")

        # Try to get P&L data from value endpoint
        try:
            url = f"https://data-api.polymarket.com/value?user={wallet.lower()}"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    data = data[0]  # Value endpoint returns array
                realized_pnl = float(data.get("realizedPnl", data.get("realized_pnl", 0)))
        except Exception:
            pass

        total_value = usdc_balance + position_value

        return {
            "balance": usdc_balance,  # On-chain USDC balance (accurate!)
            "position_value": position_value,  # Value of open positions
            "total_value": total_value,  # USDC + positions
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "positions_count": positions_count,
        }

    def get_status(self) -> dict:
        """Get current ghost mode status."""
        uptime = 0
        if self.started_at:
            uptime = (datetime.utcnow() - self.started_at).total_seconds()

        avg_detection = sum(self.detection_latencies[-100:]) / len(self.detection_latencies[-100:]) if self.detection_latencies else 0
        avg_api = sum(self.api_latencies[-100:]) / len(self.api_latencies[-100:]) if self.api_latencies else 0
        true_lats = getattr(self, 'true_latencies', [])
        avg_true_latency = sum(true_lats[-100:]) / len(true_lats[-100:]) if true_lats else 0

        # In LIVE MODE: Fetch real portfolio data from our trading wallet
        if self.is_live_mode and self.is_live_ready():
            live_portfolio = self.get_live_portfolio()
            account_balance = live_portfolio.get("balance", 0) + live_portfolio.get("total_value", 0)
            account_pnl = live_portfolio.get("realized_pnl", 0) + live_portfolio.get("unrealized_pnl", 0)
            wallet_address = self.get_live_wallet_address()
        else:
            # GHOST MODE: Use simulated values
            account_balance = float(self.simulated_balance)
            account_pnl = float(self.simulated_pnl)
            wallet_address = None

        return {
            "ghost_mode": self.enabled and not self.is_live_mode,
            "live_mode": self.is_live_mode,
            "live_ready": self.is_live_ready(),
            "live_wallet": wallet_address,
            "status": "running" if self.enabled else "stopped",
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime": uptime,
            "accounts_tracked": len([a for a in self.accounts.values() if a.enabled]),
            "trades_detected": self.trades_detected,
            "trades_would_execute": self.trades_would_execute,
            "trades_filtered_keyword": self.trades_filtered_keyword,
            "trades_filtered_stoploss": self.trades_filtered_stoploss,
            "trades_filtered_slippage": self.trades_filtered_slippage,
            "trades_filtered_limit": self.trades_filtered_limit,
            "trades_missed_offline": self.trades_missed_offline,
            "api_calls_simulated": self.api_calls_simulated,
            "positions_count": len(self.positions),
            "missed_trades_count": len(self.missed_trades),
            # Account values - LIVE data when in live mode, simulated when in ghost mode
            "account_balance": account_balance,
            "account_pnl": account_pnl,
            # Legacy fields for backwards compatibility (but now point to real data in live mode)
            "simulated_balance": account_balance,
            "simulated_pnl": account_pnl,
            "avg_detection_ms": round(avg_detection, 2),
            "avg_api_ms": round(avg_api, 2),
            "avg_true_latency_ms": round(avg_true_latency, 2),  # Time from trade to detection (the important one!)
            # Blockchain monitoring stats
            "blockchain_enabled": self.blockchain_enabled,
            "blockchain_rpc_configured": self.polygon_rpc_url is not None,
            "blockchain_trades_detected": self.blockchain_trades_detected,
            "blockchain_avg_latency_ms": round(self._blockchain_monitor.avg_latency_ms, 1) if self._blockchain_monitor else 0,
        }

    def record_ghost_trade(self, trade: GhostTrade):
        """Record a ghost trade."""
        self.ghost_trades.insert(0, trade)
        if len(self.ghost_trades) > 500:
            self.ghost_trades.pop()

        self.trades_detected += 1

        if trade.status == 'would_execute' or trade.status == 'api_simulated':
            self.trades_would_execute += 1
        elif trade.status == 'filtered_keyword':
            self.trades_filtered_keyword += 1
        elif trade.status == 'filtered_stoploss':
            self.trades_filtered_stoploss += 1
        elif trade.status == 'filtered_slippage':
            self.trades_filtered_slippage += 1
        elif trade.status == 'filtered_limit':
            self.trades_filtered_limit += 1

        if trade.detection_ms > 0:
            self.detection_latencies.append(trade.detection_ms)
        if trade.true_latency_ms > 0:
            # Track true latency (trade execution to detection) - the important metric
            if not hasattr(self, 'true_latencies'):
                self.true_latencies = []
            self.true_latencies.append(trade.true_latency_ms)
        if trade.api_call_ms > 0:
            self.api_latencies.append(trade.api_call_ms)


# Global state
ghost_state = GhostModeState()


def get_orderbook_price(token_id: str, side: str) -> Optional[Decimal]:
    """
    Fetch the ACTUAL current best price from the orderbook.

    For BUY orders: returns best ASK (lowest price someone is selling at)
    For SELL orders: returns best BID (highest price someone is buying at)

    Returns None if orderbook is empty or API fails.
    """
    if not token_id:
        return None

    try:
        url = f"https://clob.polymarket.com/book?token_id={token_id}"
        resp = requests.get(url, timeout=3)

        if resp.status_code != 200:
            print(f"  [Orderbook] HTTP {resp.status_code} for {token_id[:16]}...")
            return None

        book = resp.json()

        if side == "BUY":
            # For buying, we need the best ASK (lowest ask price)
            asks = book.get("asks", [])
            if asks:
                # Sort by price ascending, get lowest
                best_ask = min(asks, key=lambda x: Decimal(str(x.get("price", "999"))))
                return Decimal(str(best_ask.get("price", "0")))
        else:
            # For selling, we need the best BID (highest bid price)
            bids = book.get("bids", [])
            if bids:
                # Sort by price descending, get highest
                best_bid = max(bids, key=lambda x: Decimal(str(x.get("price", "0"))))
                return Decimal(str(best_bid.get("price", "0")))

        return None

    except Exception as e:
        print(f"  [Orderbook] Error fetching for {token_id[:16]}...: {e}")
        return None


async def poll_account_activity(account: CopyTradeAccount, ws_broadcast) -> List[dict]:
    """Poll Polymarket API for new trades from an account."""

    url = f"https://data-api.polymarket.com/activity?user={account.wallet}&type=TRADE&limit=10"

    try:
        t_start = time.perf_counter()
        resp = requests.get(url, timeout=5)
        detection_ms = (time.perf_counter() - t_start) * 1000

        if resp.status_code != 200:
            return []

        trades = resp.json()
        if not trades:
            return []

        new_trades = []

        for trade in trades:
            # Create unique hash to prevent duplicates
            trade_hash = f"{trade.get('transactionHash')}_{trade.get('timestamp')}_{trade.get('asset')}"

            if ghost_state._account_manager.is_trade_seen(trade_hash):
                continue

            # Only process trades that happened AFTER ghost mode started
            if ghost_state.started_at:
                trade_ts = trade.get('timestamp', 0)
                # started_at is stored as UTC naive datetime, so use UTC timestamp
                # by converting to UTC-aware datetime first
                from datetime import timezone
                started_utc = ghost_state.started_at.replace(tzinfo=timezone.utc)
                started_ts = started_utc.timestamp()

                if trade_ts < started_ts:
                    # This is a retroactive trade - skip it
                    continue

            ghost_state._account_manager.add_seen_trade_hash(trade_hash)

            # Calculate TRUE latency: time from trade execution to now
            trade_ts = trade.get('timestamp', 0)
            if trade_ts > 0:
                true_latency_ms = (time.time() - trade_ts) * 1000
            else:
                true_latency_ms = 0

            new_trades.append({
                **trade,
                '_detection_ms': detection_ms,  # API call time
                '_true_latency_ms': true_latency_ms,  # Time from trade to detection
            })

        return new_trades

    except Exception as e:
        print(f"  [Error] Polling {account.name}: {e}")
        return []


async def process_new_trade(account: CopyTradeAccount, trade: dict, ws_broadcast):
    """Process a new trade detected from a tracked account (ghost mode simulation)."""

    t_total_start = time.perf_counter()
    detection_ms = trade.get('_detection_ms', 0)
    true_latency_ms = trade.get('_true_latency_ms', 0)

    market_name = trade.get('title', 'Unknown Market')
    market_id = trade.get('conditionId', '')
    outcome = trade.get('outcome', '')
    side = trade.get('side', 'BUY')
    size = Decimal(str(trade.get('size', 0)))
    price = Decimal(str(trade.get('price', 0)))
    usdc_size = Decimal(str(trade.get('usdcSize', 0)))

    # Calculate our position size
    our_size = min(
        usdc_size * account.position_ratio,
        account.max_position_usd,
    )
    our_shares = our_size / price if price > 0 else Decimal(0)

    # Create ghost trade record
    ghost_trade = GhostTrade(
        id=f"ghost_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.utcnow(),
        account_name=account.name,
        market_id=market_id,
        market_name=market_name,
        outcome=outcome,
        side=side,
        target_size=size,
        target_price=price,
        our_size=our_shares,
        our_price=price,
        status='pending',
        detection_ms=detection_ms,
        true_latency_ms=true_latency_ms,
    )

    # Check keyword filter
    if not account.matches_keywords(market_name):
        ghost_trade.status = 'filtered_keyword'
        ghost_trade.total_ms = (time.perf_counter() - t_total_start) * 1000
        ghost_state.record_ghost_trade(ghost_trade)

        await ws_broadcast({
            "type": "ghost_trade",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                **ghost_trade.to_dict(),
                "filter_reason": f"No keyword match in '{market_name[:50]}...'",
            },
        })
        return

    # Check stoploss
    if account.stoploss_triggered:
        ghost_trade.status = 'filtered_stoploss'
        ghost_trade.total_ms = (time.perf_counter() - t_total_start) * 1000
        ghost_state.record_ghost_trade(ghost_trade)

        await ws_broadcast({
            "type": "ghost_trade",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                **ghost_trade.to_dict(),
                "filter_reason": "Stoploss triggered - not opening new positions",
            },
        })
        return

    # ===================================================================
    # PRICE CHECK - Fetch REAL orderbook price for accurate execution
    # ===================================================================
    # The token_id (asset) is what we need to look up the orderbook
    token_id = trade.get('asset', '')

    # Fetch the ACTUAL current best price from the orderbook
    current_market_price = get_orderbook_price(token_id, side)

    # If we couldn't get orderbook price, use target price with a warning
    if current_market_price is None:
        print(f"  [WARNING] Could not fetch orderbook for {token_id[:16]}..., using target price")
        current_market_price = price

    # Calculate actual slippage from target price to current market price
    if side == "BUY":
        # For BUY: slippage = (current_price - target_price) / target_price
        # Positive slippage means price went UP (bad for us)
        actual_slippage = (current_market_price - price) / price if price > 0 else Decimal(0)
    else:
        # For SELL: slippage = (target_price - current_price) / target_price
        # Positive slippage means price went DOWN (bad for us)
        actual_slippage = (price - current_market_price) / price if price > 0 else Decimal(0)

    ghost_trade.our_price = current_market_price

    # ===================================================================
    # LIMIT MODE: MUST get target price or BETTER, otherwise SKIP
    # ===================================================================
    if account.order_type == 'limit':
        # For LIMIT orders: We ONLY execute if we can get the same price or better
        # BUY: current market price must be <= target price (same or cheaper)
        # SELL: current market price must be >= target price (same or higher)
        price_ok = (side == "BUY" and current_market_price <= price) or \
                   (side == "SELL" and current_market_price >= price)

        if not price_ok:
            ghost_trade.status = 'filtered_limit'
            ghost_trade.actual_slippage_pct = actual_slippage
            ghost_trade.total_ms = (time.perf_counter() - t_total_start) * 1000
            ghost_state.record_ghost_trade(ghost_trade)

            direction = "higher" if side == "BUY" else "lower"
            slippage_display = float(abs(actual_slippage) * 100)
            await ws_broadcast({
                "type": "ghost_trade",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    **ghost_trade.to_dict(),
                    "filter_reason": f"LIMIT: Price moved {direction} by {slippage_display:.2f}% (market: {float(current_market_price):.4f} vs target: {float(price):.4f})",
                },
            })

            print(f"  [LIMIT SKIP] Price {direction}: market {float(current_market_price):.4f} vs target {float(price):.4f} ({slippage_display:.2f}% worse) - {market_name[:35]}")
            return

        # LIMIT order is good - we can get target price or better!
        ghost_trade.actual_slippage_pct = actual_slippage
        ghost_trade.max_allowed_slippage_pct = Decimal(0)  # N/A for limit orders

    # ===================================================================
    # MARKET MODE: Accept current price with slippage tolerance check
    # ===================================================================
    else:
        # For MARKET orders: Check if slippage is within acceptable range
        is_acceptable, _, max_allowed = account.is_slippage_acceptable(price, current_market_price)
        ghost_trade.actual_slippage_pct = actual_slippage
        ghost_trade.max_allowed_slippage_pct = max_allowed

        if not is_acceptable:
            ghost_trade.status = 'filtered_slippage'
            ghost_trade.total_ms = (time.perf_counter() - t_total_start) * 1000
            ghost_state.record_ghost_trade(ghost_trade)

            slippage_pct_display = float(abs(actual_slippage) * 100)
            max_pct_display = float(max_allowed * 100)

            await ws_broadcast({
                "type": "ghost_trade",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    **ghost_trade.to_dict(),
                    "filter_reason": f"MARKET: Slippage {slippage_pct_display:.1f}% > {max_pct_display:.0f}% max (market: {float(current_market_price):.4f} vs target: {float(price):.4f})",
                },
            })

            print(f"  [MARKET SLIP] {slippage_pct_display:.1f}% > {max_pct_display:.0f}% max @ {float(price)*100:.1f}¢ - {market_name[:40]}")
            return

    # EXECUTION - Live or Ghost mode
    t_api_start = time.perf_counter()

    # Store the order type from account settings
    ghost_trade.order_type = account.order_type

    # Build order params for audit trail
    order_params = {
        "tokenID": trade.get('asset', ''),
        "side": side,
        "price": str(current_market_price),
        "size": str(our_shares),
        "type": "GTC",
        "_audit": {
            "our_order_mode": account.order_type,
            "target_price": str(price),
            "execution_price": str(current_market_price),
            "slippage_pct": str(float(actual_slippage * 100)) if actual_slippage else "0",
            "position_ratio": str(account.position_ratio),
            "max_position_usd": str(account.max_position_usd),
            "triggered_by": f"{account.name} trade",
        }
    }
    ghost_trade.order_params = order_params

    # Check if we're in LIVE MODE with proper auth
    if ghost_state.is_live_mode and ghost_state.is_live_ready():
        # LIVE EXECUTION - Use real auth to place order
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType

            # Initialize client with auth
            client = ClobClient(
                host="https://clob.polymarket.com",
                key=ghost_state._private_key,
                chain_id=137,  # Polygon mainnet
            )

            # Build order args
            order_args = OrderArgs(
                token_id=trade.get('asset', ''),
                price=float(current_market_price),
                size=float(our_shares),
                side=side,
            )

            # Create and submit the order
            signed_order = client.create_order(order_args)
            result = client.post_order(signed_order, OrderType.GTC)

            ghost_trade.api_response = {
                "status": "submitted",
                "order_id": result.get("orderID", "unknown"),
                "body": str(result)[:500],
            }
            ghost_trade.api_error = None
            ghost_trade.status = 'live_executed'

            print(f"  \033[32m[LIVE ORDER]\033[0m {side} ${float(our_size):.2f} @ {float(current_market_price):.4f} - {market_name[:40]}")

        except ImportError:
            ghost_trade.api_error = "py_clob_client not installed"
            ghost_trade.api_response = {"status": "error", "body": "Missing py_clob_client library"}
            ghost_trade.status = 'live_error'
            print(f"  \033[31m[LIVE ERROR]\033[0m py_clob_client not installed")
        except Exception as e:
            ghost_trade.api_error = f"Live execution error: {str(e)[:200]}"
            ghost_trade.api_response = {"status": "error", "body": str(e)[:500]}
            ghost_trade.status = 'live_error'
            print(f"  \033[31m[LIVE ERROR]\033[0m {str(e)[:80]}")

    else:
        # GHOST EXECUTION - Simulate API call (will fail but shows timing)
        GHOST_FUNDER_ADDRESS = "0x0000000000000000000000000000000000000000"
        order_params["funderAddress"] = GHOST_FUNDER_ADDRESS

        try:
            api_url = "https://clob.polymarket.com/order"
            api_resp = requests.post(
                api_url,
                json=order_params,
                headers={"Content-Type": "application/json"},
                timeout=2,
            )
            ghost_trade.api_response = {
                "status": api_resp.status_code,
                "body": api_resp.text[:500],
                "headers": dict(api_resp.headers),
            }
            ghost_trade.api_error = f"HTTP {api_resp.status_code} (expected - ghost mode)"
        except requests.exceptions.Timeout:
            ghost_trade.api_error = "Timeout (2s limit)"
            ghost_trade.api_response = {"status": "timeout", "body": "Request timed out"}
        except requests.exceptions.ConnectionError as e:
            ghost_trade.api_error = f"Connection error: {str(e)[:80]}"
            ghost_trade.api_response = {"status": "connection_error", "body": str(e)[:200]}
        except Exception as e:
            ghost_trade.api_error = str(e)[:100]
            ghost_trade.api_response = {"status": "error", "body": str(e)[:200]}

        ghost_trade.status = 'api_simulated'

    ghost_state.api_calls_simulated += 1
    ghost_trade.api_call_ms = (time.perf_counter() - t_api_start) * 1000
    ghost_trade.total_ms = (time.perf_counter() - t_total_start) * 1000

    # Update simulated position
    pos_key = f"{market_id}:{outcome}"
    if pos_key not in ghost_state.positions:
        ghost_state.positions[pos_key] = {
            "market_id": market_id,
            "market_name": market_name,
            "outcome": outcome,
            "size": Decimal(0),
            "avg_price": Decimal(0),
            "cost_basis": Decimal(0),
            "status": "open",
        }

    pos = ghost_state.positions[pos_key]
    if side == "BUY":
        # Deduct cost from simulated balance (at actual market price)
        cost = our_shares * current_market_price
        ghost_state.simulated_balance -= cost

        new_cost = pos["cost_basis"] + cost
        new_size = pos["size"] + our_shares
        pos["avg_price"] = new_cost / new_size if new_size > 0 else Decimal(0)
        pos["size"] = new_size
        pos["cost_basis"] = new_cost
        # Re-open position if it was closed and we're buying again
        if pos["status"] == "closed":
            pos["status"] = "open"
    else:
        # SELL: reduce position size and realize P/L
        if pos["size"] == Decimal(0):
            # Don't track sells on positions we never had (retroactive close)
            # This happens when trader is closing a position they had before monitoring started
            pass
        else:
            # Calculate shares to sell (can't sell more than we have)
            shares_to_sell = min(our_shares, pos["size"])

            # Calculate proceeds from sale (at actual market price)
            proceeds = shares_to_sell * current_market_price
            ghost_state.simulated_balance += proceeds

            # Calculate realized P/L for this sale
            cost_of_sold_shares = shares_to_sell * pos["avg_price"]
            realized_pnl = proceeds - cost_of_sold_shares
            ghost_state.simulated_pnl += realized_pnl

            # Update position
            pos["size"] = max(Decimal(0), pos["size"] - shares_to_sell)
            if pos["size"] == 0:
                pos["status"] = "closed"
                pos["cost_basis"] = Decimal(0)  # Reset cost basis for closed positions

    ghost_state.record_ghost_trade(ghost_trade)

    # Broadcast ghost trade to UI (single broadcast to avoid duplicate notifications)
    await ws_broadcast({
        "type": "ghost_trade",
        "timestamp": datetime.utcnow().isoformat(),
        "data": ghost_trade.to_dict(),
    })

    # Update positions
    open_positions = [
        {**p, "size": float(p["size"]), "avg_price": float(p["avg_price"]), "cost_basis": float(p["cost_basis"])}
        for p in ghost_state.positions.values()
        if p["status"] == "open"
    ]
    await ws_broadcast({
        "type": "position_update",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "positions": open_positions,
            "count": len(open_positions),
            "ghost_mode": True,
        },
    })

    # Log with full order details
    side_color = "\033[32m" if side == "BUY" else "\033[31m"
    order_type_str = account.order_type.upper()
    order_type_color = "\033[33m" if account.order_type == "limit" else "\033[36m"  # Yellow for limit, cyan for market
    latency_str = f"{true_latency_ms/1000:.1f}s" if true_latency_ms > 0 else "??"
    api_status = ghost_trade.api_response.get("status", "?") if ghost_trade.api_response else "?"
    slip_str = f"{float(actual_slippage*100):+.2f}%" if actual_slippage else "0%"
    print(f"  [GHOST] {side_color}{side}\033[0m {order_type_color}[{order_type_str}]\033[0m ${float(our_size):.2f} target:{float(price):.4f} exec:{float(current_market_price):.4f} slip:{slip_str} | lat:{latency_str} HTTP:{api_status} - {market_name[:30]}")


async def _handle_blockchain_trade(blockchain_trade: 'BlockchainTrade', ws_broadcast):
    """
    Handle a trade detected from blockchain monitoring.

    This is called with ~2-5s latency (vs ~15-21s with API polling).
    We convert the blockchain trade format to our internal format and process it.
    """
    if not BLOCKCHAIN_AVAILABLE or blockchain_trade is None:
        return

    ghost_state.blockchain_trades_detected += 1

    # Find which account this wallet belongs to
    wallet_lower = blockchain_trade.wallet.lower()
    account = None
    for acc in ghost_state.accounts.values():
        if acc.wallet.lower() == wallet_lower and acc.enabled:
            account = acc
            break

    if not account:
        print(f"  [Blockchain] Trade from unknown wallet: {wallet_lower[:16]}...")
        return

    # Convert blockchain trade to our internal format
    # Note: We may not have full market info from blockchain, so we'll
    # enrich it via API if needed
    trade_data = {
        'transactionHash': blockchain_trade.tx_hash,
        'timestamp': int(blockchain_trade.timestamp.timestamp()),
        'asset': blockchain_trade.token_id,
        'side': blockchain_trade.side,
        'price': float(blockchain_trade.price),
        'size': float(blockchain_trade.size),
        'usdcSize': float(blockchain_trade.size * blockchain_trade.price),
        'title': f"Blockchain trade (token: {blockchain_trade.token_id[:16]}...)",
        '_blockchain_detected': True,
        '_blockchain_latency_ms': blockchain_trade.detection_latency_ms,
    }

    # Create a unique hash for this trade
    trade_hash = f"{blockchain_trade.tx_hash}_{trade_data['timestamp']}_{trade_data['asset']}"

    # Check if we've already processed this trade via API polling
    if trade_hash in ghost_state.seen_trade_hashes:
        # Already processed, but log the latency improvement
        print(f"  [Blockchain] Trade already seen via API (blockchain was {blockchain_trade.detection_latency_ms:.0f}ms faster)")
        return

    # Mark as seen
    ghost_state._account_manager.add_seen_trade_hash(trade_hash)

    print(f"  [BLOCKCHAIN] \033[35mFAST DETECT\033[0m ({blockchain_trade.detection_latency_ms:.0f}ms) "
          f"{blockchain_trade.side} from {account.name}")

    # Broadcast blockchain detection event
    await ws_broadcast({
        "type": "blockchain_trade",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "tx_hash": blockchain_trade.tx_hash,
            "wallet": blockchain_trade.wallet,
            "side": blockchain_trade.side,
            "latency_ms": blockchain_trade.detection_latency_ms,
            "account_name": account.name,
        },
    })

    # Process the trade through normal flow
    await process_new_trade(account, trade_data, ws_broadcast)


async def ghost_mode_monitor(ws_broadcast):
    """Main monitoring loop for Ghost Mode."""

    print("\n" + "="*70)
    print("  GHOST MODE - Real-time Copy Trade Simulation")
    print("="*70)
    print("  Monitoring REAL trades from tracked accounts")
    print("  API calls will be simulated (no live execution)")
    print("  Only NEW trades after start will be processed (no retroactive)")

    # Initialize blockchain monitoring if configured
    if ghost_state.polygon_rpc_url and BLOCKCHAIN_AVAILABLE:
        print(f"\n  [BLOCKCHAIN MODE] Polygon RPC configured - enabling ~2-5s latency!")
        print(f"  RPC: {ghost_state.polygon_rpc_url[:50]}...")

        # Collect all wallet addresses to monitor
        wallets = [acc.wallet for acc in ghost_state.accounts.values() if acc.enabled]

        # Create blockchain monitor
        ghost_state._blockchain_monitor = PolygonMonitor(
            rpc_url=ghost_state.polygon_rpc_url,
            wallets=wallets,
            on_trade=lambda trade: _handle_blockchain_trade(trade, ws_broadcast),
        )

        # Start the monitor
        success = await ghost_state._blockchain_monitor.start()
        ghost_state.blockchain_enabled = success

        if success:
            print("  [BLOCKCHAIN] Connected! Monitoring on-chain events...")
        else:
            print("  [BLOCKCHAIN] Failed to connect, falling back to API polling only")
    elif not BLOCKCHAIN_AVAILABLE:
        print("\n  [INFO] Blockchain monitoring not available (web3 not installed)")
        print("  API polling only (~15-21s latency)")
    else:
        print("\n  [INFO] No POLYGON_RPC_URL configured")
        print("  Set POLYGON_RPC_URL env var for ~2-5s latency (vs ~15-21s with API)")

    print("="*70 + "\n")

    poll_interval = 1.0  # Poll every 1 second for faster detection

    while ghost_state.enabled:
        try:
            for account in ghost_state.accounts.values():
                if not account.enabled:
                    continue

                # Poll for new trades
                new_trades = await poll_account_activity(account, ws_broadcast)

                for trade in new_trades:
                    await process_new_trade(account, trade, ws_broadcast)

                # Minimal delay between accounts (just to avoid hammering)
                await asyncio.sleep(0.05)

            # Broadcast status update
            await ws_broadcast({
                "type": "status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": ghost_state.get_status(),
            })

            # Broadcast latency update
            await ws_broadcast({
                "type": "latency_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "stages": {
                        "detection": {
                            "count": len(ghost_state.detection_latencies),
                            "avg_ms": sum(ghost_state.detection_latencies[-50:]) / len(ghost_state.detection_latencies[-50:]) if ghost_state.detection_latencies else 0,
                            "min_ms": min(ghost_state.detection_latencies[-50:]) if ghost_state.detection_latencies else 0,
                            "max_ms": max(ghost_state.detection_latencies[-50:]) if ghost_state.detection_latencies else 0,
                        },
                        "e2e": {
                            "count": len(ghost_state.api_latencies),
                            "avg_ms": sum(ghost_state.api_latencies[-50:]) / len(ghost_state.api_latencies[-50:]) if ghost_state.api_latencies else 0,
                            "min_ms": min(ghost_state.api_latencies[-50:]) if ghost_state.api_latencies else 0,
                            "max_ms": max(ghost_state.api_latencies[-50:]) if ghost_state.api_latencies else 0,
                        },
                    },
                    "health": {"status": "healthy", "health_score": 100},
                },
            })

            await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [Monitor Error] {e}")
            await asyncio.sleep(5)

    # Cleanup blockchain monitor if running
    if ghost_state._blockchain_monitor:
        await ghost_state._blockchain_monitor.stop()
        ghost_state._blockchain_monitor = None
        ghost_state.blockchain_enabled = False

    print("\n  [Ghost Mode] Stopped")


if __name__ == "__main__":
    # Test
    print("Ghost Mode State:")
    print(json.dumps(ghost_state.get_status(), indent=2))
    print("\nAccounts:")
    for acc in ghost_state.accounts.values():
        print(f"  - {acc.name}: {acc.wallet[:10]}... keywords={acc.keywords}")
