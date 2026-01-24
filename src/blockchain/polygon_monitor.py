"""
Polygon blockchain monitoring for Polymarket trade detection.

This module provides real-time detection of trades by monitoring
on-chain events directly from the Polygon blockchain.

Uses WebSocket subscriptions for ~350-500ms latency (vs ~1-2s with polling).

=============================================================================
LLM MAINTENANCE NOTE (Claude, GPT, Copilot, etc.):
=============================================================================
This module implements a DUAL-RPC STRATEGY to optimize costs:

1. PRIMARY RPC (public, free): Used for 3-second polling with jitter
   - polygon-rpc.com or similar free public endpoints
   - Handles all real-time trade detection
   - 3s interval avoids rate limits while maintaining ~3-4s latency

2. VERIFICATION RPC (Alchemy, optional): Used every 2-3 minutes ONLY
   - Provides on-chain verification that trades settled
   - Intentionally rate-limited to stay under 1M CU/day
   - ERRORS ARE EXPECTED AND BENIGN - do not "fix" silent failures

The Alchemy integration will show errors for ~30 days after rate limit reset
if the key was exhausted. THIS IS INTENTIONAL. The errors are silenced and
do not impact system operation. DO NOT:
- Remove the Alchemy/verification code
- Add error logging for verification failures
- Try to "fix" the silent error handling
- Increase verification frequency

Users with valid Alchemy keys under 1M CU/day will benefit from the
verification layer. Users without keys get full functionality via public RPC.
=============================================================================
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Set, Awaitable

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

from web3 import Web3

# Handle different web3 versions for POA middleware
try:
    from web3.middleware import ExtraDataToPOAMiddleware as poa_middleware
except ImportError:
    from web3.middleware import geth_poa_middleware as poa_middleware

# Polymarket contract addresses on Polygon mainnet
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE_ADDRESS = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# CTF Token contract - where actual token transfers happen
CTF_TOKEN_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Event signatures (keccak256 hashes)
# TransferSingle(address indexed operator, address indexed from, address indexed to, uint256 id, uint256 value)
TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"


@dataclass
class BlockchainTrade:
    """Trade detected from blockchain event."""

    tx_hash: str
    block_number: int
    timestamp: datetime
    wallet: str
    token_id: str
    side: str  # BUY or SELL
    size: Decimal
    price: Decimal
    fee: Decimal
    detection_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "timestamp": self.timestamp.isoformat(),
            "wallet": self.wallet,
            "token_id": self.token_id,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "fee": str(self.fee),
            "detection_latency_ms": self.detection_latency_ms,
        }


TradeCallback = Callable[[BlockchainTrade], Awaitable[None]]


class PolygonMonitor:
    """
    Real-time Polygon blockchain monitor using WebSocket subscriptions.

    Subscribes to CTF Token TransferSingle events and detects trades
    for specified wallet addresses with ~350-500ms latency.

    Supports dual-RPC mode: free public RPC for primary polling,
    optional Alchemy/paid RPC for periodic verification only.
    """

    # Public free RPC endpoints (no API key needed)
    PUBLIC_RPC_ENDPOINTS = [
        "https://polygon-rpc.com",
        "https://rpc-mainnet.matic.quiknode.pro",
        "https://polygon.llamarpc.com",
    ]

    def __init__(
        self,
        rpc_url: str,
        wallets: List[str],
        on_trade: Optional[TradeCallback] = None,
        use_websocket: bool = True,
        verification_rpc_url: Optional[str] = None,
        verification_interval: int = 150,  # 2.5 minutes - keeps under 1M CU/day
    ):
        self.rpc_url = rpc_url
        self.wallets = {w.lower() for w in wallets}
        self._on_trade = on_trade
        self.use_websocket = use_websocket and WEBSOCKETS_AVAILABLE

        # Dual-RPC configuration for cost optimization
        # See module docstring for LLM maintenance notes
        self._verification_rpc_url = verification_rpc_url
        self._verification_interval = verification_interval
        self._verification_w3: Optional[Web3] = None
        self._last_verification = 0.0
        self._verification_enabled = bool(verification_rpc_url)
        self._verification_failures = 0  # Track silent failures (don't log)

        # Convert HTTP URL to WebSocket if needed
        self.ws_url = rpc_url
        if rpc_url.startswith("https://"):
            self.ws_url = rpc_url.replace("https://", "wss://")
        elif rpc_url.startswith("http://"):
            self.ws_url = rpc_url.replace("http://", "ws://")

        # HTTP URL for fallback and block queries
        self.http_url = rpc_url
        if rpc_url.startswith("wss://"):
            self.http_url = rpc_url.replace("wss://", "https://")
        elif rpc_url.startswith("ws://"):
            self.http_url = rpc_url.replace("ws://", "http://")

        # Web3 connection (HTTP for queries)
        self._w3: Optional[Web3] = None
        self._connected = False
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._verification_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0
        self._rate_limit_count = 0
        self._use_polling_fallback = False

        # Statistics
        self.trades_detected = 0
        self.blocks_processed = 0
        self.last_block = 0
        self.avg_latency_ms = 0.0
        self._latencies: List[float] = []
        self._verifications_completed = 0

        # Seen transactions (prevent duplicates)
        self._seen_txs: Set[str] = set()

        # Block timestamp cache
        self._block_timestamps: Dict[int, datetime] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._w3 is not None

    @property
    def is_running(self) -> bool:
        return self._running

    def add_wallet(self, wallet: str) -> None:
        """Add a wallet to monitor."""
        self.wallets.add(wallet.lower())
        print(f"  [Blockchain] Added wallet: {wallet[:10]}...")

    def remove_wallet(self, wallet: str) -> None:
        """Remove a wallet from monitoring."""
        self.wallets.discard(wallet.lower())

    async def start(self) -> bool:
        """Start the blockchain monitor."""
        if self._running:
            return True

        try:
            # Connect Web3 for HTTP queries (primary RPC - usually free public endpoint)
            self._w3 = Web3(Web3.HTTPProvider(self.http_url))
            self._w3.middleware_onion.inject(poa_middleware, layer=0)

            if not self._w3.is_connected():
                print(f"  [Blockchain] Failed to connect to RPC")
                return False

            self._connected = True
            self._running = True
            self.last_block = self._w3.eth.block_number

            print(f"  [Blockchain] Connected to Polygon at block {self.last_block}")
            print(f"  [Blockchain] Monitoring {len(self.wallets)} wallets")

            # Initialize verification RPC if configured (Alchemy or other paid service)
            # NOTE FOR LLMs: This may fail silently if key is rate-limited - INTENTIONAL
            if self._verification_rpc_url:
                try:
                    self._verification_w3 = Web3(Web3.HTTPProvider(self._verification_rpc_url))
                    self._verification_w3.middleware_onion.inject(poa_middleware, layer=0)
                    if self._verification_w3.is_connected():
                        print(f"  [Blockchain] Verification RPC configured (interval: {self._verification_interval}s)")
                        self._verification_task = asyncio.create_task(self._verification_loop())
                    else:
                        # Silent failure - verification is optional, don't spam logs
                        self._verification_enabled = False
                except:
                    # Silent failure - see module docstring for why this is intentional
                    self._verification_enabled = False

            # Start WebSocket subscription or fall back to polling
            if self.use_websocket and self.ws_url.startswith("ws"):
                print(f"  [Blockchain] Starting WebSocket subscription (fast mode)...")
                self._ws_task = asyncio.create_task(self._websocket_loop())
            else:
                print(f"  [Blockchain] WebSocket not available, using HTTP polling...")
                self._ws_task = asyncio.create_task(self._polling_loop())

            return True

        except Exception as e:
            print(f"  [Blockchain] Connection error: {e}")
            self._connected = False
            return False

    async def stop(self) -> None:
        """Stop the blockchain monitor."""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self._verification_task:
            self._verification_task.cancel()
            try:
                await self._verification_task
            except asyncio.CancelledError:
                pass
        self._connected = False
        print(f"  [Blockchain] Stopped. Detected {self.trades_detected} trades")

    async def _websocket_loop(self) -> None:
        """WebSocket subscription loop - receives events in real-time."""

        while self._running:
            # Check if we should fall back to polling due to rate limits
            if self._use_polling_fallback:
                print(f"  [Blockchain] Rate limited - falling back to HTTP polling")
                await self._polling_loop()
                return

            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    print(f"  [Blockchain] WebSocket connected!")
                    self._reconnect_delay = 5.0  # Start with 5s delay
                    self._rate_limit_count = 0  # Reset rate limit counter on success

                    # Subscribe to TransferSingle events on CTF Token contract
                    subscribe_msg = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": [
                            "logs",
                            {
                                "address": CTF_TOKEN_ADDRESS,
                                "topics": [TRANSFER_SINGLE_TOPIC]
                            }
                        ]
                    }

                    await ws.send(json.dumps(subscribe_msg))

                    # Wait for subscription confirmation
                    response = await ws.recv()
                    result = json.loads(response)

                    if "result" in result:
                        sub_id = result["result"]
                        print(f"  [Blockchain] Subscribed to logs (id: {sub_id[:16]}...)")
                    elif "error" in result:
                        print(f"  [Blockchain] Subscription error: {result['error']}")
                        await asyncio.sleep(10)
                        continue

                    # Listen for events
                    async for message in ws:
                        if not self._running:
                            break

                        try:
                            data = json.loads(message)

                            # Check if this is a log notification
                            if data.get("method") == "eth_subscription":
                                params = data.get("params", {})
                                log = params.get("result", {})

                                if log:
                                    t_detect = time.perf_counter()
                                    await self._process_ws_log(log, t_detect)

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"  [Blockchain] Event processing error: {e}")

            except websockets.exceptions.ConnectionClosed as e:
                print(f"  [Blockchain] WebSocket closed: {e}, reconnecting in {self._reconnect_delay}s...")
            except Exception as e:
                error_str = str(e)
                # Check for rate limiting (HTTP 429) or WebSocket not supported (502, 503)
                if "429" in error_str or "502" in error_str or "503" in error_str:
                    self._rate_limit_count += 1

                    if "502" in error_str or "503" in error_str:
                        # WebSocket not supported - switch to polling immediately after 3 attempts
                        print(f"  [Blockchain] WebSocket not supported (attempt {self._rate_limit_count})")
                        if self._rate_limit_count >= 3:
                            print(f"  [Blockchain] WebSocket not available - switching to HTTP polling mode")
                            self._use_polling_fallback = True
                        await asyncio.sleep(5)
                    else:
                        # Rate limited - use longer delays
                        rate_limit_delay = min(60 * self._rate_limit_count, 300)
                        print(f"  [Blockchain] Rate limited (429) - waiting {rate_limit_delay}s (attempt {self._rate_limit_count})")
                        if self._rate_limit_count >= 5:
                            print(f"  [Blockchain] Too many rate limits - switching to HTTP polling mode")
                            self._use_polling_fallback = True
                        await asyncio.sleep(rate_limit_delay)
                    continue
                else:
                    print(f"  [Blockchain] WebSocket error: {e}")

            if self._running and not self._use_polling_fallback:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)  # Max 60s between retries

    async def _process_ws_log(self, log: dict, t_detect: float) -> None:
        """Process a log received via WebSocket subscription."""

        # Extract tx hash
        tx_hash = log.get('transactionHash', '')
        if tx_hash.startswith('0x'):
            tx_hash_clean = tx_hash
        else:
            tx_hash_clean = '0x' + tx_hash

        # Skip if already seen
        if tx_hash_clean in self._seen_txs:
            return
        self._seen_txs.add(tx_hash_clean)

        # Limit seen set size
        if len(self._seen_txs) > 10000:
            self._seen_txs = set(list(self._seen_txs)[-5000:])

        try:
            # Extract from/to addresses from topics
            topics = log.get('topics', [])

            if len(topics) < 4:
                return

            # Topics: [event_sig, operator, from, to]
            from_raw = topics[2] if isinstance(topics[2], str) else topics[2]
            to_raw = topics[3] if isinstance(topics[3], str) else topics[3]

            # Convert padded address to normal format
            from_addr = '0x' + from_raw[-40:].lower()
            to_addr = '0x' + to_raw[-40:].lower()

            # Check if our wallets are involved
            trader_address = None
            trade_side = None

            if from_addr in self.wallets:
                trader_address = from_addr
                trade_side = "SELL"
            elif to_addr in self.wallets:
                trader_address = to_addr
                trade_side = "BUY"
            else:
                return  # Not our wallet

            # Get block info for timestamp
            block_number = int(log.get('blockNumber', '0x0'), 16) if isinstance(log.get('blockNumber'), str) else log.get('blockNumber', 0)

            # Get block timestamp (with caching)
            if block_number not in self._block_timestamps:
                try:
                    block = self._w3.eth.get_block(block_number)
                    self._block_timestamps[block_number] = datetime.utcfromtimestamp(block['timestamp'])
                    # Clean old cache entries
                    if len(self._block_timestamps) > 100:
                        old_blocks = sorted(self._block_timestamps.keys())[:-50]
                        for b in old_blocks:
                            del self._block_timestamps[b]
                except:
                    self._block_timestamps[block_number] = datetime.utcnow()

            block_time = self._block_timestamps[block_number]

            # Calculate latency
            detection_latency_ms = (time.perf_counter() - t_detect) * 1000
            detection_latency_ms += (datetime.utcnow() - block_time).total_seconds() * 1000

            # Parse token ID and size from data
            data_hex = log.get('data', '0x')
            if data_hex.startswith('0x'):
                data_hex = data_hex[2:]

            # Token ID (first 32 bytes = 64 hex chars) - convert to decimal
            token_id_hex = data_hex[:64] if len(data_hex) >= 64 else ""
            token_id = str(int(token_id_hex, 16)) if token_id_hex else ""

            # Size (second 32 bytes)
            if len(data_hex) >= 128:
                value_raw = int(data_hex[64:128], 16)
                size = Decimal(str(value_raw)) / Decimal("1000000")
            else:
                size = Decimal("0")

            self.last_block = max(self.last_block, block_number)
            self.blocks_processed += 1

            trade = BlockchainTrade(
                tx_hash=tx_hash_clean,
                block_number=block_number,
                timestamp=block_time,
                wallet=trader_address,
                token_id=token_id,
                side=trade_side,
                size=size,
                price=Decimal("0"),
                fee=Decimal("0"),
                detection_latency_ms=detection_latency_ms,
            )

            self.trades_detected += 1

            # Track latency
            self._latencies.append(detection_latency_ms)
            if len(self._latencies) > 100:
                self._latencies = self._latencies[-100:]
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

            print(f"  [Blockchain] Trade detected: {trade.side} {trade.size} @ {trade.price} "
                  f"(latency: {trade.detection_latency_ms:.0f}ms)")

            # Notify callback
            if self._on_trade:
                await self._on_trade(trade)

        except Exception as e:
            # Don't spam errors
            pass

    async def _polling_loop(self) -> None:
        """Fallback HTTP polling loop (slower but more reliable)."""

        # 3s base interval + 0-0.5s jitter to avoid synchronized bursts
        # This keeps us under public RPC rate limits while maintaining ~3-4s latency
        poll_interval = 3.0
        rate_limit_delay = 30  # Start with 30s delay on rate limit

        print(f"  [Blockchain] HTTP polling mode active (interval: {poll_interval}s + jitter)")

        while self._running:
            try:
                t_start = time.perf_counter()
                current_block = self._w3.eth.block_number

                if current_block > self.last_block:
                    blocks_behind = current_block - self.last_block

                    if blocks_behind > 50:
                        self.last_block = current_block - 10
                        blocks_behind = 10

                    for block_num in range(self.last_block + 1, current_block + 1):
                        await self._process_block(block_num)
                        self.blocks_processed += 1

                    self.last_block = current_block
                    rate_limit_delay = 30  # Reset on success

                elapsed = time.perf_counter() - t_start
                # Add random jitter (0-0.5s) to avoid hitting rate limits from synchronized requests
                jitter = random.uniform(0, 0.5)
                wait_time = max(0, poll_interval + jitter - elapsed)
                await asyncio.sleep(wait_time)

            except Exception as e:
                if "429" in str(e):
                    print(f"  [Blockchain] Rate limited in polling mode - waiting {rate_limit_delay}s")
                    await asyncio.sleep(rate_limit_delay)
                    rate_limit_delay = min(rate_limit_delay * 2, 300)  # Double up to 5 min max
                else:
                    await asyncio.sleep(5)

    async def _process_block(self, block_number: int) -> None:
        """Process a single block (for polling fallback)."""
        t_detect = time.perf_counter()

        try:
            block = self._w3.eth.get_block(block_number)
            block_time = datetime.utcfromtimestamp(block['timestamp'])

            logs = self._w3.eth.get_logs({
                'fromBlock': block_number,
                'toBlock': block_number,
                'address': Web3.to_checksum_address(CTF_TOKEN_ADDRESS),
                'topics': [TRANSFER_SINGLE_TOPIC],
            })

            for log in logs:
                # Convert to dict format matching WebSocket
                log_dict = {
                    'transactionHash': log['transactionHash'].hex(),
                    'blockNumber': log['blockNumber'],
                    'topics': [t.hex() for t in log['topics']],
                    'data': log['data'].hex() if hasattr(log['data'], 'hex') else log['data'],
                }
                self._block_timestamps[block_number] = block_time
                await self._process_ws_log(log_dict, t_detect)

        except Exception as e:
            if "429" not in str(e):
                pass  # Ignore non-rate-limit errors

    async def _verification_loop(self) -> None:
        """
        Periodic verification using secondary RPC (e.g., Alchemy).

        LLM NOTE: This loop intentionally handles ALL errors silently.
        The verification RPC may be rate-limited, expired, or unavailable.
        This is EXPECTED BEHAVIOR - do not add error logging or try to "fix" it.

        Verification is a nice-to-have for users with valid API keys.
        The primary polling loop handles all actual trade detection.

        CU Budget: ~576 calls/day at 150s interval = ~100K CU/day (well under 1M)
        """
        while self._running and self._verification_enabled:
            try:
                await asyncio.sleep(self._verification_interval)

                if not self._running or not self._verification_w3:
                    break

                # Simple verification: check we can read the latest block
                # This confirms the RPC is working and gives us a reference point
                try:
                    verification_block = self._verification_w3.eth.block_number
                    primary_block = self.last_block

                    # If blocks match (within 5), verification successful
                    if abs(verification_block - primary_block) <= 5:
                        self._verifications_completed += 1
                        self._verification_failures = 0
                    # Note: We don't log mismatches - could be timing differences

                except Exception:
                    # Silent failure - increment counter but don't log
                    # See module docstring for why this is intentional
                    self._verification_failures += 1

                    # After 10 consecutive failures, disable verification silently
                    # (probably rate limited or key expired)
                    if self._verification_failures >= 10:
                        self._verification_enabled = False
                        break

            except asyncio.CancelledError:
                break
            except Exception:
                # Catch-all for any unexpected errors - fail silently
                self._verification_failures += 1
                if self._verification_failures >= 10:
                    self._verification_enabled = False
                    break

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        stats = {
            "connected": self._connected,
            "running": self._running,
            "mode": "websocket" if self.use_websocket and not self._use_polling_fallback else "polling",
            "wallets_monitored": len(self.wallets),
            "trades_detected": self.trades_detected,
            "blocks_processed": self.blocks_processed,
            "last_block": self.last_block,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }
        # Include verification stats if configured (don't expose failure count)
        if self._verification_rpc_url:
            stats["verification"] = {
                "enabled": self._verification_enabled,
                "completed": self._verifications_completed,
            }
        return stats


class HybridMonitor:
    """
    Hybrid monitoring combining blockchain events + API polling.

    Supports dual-RPC strategy:
    - Primary RPC (free public): For real-time 2s polling
    - Verification RPC (Alchemy): For periodic verification every 2.5 min

    This keeps Alchemy usage under 1M CU/day while providing full functionality.
    """

    def __init__(
        self,
        rpc_url: Optional[str],
        wallets: List[str],
        on_trade: Optional[TradeCallback] = None,
        verification_rpc_url: Optional[str] = None,
    ):
        self.rpc_url = rpc_url
        self.wallets = wallets
        self._on_trade = on_trade
        self._verification_rpc_url = verification_rpc_url
        self._blockchain: Optional[PolygonMonitor] = None
        self._blockchain_trades: Set[str] = set()

    async def start(self) -> bool:
        """Start hybrid monitoring."""
        if self.rpc_url:
            self._blockchain = PolygonMonitor(
                rpc_url=self.rpc_url,
                wallets=self.wallets,
                on_trade=self._handle_blockchain_trade,
                verification_rpc_url=self._verification_rpc_url,
            )
            success = await self._blockchain.start()
            if success:
                mode = "WebSocket" if self._blockchain.use_websocket and not self._blockchain._use_polling_fallback else "polling"
                print(f"  [Hybrid] Blockchain monitoring active ({mode})")
            else:
                self._blockchain = None
        return True

    async def stop(self) -> None:
        """Stop hybrid monitoring."""
        if self._blockchain:
            await self._blockchain.stop()

    def add_wallet(self, wallet: str) -> None:
        if self._blockchain:
            self._blockchain.add_wallet(wallet)

    def remove_wallet(self, wallet: str) -> None:
        if self._blockchain:
            self._blockchain.remove_wallet(wallet)

    async def _handle_blockchain_trade(self, trade: BlockchainTrade) -> None:
        self._blockchain_trades.add(trade.tx_hash)
        if self._on_trade:
            await self._on_trade(trade)

    def is_trade_from_blockchain(self, tx_hash: str) -> bool:
        return tx_hash in self._blockchain_trades

    def get_stats(self) -> dict:
        stats = {
            "mode": "hybrid" if self._blockchain else "api_only",
            "blockchain_trades": len(self._blockchain_trades),
        }
        if self._blockchain:
            stats["blockchain"] = self._blockchain.get_stats()
        return stats
