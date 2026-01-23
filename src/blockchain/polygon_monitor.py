"""
Polygon blockchain monitoring for Polymarket trade detection.

This module provides real-time detection of trades by monitoring
on-chain events directly from the Polygon blockchain.

Uses WebSocket subscriptions for ~350-500ms latency (vs ~1-2s with polling).
"""

import asyncio
import json
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
    """

    def __init__(
        self,
        rpc_url: str,
        wallets: List[str],
        on_trade: Optional[TradeCallback] = None,
        use_websocket: bool = True,
    ):
        self.rpc_url = rpc_url
        self.wallets = {w.lower() for w in wallets}
        self._on_trade = on_trade
        self.use_websocket = use_websocket and WEBSOCKETS_AVAILABLE

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
        self._reconnect_delay = 1.0

        # Statistics
        self.trades_detected = 0
        self.blocks_processed = 0
        self.last_block = 0
        self.avg_latency_ms = 0.0
        self._latencies: List[float] = []

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
            # Connect Web3 for HTTP queries
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
        self._connected = False
        print(f"  [Blockchain] Stopped. Detected {self.trades_detected} trades")

    async def _websocket_loop(self) -> None:
        """WebSocket subscription loop - receives events in real-time."""

        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    print(f"  [Blockchain] WebSocket connected!")
                    self._reconnect_delay = 1.0  # Reset backoff on successful connect

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
                        await asyncio.sleep(5)
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
                print(f"  [Blockchain] WebSocket closed: {e}, reconnecting...")
            except Exception as e:
                print(f"  [Blockchain] WebSocket error: {e}")

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 1.5, 30)

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

        poll_interval = 0.5  # Poll every 500ms for faster detection

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

                elapsed = time.perf_counter() - t_start
                wait_time = max(0, poll_interval - elapsed)
                await asyncio.sleep(wait_time)

            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(2)
                else:
                    await asyncio.sleep(1)

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

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            "connected": self._connected,
            "running": self._running,
            "mode": "websocket" if self.use_websocket else "polling",
            "wallets_monitored": len(self.wallets),
            "trades_detected": self.trades_detected,
            "blocks_processed": self.blocks_processed,
            "last_block": self.last_block,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class HybridMonitor:
    """
    Hybrid monitoring combining blockchain events + API polling.
    """

    def __init__(
        self,
        rpc_url: Optional[str],
        wallets: List[str],
        on_trade: Optional[TradeCallback] = None,
    ):
        self.rpc_url = rpc_url
        self.wallets = wallets
        self._on_trade = on_trade
        self._blockchain: Optional[PolygonMonitor] = None
        self._blockchain_trades: Set[str] = set()

    async def start(self) -> bool:
        """Start hybrid monitoring."""
        if self.rpc_url:
            self._blockchain = PolygonMonitor(
                rpc_url=self.rpc_url,
                wallets=self.wallets,
                on_trade=self._handle_blockchain_trade,
            )
            success = await self._blockchain.start()
            if success:
                mode = "WebSocket" if self._blockchain.use_websocket else "polling"
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
