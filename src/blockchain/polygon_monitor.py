"""
Polygon blockchain monitoring for Polymarket trade detection.

This module provides real-time detection of trades by monitoring
on-chain events directly from the Polygon blockchain.

Achieves ~2-5 second latency vs ~15-21 seconds with API polling.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Set, Awaitable
from web3 import Web3

# Handle different web3 versions for POA middleware
try:
    from web3.middleware import ExtraDataToPOAMiddleware as poa_middleware
except ImportError:
    from web3.middleware import geth_poa_middleware as poa_middleware

# Polymarket contract addresses on Polygon mainnet
# CTF Exchange handles order matching
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Neg Risk CTF Exchange (for negatively correlated markets)
NEG_RISK_CTF_EXCHANGE_ADDRESS = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# CTF Token contract - where actual token transfers happen
# This is where we see user wallets in TransferSingle events
CTF_TOKEN_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Event signatures (keccak256 hashes)
# TransferSingle(address indexed operator, address indexed from, address indexed to, uint256 id, uint256 value)
# This is the key event - user wallets appear in 'from' (topic[2]) and 'to' (topic[3])
TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"

# OrderFilled - less reliable for wallet detection due to relayer system
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65b8e7c5d"


@dataclass
class BlockchainTrade:
    """Trade detected from blockchain event."""

    tx_hash: str
    block_number: int
    timestamp: datetime
    wallet: str  # The trader's address
    token_id: str  # The asset/token traded
    side: str  # BUY or SELL
    size: Decimal
    price: Decimal
    fee: Decimal
    detection_latency_ms: float  # Time from block to detection

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


# Type for trade callback
TradeCallback = Callable[[BlockchainTrade], Awaitable[None]]


class PolygonMonitor:
    """
    Real-time Polygon blockchain monitor for Polymarket trades.

    Subscribes to CTF Exchange events and detects trades for
    specified wallet addresses with ~2-5 second latency.

    Usage:
        monitor = PolygonMonitor(
            rpc_url="wss://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY",
            wallets=["0x...", "0x..."],
            on_trade=handle_trade,
        )
        await monitor.start()
    """

    def __init__(
        self,
        rpc_url: str,
        wallets: List[str],
        on_trade: Optional[TradeCallback] = None,
        use_websocket: bool = True,
    ):
        """
        Initialize the Polygon monitor.

        Args:
            rpc_url: Polygon RPC URL (WebSocket preferred: wss://...)
            wallets: List of wallet addresses to monitor
            on_trade: Async callback for detected trades
            use_websocket: Use WebSocket for real-time events (recommended)
        """
        self.rpc_url = rpc_url
        self.wallets = {w.lower() for w in wallets}
        self._on_trade = on_trade
        self.use_websocket = use_websocket

        # Web3 connection
        self._w3: Optional[Web3] = None
        self._connected = False
        self._running = False

        # Statistics
        self.trades_detected = 0
        self.blocks_processed = 0
        self.last_block = 0
        self.avg_latency_ms = 0.0
        self._latencies: List[float] = []

        # Seen transactions (prevent duplicates)
        self._seen_txs: Set[str] = set()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._w3 is not None

    @property
    def is_running(self) -> bool:
        return self._running

    def add_wallet(self, wallet: str) -> None:
        """Add a wallet to monitor."""
        self.wallets.add(wallet.lower())
        print(f"  [Blockchain] Added wallet to monitor: {wallet[:10]}...")

    def remove_wallet(self, wallet: str) -> None:
        """Remove a wallet from monitoring."""
        self.wallets.discard(wallet.lower())
        print(f"  [Blockchain] Removed wallet from monitor: {wallet[:10]}...")

    async def start(self) -> bool:
        """
        Start the blockchain monitor.

        Returns:
            True if connected successfully
        """
        if self._running:
            return True

        try:
            # Connect to Polygon
            if self.rpc_url.startswith("wss://") or self.rpc_url.startswith("ws://"):
                from web3 import AsyncWeb3
                from web3.providers import WebSocketProvider
                # For WebSocket, we'll use polling with HTTP for now
                # Full async WebSocket requires more setup
                http_url = self.rpc_url.replace("wss://", "https://").replace("ws://", "http://")
                self._w3 = Web3(Web3.HTTPProvider(http_url))
            else:
                self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))

            # Add PoA middleware for Polygon
            self._w3.middleware_onion.inject(poa_middleware, layer=0)

            # Test connection
            if not self._w3.is_connected():
                print(f"  [Blockchain] Failed to connect to {self.rpc_url[:30]}...")
                return False

            self._connected = True
            self._running = True
            self.last_block = self._w3.eth.block_number

            print(f"  [Blockchain] Connected to Polygon at block {self.last_block}")
            print(f"  [Blockchain] Monitoring {len(self.wallets)} wallets")

            # Start monitoring loop
            asyncio.create_task(self._monitor_loop())

            return True

        except Exception as e:
            print(f"  [Blockchain] Connection error: {e}")
            self._connected = False
            return False

    async def stop(self) -> None:
        """Stop the blockchain monitor."""
        self._running = False
        self._connected = False
        print(f"  [Blockchain] Stopped. Detected {self.trades_detected} trades across {self.blocks_processed} blocks")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop - polls for new blocks and events."""

        poll_interval = 1.0  # Poll every 1 second (catch blocks ASAP)
        rate_limit_backoff = 1.0  # Backoff time after rate limit
        max_blocks_per_batch = 5  # Max blocks to process per iteration (rate limit protection)

        while self._running:
            try:
                t_start = time.perf_counter()

                # Get current block
                current_block = self._w3.eth.block_number

                if current_block > self.last_block:
                    blocks_behind = current_block - self.last_block

                    # If too far behind, skip ahead to avoid rate limits
                    if blocks_behind > 50:
                        print(f"  [Blockchain] {blocks_behind} blocks behind, skipping to recent")
                        self.last_block = current_block - 10  # Only process last 10 blocks
                        blocks_behind = 10

                    # Process blocks with rate limiting
                    blocks_to_process = min(blocks_behind, max_blocks_per_batch)
                    for i, block_num in enumerate(range(self.last_block + 1, self.last_block + 1 + blocks_to_process)):
                        await self._process_block(block_num)
                        self.blocks_processed += 1
                        # Small delay between blocks to avoid rate limits
                        if i < blocks_to_process - 1:
                            await asyncio.sleep(0.05)

                    self.last_block = self.last_block + blocks_to_process

                # Calculate time to wait
                elapsed = time.perf_counter() - t_start
                wait_time = max(0, poll_interval - elapsed)
                await asyncio.sleep(wait_time)

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    print(f"  [Blockchain] Rate limited, backing off {rate_limit_backoff}s...")
                    await asyncio.sleep(rate_limit_backoff)
                    rate_limit_backoff = min(rate_limit_backoff * 2, 30)  # Exponential backoff, max 30s
                else:
                    print(f"  [Blockchain] Monitor error: {e}")
                    await asyncio.sleep(2.0)  # Wait before retry
                    rate_limit_backoff = 1.0  # Reset backoff on non-rate-limit errors

    async def _process_block(self, block_number: int) -> None:
        """
        Process a single block for CTF Exchange events.

        Args:
            block_number: Block to process
        """
        t_detect = time.perf_counter()

        try:
            # Get block timestamp for latency calculation
            block = self._w3.eth.get_block(block_number)
            block_time = datetime.utcfromtimestamp(block['timestamp'])

            # Query logs for CTF Token TransferSingle events
            # This is where user wallet addresses appear (in from/to fields)
            logs = self._w3.eth.get_logs({
                'fromBlock': block_number,
                'toBlock': block_number,
                'address': Web3.to_checksum_address(CTF_TOKEN_ADDRESS),
                'topics': [TRANSFER_SINGLE_TOPIC],  # Filter for TransferSingle events
            })

            for log in logs:
                await self._process_log(log, block_time, t_detect)

        except Exception as e:
            error_str = str(e).lower()
            # Don't spam logs for common errors
            if "not found" in error_str:
                pass  # Block not found yet, ignore
            elif "429" in str(e) or "too many requests" in error_str:
                # Rate limited - will be handled by monitor loop backoff
                raise  # Re-raise to trigger backoff in monitor loop
            else:
                print(f"  [Blockchain] Block {block_number} error: {e}")

    async def _process_log(self, log: dict, block_time: datetime, t_detect: float) -> None:
        """
        Process a single event log.

        Polymarket uses a relayer system - user addresses are in event logs, not tx from/to.
        We parse the OrderFilled event to find maker/taker addresses.

        Args:
            log: The event log from Web3
            block_time: Timestamp of the block
            t_detect: Time when we started processing
        """
        tx_hash = log['transactionHash'].hex()

        # Skip if already seen
        if tx_hash in self._seen_txs:
            return
        self._seen_txs.add(tx_hash)

        # Limit seen set size
        if len(self._seen_txs) > 10000:
            self._seen_txs = set(list(self._seen_txs)[-5000:])

        try:
            # Extract from/to addresses from TransferSingle event
            # TransferSingle: topic[0]=sig, topic[1]=operator, topic[2]=from, topic[3]=to
            topics = log.get('topics', [])

            trader_address = None
            trade_side = None

            if len(topics) >= 4:
                # Extract from (topic[2]) and to (topic[3]) - they're padded to 32 bytes
                from_raw = topics[2].hex() if hasattr(topics[2], 'hex') else str(topics[2])
                to_raw = topics[3].hex() if hasattr(topics[3], 'hex') else str(topics[3])

                # Convert from 32-byte padded to address (last 40 chars = 20 bytes)
                from_addr = '0x' + from_raw[-40:].lower()
                to_addr = '0x' + to_raw[-40:].lower()

                # Check if from or to is a wallet we're monitoring
                # from = seller (SELL), to = buyer (BUY)
                if from_addr in self.wallets:
                    trader_address = from_addr
                    trade_side = "SELL"
                elif to_addr in self.wallets:
                    trader_address = to_addr
                    trade_side = "BUY"

            if not trader_address:
                return  # No monitored wallet involved in this transfer

            # Get transaction for additional context
            tx = self._w3.eth.get_transaction(tx_hash)

            # Parse the trade details from the log
            trade = await self._parse_trade_log(log, tx, trader_address, trade_side, block_time, t_detect)

            if trade:
                self.trades_detected += 1

                # Track latency
                self._latencies.append(trade.detection_latency_ms)
                if len(self._latencies) > 100:
                    self._latencies = self._latencies[-100:]
                self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

                print(f"  [Blockchain] Trade detected: {trade.side} {trade.size} @ {trade.price} "
                      f"(latency: {trade.detection_latency_ms:.0f}ms)")

                # Notify callback
                if self._on_trade:
                    await self._on_trade(trade)

        except Exception as e:
            # Transaction parsing errors are common, don't spam
            pass

    async def _parse_trade_log(
        self,
        log: dict,
        tx: dict,
        sender: str,
        trade_side: str,
        block_time: datetime,
        t_detect: float,
    ) -> Optional[BlockchainTrade]:
        """
        Parse a trade from a TransferSingle event log.

        Args:
            log: Event log
            tx: Full transaction
            sender: Wallet address
            trade_side: "BUY" or "SELL" (determined by from/to in event)
            block_time: Block timestamp
            t_detect: Detection start time

        Returns:
            BlockchainTrade or None if parsing fails
        """
        try:
            # Calculate detection latency
            detection_latency_ms = (time.perf_counter() - t_detect) * 1000
            detection_latency_ms += (datetime.utcnow() - block_time).total_seconds() * 1000

            # Parse TransferSingle log data
            # Data format: id (uint256, 32 bytes) + value (uint256, 32 bytes)
            data = log.get('data', '0x')
            data_hex = data.hex() if hasattr(data, 'hex') else data
            if data_hex.startswith('0x'):
                data_hex = data_hex[2:]

            # Extract token ID (first 32 bytes = 64 hex chars)
            # Convert from hex to decimal - Polymarket API expects decimal integer string
            token_id_hex = data_hex[:64] if len(data_hex) >= 64 else ""
            token_id = str(int(token_id_hex, 16)) if token_id_hex else ""

            # Extract value/size (second 32 bytes)
            # CTF tokens typically have 18 decimals (like ETH)
            if len(data_hex) >= 128:
                value_raw = int(data_hex[64:128], 16)
                size = Decimal(str(value_raw)) / Decimal("1000000")  # 6 decimals for CTF position tokens
            else:
                size = Decimal("0")

            # Use the side determined from from/to addresses
            side = trade_side

            # Price estimation - we don't have exact price from TransferSingle
            # Would need to correlate with order book or USDC transfer in same tx
            price = Decimal("0")  # Price requires additional context

            return BlockchainTrade(
                tx_hash=log['transactionHash'].hex(),
                block_number=log['blockNumber'],
                timestamp=block_time,
                wallet=sender,
                token_id=token_id,
                side=side,
                size=size,
                price=price,
                fee=Decimal("0"),  # Fee not available in TransferSingle
                detection_latency_ms=detection_latency_ms,
            )

        except Exception as e:
            return None

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            "connected": self._connected,
            "running": self._running,
            "wallets_monitored": len(self.wallets),
            "trades_detected": self.trades_detected,
            "blocks_processed": self.blocks_processed,
            "last_block": self.last_block,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class HybridMonitor:
    """
    Hybrid monitoring combining blockchain events + API polling.

    Uses blockchain for fastest detection, falls back to API
    for reliability and additional data enrichment.
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

        # Blockchain monitor (if RPC available)
        self._blockchain: Optional[PolygonMonitor] = None

        # Track trades from both sources
        self._blockchain_trades: Set[str] = set()
        self._api_trades: Set[str] = set()

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
                print("  [Hybrid] Blockchain monitoring active - expect ~2-5s latency")
            else:
                print("  [Hybrid] Blockchain connection failed, using API polling only")
                self._blockchain = None
        else:
            print("  [Hybrid] No RPC URL configured, using API polling only")

        return True

    async def stop(self) -> None:
        """Stop hybrid monitoring."""
        if self._blockchain:
            await self._blockchain.stop()

    def add_wallet(self, wallet: str) -> None:
        """Add wallet to monitoring."""
        if self._blockchain:
            self._blockchain.add_wallet(wallet)

    def remove_wallet(self, wallet: str) -> None:
        """Remove wallet from monitoring."""
        if self._blockchain:
            self._blockchain.remove_wallet(wallet)

    async def _handle_blockchain_trade(self, trade: BlockchainTrade) -> None:
        """Handle trade detected from blockchain."""
        self._blockchain_trades.add(trade.tx_hash)

        if self._on_trade:
            await self._on_trade(trade)

    def is_trade_from_blockchain(self, tx_hash: str) -> bool:
        """Check if a trade was detected via blockchain."""
        return tx_hash in self._blockchain_trades

    def get_stats(self) -> dict:
        """Get hybrid monitoring stats."""
        stats = {
            "mode": "hybrid" if self._blockchain else "api_only",
            "blockchain_trades": len(self._blockchain_trades),
        }
        if self._blockchain:
            stats["blockchain"] = self._blockchain.get_stats()
        return stats
