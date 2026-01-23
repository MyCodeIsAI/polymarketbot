"""
Blockchain monitoring for fast trade detection on Polygon.

This module watches for on-chain events from Polymarket's CTF Exchange
to detect trades with ~1-2 second latency (vs ~15-21s with API polling).

Requires: web3.py and a Polygon RPC URL (HTTP or WebSocket)
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None


# Polymarket CTF Exchange contract on Polygon
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Conditional Tokens Framework contract (where transfers happen)
CTF_CONTRACT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ERC1155 Transfer events (used by CTF for outcome tokens)
TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
TRANSFER_BATCH_TOPIC = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"


@dataclass
class BlockchainTrade:
    """A trade detected from blockchain events."""

    tx_hash: str
    wallet: str
    token_id: str
    side: str  # BUY or SELL
    timestamp: datetime
    block_number: int
    detection_latency_ms: float  # Time from block to detection

    def __repr__(self):
        return f"BlockchainTrade({self.side} by {self.wallet[:10]}... latency={self.detection_latency_ms:.0f}ms)"


class PolygonMonitor:
    """
    Monitors Polygon blockchain for Polymarket trades.

    Uses polling of recent blocks to detect ERC1155 transfers
    involving tracked wallets. Much faster than API polling.
    """

    def __init__(
        self,
        rpc_url: str,
        wallets: List[str],
        on_trade: Optional[Callable[[BlockchainTrade], None]] = None,
        poll_interval: float = 1.0,
    ):
        """
        Initialize the Polygon monitor.

        Args:
            rpc_url: Polygon RPC URL (HTTP or WebSocket)
            wallets: List of wallet addresses to monitor
            on_trade: Callback function when trade detected
            poll_interval: How often to poll for new blocks (seconds)
        """
        self.rpc_url = rpc_url
        self.wallets = set(w.lower() for w in wallets)
        self.on_trade = on_trade
        self.poll_interval = poll_interval

        self._running = False
        self._w3: Optional[Web3] = None
        self._last_block = 0
        self._seen_txs: Set[str] = set()
        self._latencies: List[float] = []
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._task: Optional[asyncio.Task] = None

    @property
    def avg_latency_ms(self) -> float:
        """Average detection latency in milliseconds."""
        if not self._latencies:
            return 0
        return sum(self._latencies[-50:]) / len(self._latencies[-50:])

    async def start(self) -> bool:
        """
        Start monitoring the blockchain.

        Returns:
            True if started successfully, False otherwise
        """
        if not WEB3_AVAILABLE:
            print("  [Blockchain] web3.py not installed")
            return False

        try:
            # Connect to Polygon
            if self.rpc_url.startswith("ws"):
                self._w3 = Web3(Web3.WebsocketProvider(self.rpc_url))
            else:
                self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))

            # Add PoA middleware for Polygon
            self._w3.middleware_onion.inject(geth_poa_middleware, layer=0)

            if not self._w3.is_connected():
                print("  [Blockchain] Failed to connect to RPC")
                return False

            # Get current block
            self._last_block = self._w3.eth.block_number
            print(f"  [Blockchain] Connected! Current block: {self._last_block}")
            print(f"  [Blockchain] Monitoring {len(self.wallets)} wallets")

            self._running = True
            self._task = asyncio.create_task(self._poll_loop())
            return True

        except Exception as e:
            print(f"  [Blockchain] Error starting: {e}")
            return False

    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)
        print("  [Blockchain] Stopped")

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                await self._check_new_blocks()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  [Blockchain] Poll error: {e}")
                await asyncio.sleep(2)

    async def _check_new_blocks(self):
        """Check for new blocks and process them."""
        if not self._w3:
            return

        try:
            # Run web3 call in thread pool to not block async loop
            loop = asyncio.get_event_loop()
            current_block = await loop.run_in_executor(
                self._executor,
                lambda: self._w3.eth.block_number
            )

            if current_block <= self._last_block:
                return

            # Process new blocks (limit to last 5 to avoid overload)
            start_block = max(self._last_block + 1, current_block - 5)

            for block_num in range(start_block, current_block + 1):
                await self._process_block(block_num)

            self._last_block = current_block

        except Exception as e:
            print(f"  [Blockchain] Block check error: {e}")

    async def _process_block(self, block_num: int):
        """Process a single block for trades."""
        if not self._w3:
            return

        try:
            loop = asyncio.get_event_loop()

            # Get block with transactions
            block = await loop.run_in_executor(
                self._executor,
                lambda: self._w3.eth.get_block(block_num, full_transactions=True)
            )

            block_time = datetime.utcfromtimestamp(block['timestamp'])

            # Check each transaction
            for tx in block['transactions']:
                await self._check_transaction(tx, block_time, block_num)

        except Exception as e:
            # Block not available yet, skip
            pass

    async def _check_transaction(self, tx, block_time: datetime, block_num: int):
        """Check if a transaction involves our tracked wallets."""
        if not self._w3:
            return

        tx_hash = tx['hash'].hex() if hasattr(tx['hash'], 'hex') else str(tx['hash'])

        # Skip if already seen
        if tx_hash in self._seen_txs:
            return
        self._seen_txs.add(tx_hash)

        # Limit seen txs size
        if len(self._seen_txs) > 10000:
            self._seen_txs = set(list(self._seen_txs)[-5000:])

        # Check if transaction is to CTF Exchange or CTF contract
        to_addr = tx.get('to', '').lower() if tx.get('to') else ''
        if to_addr not in [CTF_EXCHANGE_ADDRESS.lower(), CTF_CONTRACT_ADDRESS.lower()]:
            return

        from_addr = tx.get('from', '').lower() if tx.get('from') else ''

        # Check if sender is one of our tracked wallets
        if from_addr not in self.wallets:
            return

        # This is a potential trade from a tracked wallet!
        detection_time = datetime.utcnow()
        latency_ms = (detection_time - block_time).total_seconds() * 1000
        self._latencies.append(latency_ms)

        # Try to get receipt for more details
        try:
            loop = asyncio.get_event_loop()
            receipt = await loop.run_in_executor(
                self._executor,
                lambda: self._w3.eth.get_transaction_receipt(tx_hash)
            )

            # Look for Transfer events to determine token and direction
            for log in receipt.get('logs', []):
                topics = [t.hex() if hasattr(t, 'hex') else str(t) for t in log.get('topics', [])]

                if not topics:
                    continue

                # Check for ERC1155 TransferSingle
                if topics[0] == TRANSFER_SINGLE_TOPIC and len(topics) >= 4:
                    # TransferSingle(operator, from, to, id, value)
                    # topics: [event_sig, operator, from, to]
                    from_transfer = '0x' + topics[2][-40:]
                    to_transfer = '0x' + topics[3][-40:]

                    # Decode token_id from data
                    data = log.get('data', '0x')
                    if len(data) >= 66:
                        token_id = data[2:66]  # First 32 bytes is token_id
                    else:
                        token_id = "unknown"

                    # Determine if BUY or SELL based on direction
                    if to_transfer.lower() == from_addr:
                        side = "BUY"
                    elif from_transfer.lower() == from_addr:
                        side = "SELL"
                    else:
                        continue

                    trade = BlockchainTrade(
                        tx_hash=tx_hash,
                        wallet=from_addr,
                        token_id=token_id,
                        side=side,
                        timestamp=block_time,
                        block_number=block_num,
                        detection_latency_ms=latency_ms,
                    )

                    print(f"  [BLOCKCHAIN] {side} detected! wallet={from_addr[:10]}... latency={latency_ms:.0f}ms")

                    # Call the callback
                    if self.on_trade:
                        # Run callback in async context
                        if asyncio.iscoroutinefunction(self.on_trade):
                            await self.on_trade(trade)
                        else:
                            self.on_trade(trade)

                    return  # Only report first matching transfer

        except Exception as e:
            # Couldn't get receipt, still report basic trade
            print(f"  [Blockchain] Receipt error: {e}")

            trade = BlockchainTrade(
                tx_hash=tx_hash,
                wallet=from_addr,
                token_id="unknown",
                side="UNKNOWN",
                timestamp=block_time,
                block_number=block_num,
                detection_latency_ms=latency_ms,
            )

            if self.on_trade:
                if asyncio.iscoroutinefunction(self.on_trade):
                    await self.on_trade(trade)
                else:
                    self.on_trade(trade)

    def add_wallet(self, wallet: str):
        """Add a wallet to monitor."""
        self.wallets.add(wallet.lower())

    def remove_wallet(self, wallet: str):
        """Remove a wallet from monitoring."""
        self.wallets.discard(wallet.lower())
