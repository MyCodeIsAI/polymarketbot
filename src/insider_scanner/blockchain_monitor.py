"""Blockchain Monitor for Automatic Sybil Detection.

Watches Polygon blockchain for new wallet funding transactions and
automatically checks them against known profitable trader funding sources.

This enables fully automatic detection of profitable trader alts:
1. Poll Polygonscan for recent MATIC/USDC transfers
2. Detect wallets receiving their first funding
3. Check if funding source matches indexed profitable traders
4. Auto-flag as sybil if match found
"""

import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, Awaitable, Dict, Set, List, Any
from dataclasses import dataclass, field

import httpx

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FundingTransaction:
    """A funding transaction detected on-chain."""
    tx_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    value_matic: Decimal
    value_usd: Optional[Decimal] = None


@dataclass
class BlockchainMonitorStats:
    """Statistics for blockchain monitoring."""
    started_at: Optional[datetime] = None
    blocks_scanned: int = 0
    transactions_seen: int = 0
    new_wallets_detected: int = 0
    sybil_checks_performed: int = 0
    sybil_matches_found: int = 0
    last_block: int = 0
    last_scan_at: Optional[datetime] = None
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "blocks_scanned": self.blocks_scanned,
            "transactions_seen": self.transactions_seen,
            "new_wallets_detected": self.new_wallets_detected,
            "sybil_checks_performed": self.sybil_checks_performed,
            "sybil_matches_found": self.sybil_matches_found,
            "last_block": self.last_block,
            "last_scan_at": self.last_scan_at.isoformat() if self.last_scan_at else None,
            "errors": self.errors,
        }


# Callback type for sybil check
SybilCheckCallback = Callable[[str, str, Decimal, str], Awaitable[Optional[Any]]]


class BlockchainMonitor:
    """Monitor Polygon blockchain for new wallet funding transactions.

    Polls Polygonscan API to detect:
    - New wallets receiving their first MATIC funding
    - Large deposits that could indicate whale/insider activity

    When a new funded wallet is detected, triggers sybil detection check.

    Example:
        async def on_new_funding(new_wallet, funding_source, amount, tx_hash):
            # Check against sybil detector
            match = sybil_detector.check_funding_source(funding_source)
            if match.matched:
                sybil_detector.flag_sybil_wallet(new_wallet, funding_source)

        monitor = BlockchainMonitor(
            api_key="YOUR_KEY",
            on_new_funding=on_new_funding,
        )
        await monitor.start()
    """

    # Etherscan V2 API (supports Polygon)
    API_BASE = "https://api.etherscan.io/v2/api"
    POLYGON_CHAIN_ID = "137"

    # Polymarket contract addresses on Polygon
    POLYMARKET_CONTRACTS = {
        # CTF Exchange
        "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
        # Neg Risk CTF Exchange
        "0xc5d563a36ae78145c45a50134d48a1215220f80a",
        # USDC.e on Polygon
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        # Conditional Tokens
        "0x4d97dcd97ec945f40cf65f87097ace5ea0476045",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        on_new_funding: Optional[SybilCheckCallback] = None,
        poll_interval_s: float = 15.0,
        min_funding_matic: float = 0.5,
        lookback_blocks: int = 100,
    ):
        """Initialize blockchain monitor.

        Args:
            api_key: Polygonscan/Etherscan API key (uses env var if not provided)
            on_new_funding: Callback when new wallet funding detected
            poll_interval_s: Seconds between polling cycles
            min_funding_matic: Minimum MATIC to consider as funding
            lookback_blocks: How many blocks to look back on first scan
        """
        self.api_key = api_key or os.getenv("POLYGONSCAN_API_KEY") or os.getenv("ETHERSCAN_API_KEY")
        self.on_new_funding = on_new_funding
        self.poll_interval_s = poll_interval_s
        self.min_funding_matic = min_funding_matic
        self.lookback_blocks = lookback_blocks

        # State
        self._should_run = False
        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        # Tracking
        self._known_wallets: Set[str] = set()
        self._processed_tx_hashes: Set[str] = set()
        self._last_block: int = 0

        # Statistics
        self.stats = BlockchainMonitorStats()

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._should_run and self._poll_task is not None

    async def start(self) -> bool:
        """Start the blockchain monitor.

        Returns:
            True if started successfully
        """
        if not self.api_key:
            logger.error("blockchain_monitor_no_api_key",
                        message="Set POLYGONSCAN_API_KEY or ETHERSCAN_API_KEY env var")
            return False

        if self.is_running:
            return True

        logger.info("blockchain_monitor_starting",
                   poll_interval=self.poll_interval_s,
                   min_funding=self.min_funding_matic)

        self._should_run = True
        self.stats = BlockchainMonitorStats(started_at=datetime.utcnow())

        # Create HTTP client
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Get current block number
        self._last_block = await self._get_latest_block() - self.lookback_blocks

        # Start polling task
        self._poll_task = asyncio.create_task(self._poll_loop())

        logger.info("blockchain_monitor_started", starting_block=self._last_block)
        return True

    async def stop(self) -> None:
        """Stop the blockchain monitor."""
        if not self.is_running:
            return

        logger.info("blockchain_monitor_stopping")
        self._should_run = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("blockchain_monitor_stopped", stats=self.stats.to_dict())

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._should_run:
            try:
                await self._scan_new_blocks()
                self.stats.last_scan_at = datetime.utcnow()
                await asyncio.sleep(self.poll_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("blockchain_monitor_poll_error", error=str(e))
                self.stats.errors += 1
                await asyncio.sleep(self.poll_interval_s * 2)  # Back off on error

    async def _get_latest_block(self) -> int:
        """Get the latest block number on Polygon."""
        params = {
            "chainid": self.POLYGON_CHAIN_ID,
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": self.api_key,
        }

        try:
            response = await self._http_client.get(self.API_BASE, params=params)
            data = response.json()

            if "result" in data:
                return int(data["result"], 16)
        except Exception as e:
            logger.error("get_latest_block_error", error=str(e))

        return 0

    async def _scan_new_blocks(self) -> None:
        """Scan for new funding transactions in recent blocks."""
        current_block = await self._get_latest_block()

        if current_block <= self._last_block:
            return

        # Get internal transactions (MATIC transfers) in block range
        # Using txlistinternal for value transfers
        params = {
            "chainid": self.POLYGON_CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "address": "",  # We'll scan by block range
            "startblock": str(self._last_block + 1),
            "endblock": str(min(current_block, self._last_block + 500)),  # Max 500 blocks
            "page": "1",
            "offset": "1000",
            "sort": "asc",
            "apikey": self.api_key,
        }

        # Instead of scanning all txs, let's check Polymarket-related addresses
        # We'll look for deposits to known Polymarket contracts and new wallets
        await self._scan_polymarket_deposits(self._last_block + 1, current_block)

        self.stats.blocks_scanned += (current_block - self._last_block)
        self.stats.last_block = current_block
        self._last_block = current_block

    async def _scan_polymarket_deposits(self, start_block: int, end_block: int) -> None:
        """Scan for deposits related to Polymarket activity.

        Strategy: Look for recent MATIC transfers where recipient then
        interacts with Polymarket contracts.
        """
        # For efficiency, we'll check recent large MATIC transfers
        # and see if recipients are new to us

        # Get recent large transfers using token transfer endpoint (for MATIC, use normal tx)
        params = {
            "chainid": self.POLYGON_CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "startblock": str(start_block),
            "endblock": str(min(end_block, start_block + 200)),
            "page": "1",
            "offset": "500",
            "sort": "desc",
            "apikey": self.api_key,
        }

        # We need an address to query - let's check deposits TO Polymarket
        # For new wallet detection, we check USDC.e transfers to the exchange
        for contract in list(self.POLYMARKET_CONTRACTS)[:2]:  # Check main contracts
            params["address"] = contract

            try:
                response = await self._http_client.get(self.API_BASE, params=params)
                data = response.json()

                if data.get("status") != "1" or not isinstance(data.get("result"), list):
                    continue

                for tx in data["result"]:
                    await self._process_transaction(tx)

            except Exception as e:
                logger.debug("scan_contract_error", contract=contract[:10], error=str(e))

    async def _process_transaction(self, tx: dict) -> None:
        """Process a transaction to detect new wallet funding."""
        tx_hash = tx.get("hash", "")

        # Skip already processed
        if tx_hash in self._processed_tx_hashes:
            return
        self._processed_tx_hashes.add(tx_hash)

        # Limit memory - keep last 10k transactions
        if len(self._processed_tx_hashes) > 10000:
            # Remove oldest (convert to list, slice, convert back)
            self._processed_tx_hashes = set(list(self._processed_tx_hashes)[-5000:])

        self.stats.transactions_seen += 1

        from_addr = tx.get("from", "").lower()
        to_addr = tx.get("to", "").lower()
        value_wei = int(tx.get("value", "0"))
        value_matic = Decimal(value_wei) / Decimal("1e18")

        # Skip small transactions
        if value_matic < self.min_funding_matic:
            return

        # Check if this is a NEW wallet we haven't seen
        if to_addr not in self._known_wallets:
            self._known_wallets.add(to_addr)

            # Limit memory
            if len(self._known_wallets) > 50000:
                # Keep most recent
                self._known_wallets = set(list(self._known_wallets)[-25000:])

            # This looks like first funding to a wallet!
            self.stats.new_wallets_detected += 1

            logger.info(
                "new_wallet_funding_detected",
                new_wallet=to_addr[:10] + "...",
                funding_source=from_addr[:10] + "...",
                amount_matic=float(value_matic),
                tx_hash=tx_hash[:10] + "...",
            )

            # Trigger sybil check callback
            if self.on_new_funding:
                try:
                    self.stats.sybil_checks_performed += 1
                    result = await self.on_new_funding(
                        to_addr,
                        from_addr,
                        value_matic,
                        tx_hash,
                    )

                    if result:  # Match found
                        self.stats.sybil_matches_found += 1

                except Exception as e:
                    logger.error("sybil_check_callback_error", error=str(e))

    async def check_wallet_funding_source(self, wallet_address: str) -> Optional[FundingTransaction]:
        """Manually check the funding source for a wallet.

        Args:
            wallet_address: Wallet to check

        Returns:
            FundingTransaction if found, None otherwise
        """
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        params = {
            "chainid": self.POLYGON_CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "address": wallet_address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": "50",
            "sort": "asc",
            "apikey": self.api_key,
        }

        try:
            response = await self._http_client.get(self.API_BASE, params=params)
            data = response.json()

            if data.get("status") != "1" or not isinstance(data.get("result"), list):
                return None

            # Find first inbound MATIC transfer
            for tx in data["result"]:
                to_addr = tx.get("to", "").lower()
                from_addr = tx.get("from", "").lower()
                value_wei = int(tx.get("value", "0"))
                value_matic = Decimal(value_wei) / Decimal("1e18")

                if to_addr == wallet_address.lower() and value_matic >= self.min_funding_matic:
                    return FundingTransaction(
                        tx_hash=tx.get("hash", ""),
                        block_number=int(tx.get("blockNumber", "0")),
                        timestamp=datetime.fromtimestamp(int(tx.get("timeStamp", "0"))),
                        from_address=from_addr,
                        to_address=to_addr,
                        value_matic=value_matic,
                    )

        except Exception as e:
            logger.error("check_funding_source_error", error=str(e))

        return None

    def add_known_wallet(self, wallet_address: str) -> None:
        """Add a wallet to the known set (won't trigger new wallet detection).

        Call this for existing wallets to prevent false positives.
        """
        self._known_wallets.add(wallet_address.lower())

    def add_known_wallets(self, wallets: List[str]) -> None:
        """Bulk add known wallets."""
        for wallet in wallets:
            self._known_wallets.add(wallet.lower())

    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return self.stats.to_dict()
