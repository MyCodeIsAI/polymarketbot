"""Comprehensive Scanner Service for Insider Detection.

Integrates all components:
- Real-time monitoring (WebSocket + polling)
- Insider pattern scoring
- Sybil detection (profitable trader alts)
- Funding source extraction for flagged wallets
- Alert generation
- Auto-population of data

This is the main entry point for the scanner system.
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Callable, Awaitable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session

from .scoring import InsiderScorer, ScoringResult, MarketCategory
from .sybil_detector import SybilDetector, SybilMatch
from .monitor import RealTimeMonitor, NewWalletMonitor, TradeEvent, WalletAccumulation, MonitorState
from .blockchain_monitor import BlockchainMonitor
from .profile import ProfileFetcher
from .audit import AuditTrailManager
from .alerts import InsiderAlertManager, InsiderAlert, InsiderAlertType
from .models import (
    FlaggedWallet,
    FlaggedFundingSource,
    InsiderPriority,
    WalletStatus,
    FlagType,
    KnownProfitableTrader,
    KnownFundingSource,
    DetectionRecord,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Data directory for auto-population
DATA_DIR = Path(__file__).parent.parent.parent / "data"


class ScannerMode(str, Enum):
    """Scanner operational modes."""
    STOPPED = "stopped"
    INSIDER_ONLY = "insider_only"  # Only insider pattern detection
    SYBIL_ONLY = "sybil_only"      # Only sybil detection
    FULL = "full"                   # Both scanners active


@dataclass
class ScannerStats:
    """Comprehensive scanner statistics."""
    mode: ScannerMode = ScannerMode.STOPPED
    started_at: Optional[datetime] = None

    # Insider scanner stats
    insider_trades_processed: int = 0
    insider_wallets_scored: int = 0
    insider_flags_generated: int = 0

    # Sybil scanner stats
    sybil_checks_performed: int = 0
    sybil_matches_found: int = 0
    sybil_flags_generated: int = 0

    # Funding extraction stats
    funding_extractions_queued: int = 0
    funding_extractions_completed: int = 0
    funding_sources_discovered: int = 0

    # Data population stats
    profitable_traders_loaded: int = 0
    funding_sources_loaded: int = 0
    high_volume_excluded: int = 0

    # Alert stats
    alerts_generated: int = 0
    critical_alerts: int = 0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "insider": {
                "trades_processed": self.insider_trades_processed,
                "wallets_scored": self.insider_wallets_scored,
                "flags_generated": self.insider_flags_generated,
            },
            "sybil": {
                "checks_performed": self.sybil_checks_performed,
                "matches_found": self.sybil_matches_found,
                "flags_generated": self.sybil_flags_generated,
            },
            "funding_extraction": {
                "queued": self.funding_extractions_queued,
                "completed": self.funding_extractions_completed,
                "sources_discovered": self.funding_sources_discovered,
            },
            "data": {
                "profitable_traders": self.profitable_traders_loaded,
                "funding_sources": self.funding_sources_loaded,
                "high_volume_excluded": self.high_volume_excluded,
            },
            "alerts": {
                "total": self.alerts_generated,
                "critical": self.critical_alerts,
            },
        }


class InsiderScannerService:
    """Main scanner service integrating all components.

    Provides:
    1. Insider Pattern Detection - Scores wallets against documented insider footprints
    2. Sybil Detection - Detects profitable trader alts via funding source matching
    3. Funding Source Extraction - Extracts and tracks funding sources for flagged wallets
    4. Alert Generation - Multi-channel alerts for suspicious activity
    5. Auto-population - Loads data from extracted files on startup

    Example:
        async with InsiderScannerService(session_factory) as service:
            # Auto-populates data from files
            await service.start(mode=ScannerMode.FULL)

            # Scanner runs in background...

            await service.stop()
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        alert_manager: Optional[InsiderAlertManager] = None,
        polygonscan_api_key: Optional[str] = None,
        insider_threshold: float = 55.0,
        sybil_high_volume_threshold: int = 10,
        auto_populate: bool = True,
    ):
        """Initialize the scanner service.

        Args:
            session_factory: Factory function returning database sessions
            alert_manager: Optional alert manager (created if not provided)
            polygonscan_api_key: API key for Polygonscan funding extraction
            insider_threshold: Minimum score to flag as insider (default 55)
            sybil_high_volume_threshold: Max traders a source can fund before exclusion
            auto_populate: Whether to auto-load data from files on start
        """
        self.session_factory = session_factory
        self.alert_manager = alert_manager
        self.polygonscan_api_key = polygonscan_api_key
        self.insider_threshold = insider_threshold
        self.sybil_high_volume_threshold = sybil_high_volume_threshold
        self.auto_populate = auto_populate

        # Components
        self.scorer = InsiderScorer()
        self.sybil_detector: Optional[SybilDetector] = None
        self.monitor: Optional[RealTimeMonitor] = None
        self.new_wallet_monitor: Optional[NewWalletMonitor] = None
        self.blockchain_monitor: Optional[BlockchainMonitor] = None

        # State
        self._mode = ScannerMode.STOPPED
        self._should_run = False
        self._funding_extraction_queue: asyncio.Queue = asyncio.Queue()
        self._funding_extraction_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = ScannerStats()

    async def __aenter__(self) -> "InsiderScannerService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    @property
    def mode(self) -> ScannerMode:
        """Get current scanner mode."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """Check if scanner is running."""
        return self._mode != ScannerMode.STOPPED

    async def start(self, mode: ScannerMode = ScannerMode.FULL) -> bool:
        """Start the scanner service.

        Args:
            mode: Which scanner(s) to run

        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("scanner_already_running", current_mode=self._mode.value)
            return False

        logger.info("scanner_starting", mode=mode.value)
        self._should_run = True
        self._mode = mode
        self.stats = ScannerStats(mode=mode, started_at=datetime.utcnow())

        # Initialize sybil detector
        self.sybil_detector = SybilDetector(self.session_factory)
        self.sybil_detector.max_funded_traders_threshold = self.sybil_high_volume_threshold

        # Auto-populate data if enabled
        if self.auto_populate:
            await self._auto_populate_data()

        # Build sybil index
        index_stats = self.sybil_detector.rebuild_index()
        self.stats.funding_sources_loaded = index_stats.get("funding_sources", 0)
        self.stats.high_volume_excluded = index_stats.get("high_volume_excluded", 0)
        logger.info("sybil_index_built", **index_stats)

        # Start funding extraction worker
        self._funding_extraction_task = asyncio.create_task(self._funding_extraction_worker())

        # Start real-time monitor for insider detection
        if mode in (ScannerMode.INSIDER_ONLY, ScannerMode.FULL):
            self.monitor = RealTimeMonitor(
                scorer=self.scorer,
                on_alert=self._on_insider_alert,
                alert_threshold=self.insider_threshold,
            )
            # Pass sybil detector reference for cluster/funding source lookup
            self.monitor.sybil_detector = self.sybil_detector
            await self.monitor.start()

        # Start blockchain monitor for automatic sybil detection
        if mode in (ScannerMode.SYBIL_ONLY, ScannerMode.FULL):
            self.new_wallet_monitor = NewWalletMonitor(
                on_new_wallet=self._on_new_wallet,
                min_deposit_usd=1000.0,
            )

            # Start blockchain monitor for real-time funding detection
            self.blockchain_monitor = BlockchainMonitor(
                api_key=self.polygonscan_api_key,
                on_new_funding=self._on_blockchain_funding,
                poll_interval_s=15.0,
                min_funding_matic=0.5,
            )

            # Pre-populate known wallets from database to avoid false positives
            await self._load_known_wallets_to_monitor()

            # Start the monitor
            blockchain_started = await self.blockchain_monitor.start()
            if blockchain_started:
                logger.info("blockchain_monitor_started_for_sybil_detection")
            else:
                logger.warning("blockchain_monitor_failed_to_start",
                              message="Set POLYGONSCAN_API_KEY env var for automatic sybil detection")

        logger.info("scanner_started", mode=mode.value, stats=self.stats.to_dict())
        return True

    async def stop(self) -> None:
        """Stop the scanner service."""
        if not self.is_running:
            return

        logger.info("scanner_stopping", mode=self._mode.value)
        self._should_run = False

        # Stop trade monitor
        if self.monitor:
            await self.monitor.stop()
            self.monitor = None

        # Stop blockchain monitor
        if self.blockchain_monitor:
            await self.blockchain_monitor.stop()
            self.blockchain_monitor = None

        # Stop funding extraction
        if self._funding_extraction_task:
            self._funding_extraction_task.cancel()
            try:
                await self._funding_extraction_task
            except asyncio.CancelledError:
                pass
            self._funding_extraction_task = None

        self._mode = ScannerMode.STOPPED
        self.stats.mode = ScannerMode.STOPPED
        logger.info("scanner_stopped", stats=self.stats.to_dict())

    async def _auto_populate_data(self) -> None:
        """Auto-populate sybil detection data from files."""
        logger.info("auto_populating_data", data_dir=str(DATA_DIR))

        # Try checkpoint first (most up-to-date)
        checkpoint_file = DATA_DIR / "extraction_checkpoint.json"
        if checkpoint_file.exists():
            await self._import_from_checkpoint(checkpoint_file)
            return

        # Fall back to completed files
        files_to_import = [
            ("profitable_wallets", DATA_DIR / "profitable_wallets_full.json"),
            ("funding_sources", DATA_DIR / "funding_sources.json"),
            ("withdrawal_destinations", DATA_DIR / "withdrawal_destinations.json"),
        ]

        for file_type, file_path in files_to_import:
            if file_path.exists():
                await self._import_data_file(file_type, file_path)

    async def _import_from_checkpoint(self, checkpoint_file: Path) -> None:
        """Import data from extraction checkpoint."""
        logger.info("importing_from_checkpoint", file=str(checkpoint_file))

        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)

            results = checkpoint.get("results", [])
            if not results:
                logger.warning("checkpoint_empty")
                return

            with self.session_factory() as session:
                traders_imported = 0
                funding_sources = {}

                for wallet_data in results:
                    wallet_addr = wallet_data.get("wallet", "").lower()
                    if not wallet_addr:
                        continue

                    profit = Decimal(str(wallet_data.get("profit_usd", 0)))
                    funding_source = (wallet_data.get("funding_source") or "").lower()

                    # Upsert trader
                    existing = session.query(KnownProfitableTrader).filter(
                        KnownProfitableTrader.wallet_address == wallet_addr
                    ).first()

                    if not existing:
                        trader = KnownProfitableTrader(
                            wallet_address=wallet_addr,
                            profit_usd=profit,
                            funding_source=funding_source if funding_source else None,
                            funding_source_type=wallet_data.get("funding_source_type"),
                            primary_withdrawal_dest=wallet_data.get("primary_withdrawal_dest", "").lower() or None,
                            data_source="checkpoint",
                        )
                        session.add(trader)
                        traders_imported += 1

                    # Track funding sources
                    if funding_source:
                        if funding_source not in funding_sources:
                            funding_sources[funding_source] = {
                                "wallets": [],
                                "total_profit": Decimal("0"),
                            }
                        funding_sources[funding_source]["wallets"].append(wallet_addr)
                        funding_sources[funding_source]["total_profit"] += profit

                session.commit()

                # Import funding sources
                for addr, data in funding_sources.items():
                    existing = session.query(KnownFundingSource).filter(
                        KnownFundingSource.address == addr
                    ).first()

                    if not existing:
                        fs = KnownFundingSource(
                            address=addr,
                            funded_trader_wallets=data["wallets"],
                            funded_trader_count=len(data["wallets"]),
                            total_profit_funded=data["total_profit"],
                        )
                        session.add(fs)

                session.commit()

            self.stats.profitable_traders_loaded = traders_imported
            self.stats.funding_sources_loaded = len(funding_sources)
            logger.info(
                "checkpoint_imported",
                traders=traders_imported,
                funding_sources=len(funding_sources),
            )

        except Exception as e:
            logger.error("checkpoint_import_error", error=str(e))

    async def _import_data_file(self, file_type: str, file_path: Path) -> None:
        """Import a specific data file."""
        logger.info("importing_data_file", type=file_type, file=str(file_path))

        try:
            if file_type == "profitable_wallets":
                stats = self.sybil_detector.import_profitable_wallets(str(file_path))
            elif file_type == "funding_sources":
                stats = self.sybil_detector.import_funding_sources(str(file_path))
            elif file_type == "withdrawal_destinations":
                stats = self.sybil_detector.import_withdrawal_destinations(str(file_path))
            else:
                logger.warning("unknown_file_type", type=file_type)
                return

            logger.info("data_file_imported", type=file_type, stats=stats)

        except Exception as e:
            logger.error("data_file_import_error", type=file_type, error=str(e))

    async def _on_insider_alert(
        self,
        wallet_address: str,
        result: ScoringResult,
        accumulation: WalletAccumulation,
    ) -> None:
        """Handle insider detection alert from monitor.

        This is called when a wallet exceeds the insider score threshold.
        """
        self.stats.insider_wallets_scored += 1

        if result.score < self.insider_threshold:
            return

        logger.info(
            "insider_detected",
            wallet=wallet_address[:10] + "...",
            score=result.score,
            priority=result.priority,
            position_usd=float(accumulation.total_usd),
        )

        with self.session_factory() as session:
            # Check if already flagged
            existing = session.query(FlaggedWallet).filter(
                FlaggedWallet.wallet_address == wallet_address.lower()
            ).first()

            if existing:
                # Update score if higher
                if result.score > float(existing.insider_score or 0):
                    existing.insider_score = Decimal(str(result.score))
                    existing.priority = InsiderPriority(result.priority)
                    existing.signal_count = result.signal_count
                    existing.active_dimensions = result.active_dimensions
                    existing.last_scored_at = datetime.utcnow()
                    session.commit()
                return

            # Create new flagged wallet
            flagged = FlaggedWallet(
                wallet_address=wallet_address.lower(),
                flag_type=FlagType.INSIDER,
                insider_score=Decimal(str(result.score)),
                confidence_low=Decimal(str(result.confidence_low)),
                confidence_high=Decimal(str(result.confidence_high)),
                priority=InsiderPriority(result.priority),
                status=WalletStatus.NEW,
                score_account=Decimal(str(result.dimensions.get("account", 0))),
                score_trading=Decimal(str(result.dimensions.get("trading", 0))),
                score_behavioral=Decimal(str(result.dimensions.get("behavioral", 0))),
                score_contextual=Decimal(str(result.dimensions.get("contextual", 0))),
                score_cluster=Decimal(str(result.dimensions.get("cluster", 0))),
                signal_count=result.signal_count,
                active_dimensions=result.active_dimensions,
                signals_json=[{
                    "name": s.name,
                    "category": s.category,
                    "weight": s.weight,
                    "raw_value": str(s.raw_value) if s.raw_value else None,
                } for s in result.signals],
                downgraded=result.downgraded,
                downgrade_reason=result.downgrade_reason,
                total_position_usd=Decimal(str(accumulation.total_usd)),
            )
            session.add(flagged)
            session.commit()

            self.stats.insider_flags_generated += 1

            # Queue funding source extraction
            await self._queue_funding_extraction(wallet_address)

            # Create audit record
            audit = AuditTrailManager(session)
            audit.create_detection_record(
                flagged_wallet=flagged,
                scoring_result=result,
                market_ids=[accumulation.market_id],
            )
            session.commit()

        # Send alert
        if self.alert_manager:
            await self.alert_manager.send_alert(InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                wallet_address=wallet_address,
                score=result.score,
                priority=result.priority,
                signal_count=result.signal_count,
                market_id=accumulation.market_id,
                position_usd=float(accumulation.total_usd),
            ))
            self.stats.alerts_generated += 1
            if result.priority == "critical":
                self.stats.critical_alerts += 1

    async def _on_new_wallet(
        self,
        wallet_address: str,
        deposit_amount: Decimal,
    ) -> None:
        """Handle new wallet detection for sybil checking.

        Called when a new wallet is created and funded.
        """
        self.stats.sybil_checks_performed += 1

        # For now, we need to know the funding source
        # This would come from the new wallet monitor which tracks Polygonscan
        # For testing purposes, we'll check the wallet against known funders
        logger.debug(
            "new_wallet_for_sybil_check",
            wallet=wallet_address[:10] + "...",
            deposit=float(deposit_amount),
        )

    async def _on_blockchain_funding(
        self,
        new_wallet: str,
        funding_address: str,
        amount_matic: Decimal,
        tx_hash: str,
    ) -> Optional[SybilMatch]:
        """Handle new wallet funding detected by blockchain monitor.

        This is the automatic sybil detection callback - called whenever
        a new wallet receives its first funding on Polygon.

        Args:
            new_wallet: The newly funded wallet address
            funding_address: The address that funded it
            amount_matic: Amount of MATIC transferred
            tx_hash: Transaction hash

        Returns:
            SybilMatch if sybil detected, None otherwise
        """
        # Validate addresses aren't test/null addresses
        if self.sybil_detector:
            is_excluded, reason = self.sybil_detector.is_excluded(new_wallet)
            if is_excluded:
                logger.debug("blockchain_funding_skipped_excluded_wallet",
                           wallet=new_wallet[:10], reason=reason)
                return None

        # Perform sybil check
        match = await self.check_funding_source(
            new_wallet=new_wallet,
            funding_address=funding_address,
            funding_amount=amount_matic,
            funding_tx_hash=tx_hash,
        )

        if match:
            logger.warning(
                "auto_sybil_detected",
                new_wallet=new_wallet[:10] + "...",
                funding_source=funding_address[:10] + "...",
                linked_trader=match.linked_trader_wallet[:10] + "..." if match.linked_trader_wallet else None,
                linked_profit=float(match.linked_trader_profit) if match.linked_trader_profit else 0,
                tx_hash=tx_hash[:10] + "...",
            )

        return match

    async def _load_known_wallets_to_monitor(self) -> None:
        """Pre-populate blockchain monitor with known wallets.

        This prevents flagging existing wallets as "new" when we first
        see their transactions.
        """
        if not self.blockchain_monitor:
            return

        with self.session_factory() as session:
            # Load all known profitable trader wallets
            traders = session.query(KnownProfitableTrader.wallet_address).all()
            trader_wallets = [t[0] for t in traders if t[0]]

            # Load all funding sources
            sources = session.query(KnownFundingSource.address).all()
            source_wallets = [s[0] for s in sources if s[0]]

            # Load already flagged wallets
            flagged = session.query(FlaggedWallet.wallet_address).all()
            flagged_wallets = [f[0] for f in flagged if f[0]]

        all_known = set(trader_wallets + source_wallets + flagged_wallets)
        self.blockchain_monitor.add_known_wallets(list(all_known))

        logger.info("loaded_known_wallets_to_monitor", count=len(all_known))

    async def check_funding_source(
        self,
        new_wallet: str,
        funding_address: str,
        funding_amount: Optional[Decimal] = None,
        funding_tx_hash: Optional[str] = None,
    ) -> Optional[SybilMatch]:
        """Check if a funding address matches known profitable traders.

        This is the core sybil detection check.

        Args:
            new_wallet: The new wallet being funded
            funding_address: The address funding the new wallet
            funding_amount: Amount of funding (optional)
            funding_tx_hash: Transaction hash (optional)

        Returns:
            SybilMatch if detected, None otherwise
        """
        if not self.sybil_detector:
            logger.warning("sybil_detector_not_initialized")
            return None

        self.stats.sybil_checks_performed += 1

        match = self.sybil_detector.check_funding_source(funding_address)

        if not match.matched:
            return None

        self.stats.sybil_matches_found += 1

        logger.info(
            "sybil_match_found",
            new_wallet=new_wallet[:10] + "...",
            funding_address=funding_address[:10] + "...",
            match_type=match.match_type,
            linked_trader=match.linked_trader_wallet[:10] + "..." if match.linked_trader_wallet else None,
            linked_profit=float(match.linked_trader_profit) if match.linked_trader_profit else 0,
        )

        # Flag the wallet
        flagged = self.sybil_detector.flag_sybil_wallet(
            new_wallet=new_wallet,
            funding_address=funding_address,
            funding_amount_matic=funding_amount,
            funding_tx_hash=funding_tx_hash,
        )

        if flagged:
            self.stats.sybil_flags_generated += 1

            # Send alert
            if self.alert_manager:
                await self.alert_manager.send_alert(InsiderAlert(
                    alert_type=InsiderAlertType.FLAGGED_FUNDING_SOURCE,
                    wallet_address=new_wallet,
                    score=75.0,  # High score for sybil
                    priority="high",
                    signal_count=1,
                    extra_data={
                        "match_type": match.match_type,
                        "funding_address": funding_address,
                        "linked_trader": match.linked_trader_wallet,
                        "linked_profit": float(match.linked_trader_profit) if match.linked_trader_profit else 0,
                    },
                ))
                self.stats.alerts_generated += 1

        return match

    async def _queue_funding_extraction(self, wallet_address: str) -> None:
        """Queue a wallet for funding source extraction."""
        await self._funding_extraction_queue.put(wallet_address)
        self.stats.funding_extractions_queued += 1
        logger.debug("funding_extraction_queued", wallet=wallet_address[:10] + "...")

    async def _funding_extraction_worker(self) -> None:
        """Background worker to extract funding sources for flagged wallets."""
        logger.info("funding_extraction_worker_started")

        while self._should_run:
            try:
                # Get next wallet with timeout
                try:
                    wallet_address = await asyncio.wait_for(
                        self._funding_extraction_queue.get(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue

                await self._extract_funding_source(wallet_address)
                self.stats.funding_extractions_completed += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("funding_extraction_error", error=str(e))

        logger.info("funding_extraction_worker_stopped")

    async def _extract_funding_source(self, wallet_address: str) -> None:
        """Extract funding source for a wallet via Polygonscan.

        When an insider is flagged, we extract their funding source and add it
        to the insider funding sources table for future matching.
        """
        if not self.polygonscan_api_key:
            logger.debug("no_polygonscan_api_key_skipping_extraction")
            return

        logger.info("extracting_funding_source", wallet=wallet_address[:10] + "...")

        # Import here to avoid circular dependency
        import httpx

        # Etherscan V2 API for Polygon
        url = "https://api.etherscan.io/v2/api"
        params = {
            "chainid": "137",
            "module": "account",
            "action": "txlist",
            "address": wallet_address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": "50",
            "sort": "asc",
            "apikey": self.polygonscan_api_key,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                data = response.json()

            if data.get("status") != "1" or not isinstance(data.get("result"), list):
                logger.warning("polygonscan_no_results", wallet=wallet_address[:10])
                return

            # Find first significant inbound transfer
            for tx in data["result"]:
                to_addr = tx.get("to", "").lower()
                from_addr = tx.get("from", "").lower()
                value_wei = int(tx.get("value", "0"))
                value_matic = value_wei / 1e18

                if to_addr == wallet_address.lower() and value_matic >= 0.1:
                    # Found funding source
                    await self._add_insider_funding_source(
                        funding_address=from_addr,
                        funded_wallet=wallet_address,
                        amount_matic=Decimal(str(value_matic)),
                        tx_hash=tx.get("hash"),
                    )
                    self.stats.funding_sources_discovered += 1
                    return

            logger.debug("no_funding_source_found", wallet=wallet_address[:10])

        except Exception as e:
            logger.error("polygonscan_error", error=str(e))

    async def _add_insider_funding_source(
        self,
        funding_address: str,
        funded_wallet: str,
        amount_matic: Decimal,
        tx_hash: Optional[str] = None,
    ) -> None:
        """Add a funding source to the insider funding sources table.

        This creates a feedback loop - funding sources discovered from flagged
        insider wallets can be used to detect future sybil accounts.
        """
        with self.session_factory() as session:
            # Check if source already exists
            existing = session.query(FlaggedFundingSource).filter(
                FlaggedFundingSource.funding_address == funding_address
            ).first()

            if existing:
                existing.associated_wallet_count += 1
                existing.last_activity = datetime.utcnow()
            else:
                source = FlaggedFundingSource(
                    funding_address=funding_address,
                    source_type="eoa",
                    associated_wallet_count=1,
                    risk_level="high",
                    reason=f"Funded flagged insider wallet {funded_wallet[:10]}...",
                )
                session.add(source)

            # Update flagged wallet with funding source
            flagged = session.query(FlaggedWallet).filter(
                FlaggedWallet.wallet_address == funded_wallet.lower()
            ).first()

            if flagged and not flagged.funding_match_address:
                flagged.funding_match_address = funding_address

            session.commit()

        logger.info(
            "insider_funding_source_added",
            funding_address=funding_address[:10] + "...",
            funded_wallet=funded_wallet[:10] + "...",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current scanner statistics."""
        stats = self.stats.to_dict()

        # Add blockchain monitor stats if running
        if self.blockchain_monitor and self.blockchain_monitor.is_running:
            stats["blockchain_monitor"] = self.blockchain_monitor.get_stats()

        return stats

    async def manual_score_wallet(
        self,
        wallet_address: str,
        positions: Optional[List[Dict]] = None,
        account_age_days: Optional[int] = None,
        transaction_count: Optional[int] = None,
        market_category: Optional[str] = None,
        event_hours_away: Optional[float] = None,
    ) -> ScoringResult:
        """Manually score a wallet against insider patterns.

        Useful for testing and ad-hoc analysis.
        """
        market_cat = None
        if market_category:
            try:
                market_cat = MarketCategory(market_category)
            except ValueError:
                pass

        result = self.scorer.score_wallet(
            wallet_address=wallet_address,
            account_age_days=account_age_days,
            transaction_count=transaction_count,
            positions=positions,
            market_category=market_cat,
            event_hours_away=event_hours_away,
        )

        self.stats.insider_wallets_scored += 1
        return result

    async def reload_sybil_data(self) -> Dict[str, int]:
        """Reload sybil detection data and rebuild index.

        Call this after new data files are generated.
        """
        if not self.sybil_detector:
            self.sybil_detector = SybilDetector(self.session_factory)

        await self._auto_populate_data()
        return self.sybil_detector.rebuild_index()
