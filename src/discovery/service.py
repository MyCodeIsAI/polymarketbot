"""Discovery Service for finding copy-trading candidates.

Redesigned two-phase approach based on research:
- "0.04% of accounts make 70% of profits"
- Cast a WIDE NET first (2000+ accounts)
- Store everything in database with checkpointing
- Sort/rank later based on profitability metrics

Phase 1: Collection (minimal filtering)
- Collect from all leaderboard categories
- Store basic metrics in database
- Checkpoint progress for long-running scans

Phase 2: Analysis & Ranking
- Light scan for basic profitability metrics
- Deep analysis on top candidates
- Score and rank by composite score
- Can pause/resume at any time
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any, Callable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

# Project root - works on both local dev and VPS
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
from sqlalchemy import select, func

from ..api.data import DataAPIClient
from ..api.gamma import GammaAPIClient
from ..api.base import BaseAPIClient
from ..utils.logging import get_logger
from ..database.models import (
    DiscoveryScanRecord,
    DiscoveredAccount,
    DiscoveryScanStatus,
    DiscoveredAccountStatus,
)

from .models import DiscoveryMode, ScanStatus
from .analyzer import (
    AccountAnalyzer,
    LeaderboardEntry,
    QuickFilterResult,
    LightScanResult,
    DeepAnalysisResult,
)
from .scoring import ScoringEngine, ScoringResult, MODE_CONFIGS

logger = get_logger(__name__)


# Leaderboard categories available from the API
LEADERBOARD_CATEGORIES = [
    "WEATHER",
    "ECONOMICS",
    "TECH",
    "FINANCE",
    "CRYPTO",
    "POLITICS",
    "SPORTS",
    "CULTURE",
    "OVERALL",
]


@dataclass
class ScanConfig:
    """Configuration for a discovery scan.

    Philosophy: "0.04% of accounts make 70% of profits"
    - Cast a wide net first (collect 2000+ accounts)
    - Store everything, sort later
    - Minimal hard filters, rely on scoring
    """

    mode: DiscoveryMode = DiscoveryMode.WIDE_NET_PROFITABILITY

    # Seed collection settings - CAST WIDE NET
    categories: list[str] = None
    market_ids: list[str] = None
    reference_wallet: str = None

    # Analysis settings - SCALED UP
    lookback_days: int = 90
    min_score_threshold: float = 20.0  # Very low - collect broadly
    max_candidates: int = 2000  # Phase 1 max - SCALED UP from 500
    analyze_top_n: int = 500  # Phase 2 max - SCALED UP from 100

    # Filtering - RELAXED
    min_trades: int = 5  # Minimal - let scoring sort
    max_trades: int = None  # Max trades - filter out high-frequency bots
    min_profit: float = None  # Min profit USD - filter out small accounts
    max_profit: float = None  # Max profit USD - filter out whales
    max_avg_position_size: float = 50000.0  # Very relaxed
    exclude_wallets: list[str] = None

    # Phase limits - SCALED UP
    max_phase2: int = 500  # Light scan many accounts
    max_phase3: int = 100  # Deep analyze top 100

    # Checkpointing
    checkpoint_interval: int = 50  # Save progress every N accounts
    persist_to_db: bool = True  # Store results in database

    # Resume support
    resume_from_scan_id: int = None  # Resume an existing scan

    # Insider detection specific
    fresh_account_days: int = 30
    large_position_threshold: float = 5000.0

    def __post_init__(self):
        if self.categories is None:
            # Wide net: ALL categories by default
            self.categories = LEADERBOARD_CATEGORIES.copy()
        if self.market_ids is None:
            self.market_ids = []
        if self.exclude_wallets is None:
            self.exclude_wallets = []

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "categories": self.categories,
            "market_ids": self.market_ids,
            "reference_wallet": self.reference_wallet,
            "lookback_days": self.lookback_days,
            "min_score_threshold": self.min_score_threshold,
            "max_candidates": self.max_candidates,
            "analyze_top_n": self.analyze_top_n,
            "min_trades": self.min_trades,
            "max_trades": self.max_trades,
            "min_profit": self.min_profit,
            "max_profit": self.max_profit,
            "max_avg_position_size": self.max_avg_position_size,
            "max_phase2": self.max_phase2,
            "max_phase3": self.max_phase3,
            "checkpoint_interval": self.checkpoint_interval,
            "persist_to_db": self.persist_to_db,
            "fresh_account_days": self.fresh_account_days,
            "large_position_threshold": self.large_position_threshold,
        }

    @classmethod
    def wide_net_preset(cls) -> "ScanConfig":
        """Preset for casting the widest net - recommended for initial discovery."""
        return cls(
            mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
            categories=LEADERBOARD_CATEGORIES.copy(),
            max_candidates=3000,
            analyze_top_n=1000,
            max_phase2=1000,
            max_phase3=200,
            min_score_threshold=15.0,
            min_trades=3,
        )

    @classmethod
    def niche_focused_preset(cls) -> "ScanConfig":
        """Preset for finding niche market specialists."""
        return cls(
            mode=DiscoveryMode.NICHE_SPECIALIST,
            categories=["WEATHER", "ECONOMICS", "TECH", "FINANCE", "CRYPTO"],
            max_candidates=1000,
            analyze_top_n=300,
            max_phase2=300,
            max_phase3=75,
            min_score_threshold=35.0,
        )

    @classmethod
    def quick_scan_preset(cls) -> "ScanConfig":
        """Preset for quick testing - smaller sample."""
        return cls(
            mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
            categories=["ECONOMICS", "TECH"],
            max_candidates=200,
            analyze_top_n=50,
            max_phase2=50,
            max_phase3=15,
        )


@dataclass
class ScanProgress:
    """Progress information for a running scan."""

    scan_id: int
    status: ScanStatus
    progress_pct: int
    current_step: str
    current_phase: str
    candidates_found: int
    candidates_analyzed: int
    candidates_passed: int
    errors: list[str] = field(default_factory=list)

    # Phase-specific stats
    phase1_total: int = 0
    phase1_passed: int = 0
    phase2_total: int = 0
    phase2_passed: int = 0
    phase3_total: int = 0
    phase3_passed: int = 0

    # API efficiency
    api_calls_made: int = 0


@dataclass
class DiscoveryScan:
    """Record of a discovery scan."""

    mode: DiscoveryMode
    config: dict
    status: ScanStatus = ScanStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: int = 0

    candidates_found: int = 0
    candidates_analyzed: int = 0
    candidates_passed: int = 0

    progress_pct: int = 0
    current_step: str = ""
    current_phase: str = ""

    errors: Optional[list[str]] = None
    id: Optional[int] = None


class LeaderboardClient(BaseAPIClient):
    """Client for the Polymarket leaderboard API."""

    DEFAULT_BASE_URL = "https://data-api.polymarket.com"

    def __init__(self, timeout_s: float = 15.0):
        super().__init__(base_url=self.DEFAULT_BASE_URL, timeout_s=timeout_s)

    async def get_leaderboard(
        self,
        category: str = "OVERALL",
        time_period: str = "ALL",
        order_by: str = "PNL",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Get leaderboard rankings.

        Args:
            category: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, WEATHER, ECONOMICS, TECH, FINANCE
            time_period: DAY, WEEK, MONTH, ALL
            order_by: PNL or VOL
            limit: 1-50
            offset: 0-1000

        Returns:
            List of leaderboard entries
        """
        response = await self.get(
            "/v1/leaderboard",
            params={
                "category": category.upper(),
                "timePeriod": time_period.upper(),
                "orderBy": order_by.upper(),
                "limit": min(50, limit),
                "offset": offset,
            },
        )

        return response.data if isinstance(response.data, list) else response.data.get("leaderboard", [])

    async def get_leaderboard_full(
        self,
        category: str = "OVERALL",
        limit: int = 500,  # SCALED UP from 100
    ) -> list[LeaderboardEntry]:
        """Get full leaderboard data for wide-net collection.

        Fetches multiple time periods and order-by criteria to maximize coverage.
        Scaled up to support 2000+ total account collection.
        """
        seen_wallets = set()
        entries = []

        # Fetch from multiple angles to maximize coverage
        fetch_configs = [
            ("WEEK", "PNL"),
            ("MONTH", "PNL"),
            ("ALL", "PNL"),
            ("WEEK", "VOL"),  # Also get volume leaders
            ("MONTH", "VOL"),
        ]

        for time_period, order_by in fetch_configs:
            if len(entries) >= limit:
                break

            try:
                offset = 0
                max_offset = 500  # SCALED UP from 200 - fetch more per category
                while len(entries) < limit and offset < max_offset:
                    raw_entries = await self.get_leaderboard(
                        category=category,
                        time_period=time_period,
                        order_by=order_by,
                        limit=50,
                        offset=offset,
                    )

                    if not raw_entries:
                        break

                    for raw in raw_entries:
                        wallet = raw.get("proxyWallet", raw.get("address", "")).lower()
                        if wallet and wallet not in seen_wallets:
                            seen_wallets.add(wallet)

                            # Parse leaderboard data
                            entry = LeaderboardEntry(
                                wallet_address=wallet,
                                rank=raw.get("rank", 0),
                                total_pnl=Decimal(str(raw.get("pnl", "0"))),
                                volume=Decimal(str(raw.get("volume", "0"))),
                                num_trades=int(raw.get("tradeCount", 0)),
                                position_count=int(raw.get("positionCount", 0)),
                                categories=[category],
                            )
                            entries.append(entry)

                    offset += 50
                    await asyncio.sleep(0.05)  # Slightly faster rate

            except Exception as e:
                logger.warning("leaderboard_fetch_error", category=category, period=time_period, error=str(e))

        logger.info("leaderboard_category_fetched", category=category, count=len(entries))
        return entries[:limit]


class DiscoveryService:
    """Service for discovering and analyzing copy-trading candidates.

    Redesigned for wide-net collection with database persistence:
    - Collects 2000+ accounts from all leaderboard categories
    - Stores everything in database with checkpoint support
    - Can pause/resume long-running scans
    - Sorts by profitability metrics, not category gatekeeping
    """

    def __init__(
        self,
        data_client: Optional[DataAPIClient] = None,
        gamma_client: Optional[GammaAPIClient] = None,
        leaderboard_client: Optional[LeaderboardClient] = None,
        db_session: Optional[AsyncSession] = None,
    ):
        """Initialize discovery service.

        Args:
            data_client: Data API client (created if not provided)
            gamma_client: Gamma API client (created if not provided)
            leaderboard_client: Leaderboard client (created if not provided)
            db_session: SQLAlchemy async session for persistence (optional)
        """
        self._data_client = data_client
        self._gamma_client = gamma_client
        self._leaderboard_client = leaderboard_client
        self._db_session = db_session
        self._owns_clients = data_client is None

        self._analyzer: Optional[AccountAnalyzer] = None

        # Scan state
        self._current_scan: Optional[DiscoveryScan] = None
        self._current_scan_record: Optional[DiscoveryScanRecord] = None
        self._scan_cancelled = False
        self._scan_paused = False
        self._scan_progress: Optional[ScanProgress] = None

    async def __aenter__(self) -> "DiscoveryService":
        """Async context manager entry."""
        if self._owns_clients:
            self._data_client = DataAPIClient()
            self._gamma_client = GammaAPIClient()
            self._leaderboard_client = LeaderboardClient()

            await self._data_client.__aenter__()
            await self._gamma_client.__aenter__()
            await self._leaderboard_client.__aenter__()

        # Default to wide net mode for maximum collection
        self._analyzer = AccountAnalyzer(
            self._data_client,
            self._gamma_client,
            mode=DiscoveryMode.WIDE_NET_PROFITABILITY,
        )
        await self._analyzer.__aenter__()

        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._analyzer:
            await self._analyzer.__aexit__(*args)

        if self._owns_clients:
            if self._data_client:
                await self._data_client.__aexit__(*args)
            if self._gamma_client:
                await self._gamma_client.__aexit__(*args)
            if self._leaderboard_client:
                await self._leaderboard_client.__aexit__(*args)

    # =========================================================================
    # PERSISTENCE & CHECKPOINT METHODS
    # =========================================================================

    async def _create_scan_record(self, config: ScanConfig) -> Optional[DiscoveryScanRecord]:
        """Create a scan record in the database."""
        if not self._db_session:
            return None

        record = DiscoveryScanRecord(
            mode=config.mode.value,
            config_json=config.to_dict(),
            status=DiscoveryScanStatus.COLLECTING,
            started_at=datetime.utcnow(),
        )
        self._db_session.add(record)
        await self._db_session.commit()
        await self._db_session.refresh(record)

        self._current_scan_record = record
        return record

    async def _persist_collected_account(
        self,
        entry: LeaderboardEntry,
        scan_id: Optional[int] = None,
    ) -> Optional[DiscoveredAccount]:
        """Persist a collected account to the database."""
        if not self._db_session:
            return None

        # Check if account already exists
        existing = await self._db_session.execute(
            select(DiscoveredAccount).where(
                DiscoveredAccount.wallet_address == entry.wallet_address
            )
        )
        account = existing.scalar_one_or_none()

        if account:
            # Update existing with latest leaderboard data
            account.leaderboard_rank = entry.rank
            account.total_pnl = entry.total_pnl
            account.total_volume = entry.volume
            account.num_trades = entry.num_trades
            account.position_count = entry.position_count
            account.updated_at = datetime.utcnow()
        else:
            # Create new
            account = DiscoveredAccount(
                scan_id=scan_id,
                wallet_address=entry.wallet_address,
                status=DiscoveredAccountStatus.COLLECTED,
                leaderboard_rank=entry.rank,
                leaderboard_category=entry.categories[0] if entry.categories else None,
                total_pnl=entry.total_pnl,
                total_volume=entry.volume,
                num_trades=entry.num_trades,
                position_count=entry.position_count,
            )
            self._db_session.add(account)

        return account

    async def _update_account_light_scan(
        self,
        wallet: str,
        result: LightScanResult,
    ) -> None:
        """Update account with light scan results."""
        if not self._db_session:
            return

        existing = await self._db_session.execute(
            select(DiscoveredAccount).where(
                DiscoveredAccount.wallet_address == wallet
            )
        )
        account = existing.scalar_one_or_none()

        if account and result.metrics:
            m = result.metrics
            account.status = DiscoveredAccountStatus.LIGHT_SCANNED
            account.win_rate = Decimal(str(m.win_rate)) if hasattr(m, 'win_rate') else None
            account.avg_position_size = m.avg_position_size_usd
            account.account_age_days = m.account_age_days
            account.active_days = m.active_days
            account.unique_markets = m.unique_markets_traded
            account.light_scanned_at = datetime.utcnow()

    async def _update_account_deep_analysis(
        self,
        wallet: str,
        result: DeepAnalysisResult,
    ) -> None:
        """Update account with deep analysis results."""
        if not self._db_session:
            return

        existing = await self._db_session.execute(
            select(DiscoveredAccount).where(
                DiscoveredAccount.wallet_address == wallet
            )
        )
        account = existing.scalar_one_or_none()

        if account:
            account.status = DiscoveredAccountStatus.DEEP_ANALYZED
            account.deep_analyzed_at = datetime.utcnow()

            if result.pl_metrics:
                pl = result.pl_metrics
                account.sharpe_ratio = Decimal(str(pl.sharpe_ratio))
                account.sortino_ratio = Decimal(str(pl.sortino_ratio))
                account.profit_factor = Decimal(str(pl.profit_factor))
                account.max_drawdown_pct = Decimal(str(pl.max_drawdown_pct))
                account.largest_win_pct = Decimal(str(pl.largest_win_pct_of_total))
                account.top3_wins_pct = Decimal(str(pl.top_3_wins_pct_of_total))

            if result.pattern_metrics:
                account.category_breakdown = result.pattern_metrics.category_breakdown

            if result.scoring_result:
                sr = result.scoring_result
                account.composite_score = Decimal(str(sr.composite_score))
                account.passes_threshold = sr.passes_threshold
                account.red_flags = sr.red_flags
                account.red_flag_count = sr.red_flag_count

            # Store full analysis for detailed review
            account.full_analysis_json = result.to_dict()

    async def _save_checkpoint(
        self,
        phase: str,
        processed_wallets: list[str],
        progress_pct: int,
    ) -> None:
        """Save checkpoint for resume capability (both DB and file)."""
        checkpoint_data = {
            "phase": phase,
            "processed_wallets": processed_wallets,  # Keep ALL for resume
            "processed_count": len(processed_wallets),
            "progress_pct": progress_pct,
            "timestamp": datetime.utcnow().isoformat(),
            "scan_config": self._current_scan.config if self._current_scan else {},
        }

        # Save to file for crash recovery
        self._save_checkpoint_to_file(checkpoint_data)

        # Also save to DB if available
        if self._db_session and self._current_scan_record:
            self._current_scan_record.checkpoint_data = {
                "phase": phase,
                "processed_wallets": processed_wallets[-100:],  # DB gets truncated list
                "processed_count": len(processed_wallets),
                "timestamp": checkpoint_data["timestamp"],
            }
            self._current_scan_record.progress_pct = progress_pct
            self._current_scan_record.current_phase = phase
            self._current_scan_record.last_checkpoint_at = datetime.utcnow()
            await self._db_session.commit()

    def _save_checkpoint_to_file(self, checkpoint_data: dict) -> None:
        """Save checkpoint to JSON file for crash recovery."""
        from pathlib import Path
        checkpoint_file = DATA_DIR / "scan_checkpoint.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug("checkpoint_saved_to_file", phase=checkpoint_data["phase"],
                        processed=checkpoint_data["processed_count"])
        except Exception as e:
            logger.error("checkpoint_file_save_failed", error=str(e))

    @staticmethod
    def load_checkpoint_from_file() -> Optional[dict]:
        """Load checkpoint from file for resume."""
        from pathlib import Path
        checkpoint_file = DATA_DIR / "scan_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file) as f:
                data = json.load(f)
            logger.info("checkpoint_loaded", phase=data.get("phase"),
                       processed=data.get("processed_count", 0))
            return data
        except Exception as e:
            logger.error("checkpoint_load_failed", error=str(e))
            return None

    @staticmethod
    def clear_checkpoint_file() -> None:
        """Clear the checkpoint file after successful completion."""
        from pathlib import Path
        checkpoint_file = DATA_DIR / "scan_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("checkpoint_file_cleared")

    async def _commit_batch(self) -> None:
        """Commit current batch to database."""
        if self._db_session:
            await self._db_session.commit()

    async def get_discovered_accounts(
        self,
        min_score: float = 0,
        status: Optional[DiscoveredAccountStatus] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "composite_score",
    ) -> list[dict]:
        """Query discovered accounts from database.

        Args:
            min_score: Minimum composite score filter
            status: Filter by status
            limit: Max results
            offset: Pagination offset
            order_by: Sort field (composite_score, total_pnl, win_rate)

        Returns:
            List of account dictionaries
        """
        if not self._db_session:
            return []

        query = select(DiscoveredAccount)

        if min_score > 0:
            query = query.where(DiscoveredAccount.composite_score >= min_score)
        if status:
            query = query.where(DiscoveredAccount.status == status)

        # Order by
        if order_by == "total_pnl":
            query = query.order_by(DiscoveredAccount.total_pnl.desc())
        elif order_by == "win_rate":
            query = query.order_by(DiscoveredAccount.win_rate.desc())
        else:
            query = query.order_by(DiscoveredAccount.composite_score.desc())

        query = query.limit(limit).offset(offset)

        result = await self._db_session.execute(query)
        accounts = result.scalars().all()

        return [a.to_dict() for a in accounts]

    async def get_scan_summary(self) -> dict:
        """Get summary statistics of discovered accounts."""
        if not self._db_session:
            return {}

        # Count by status
        status_counts = {}
        for status in DiscoveredAccountStatus:
            result = await self._db_session.execute(
                select(func.count(DiscoveredAccount.id)).where(
                    DiscoveredAccount.status == status
                )
            )
            status_counts[status.value] = result.scalar() or 0

        # Score distribution
        result = await self._db_session.execute(
            select(
                func.count(DiscoveredAccount.id),
                func.avg(DiscoveredAccount.composite_score),
                func.max(DiscoveredAccount.composite_score),
            ).where(DiscoveredAccount.composite_score.isnot(None))
        )
        row = result.one()

        return {
            "total_accounts": sum(status_counts.values()),
            "status_breakdown": status_counts,
            "scored_accounts": row[0] or 0,
            "avg_score": float(row[1]) if row[1] else 0,
            "max_score": float(row[2]) if row[2] else 0,
        }

    def pause_scan(self) -> None:
        """Pause the current scan (will checkpoint and stop)."""
        self._scan_paused = True
        logger.info("scan_pause_requested")

    def get_scoring_config(self, mode: Optional[DiscoveryMode] = None) -> dict:
        """Get current scoring configuration for UI display.

        Args:
            mode: Discovery mode (uses current if not specified)

        Returns:
            Configuration dict with hard_filters, soft_filters, etc.
        """
        if mode:
            self._analyzer.set_mode(mode)
        return self._analyzer.scoring_engine.get_config()

    def update_scoring_config(self, updates: dict) -> dict:
        """Update scoring configuration from UI.

        Args:
            updates: Dict with filter updates like:
                {"hard_filters": {"min_trades": {"threshold": 150}}}

        Returns:
            Updated configuration
        """
        self._analyzer.scoring_engine.update_config(updates)
        return self._analyzer.scoring_engine.get_config()

    async def run_scan(
        self,
        config: ScanConfig,
        progress_callback: Optional[Callable[[ScanProgress], Any]] = None,
    ) -> tuple[list[dict], DiscoveryScan]:
        """Run a complete multi-phase discovery scan with persistence.

        Wide-net approach:
        - Collects 2000+ accounts from all categories
        - Persists to database with checkpointing
        - Can be paused/resumed
        - Focuses on profitability, not category gatekeeping

        Args:
            config: Scan configuration (use ScanConfig.wide_net_preset() for max collection)
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (discovered accounts, scan record)
        """
        self._scan_cancelled = False
        self._scan_paused = False
        self._analyzer.set_mode(config.mode)

        # Update scoring thresholds from user config
        # This ensures user's min_trades and other filters are respected
        scoring_updates = {
            "min_composite_score": config.min_score_threshold,
        }

        # Pass user's min_trades to the hard filter in scoring engine
        if config.min_trades:
            scoring_updates["hard_filters"] = {
                "min_trades": {"threshold": config.min_trades}
            }

        self._analyzer.scoring_engine.update_config(scoring_updates)

        scan = DiscoveryScan(
            mode=config.mode,
            config=config.to_dict(),
            status=ScanStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        self._current_scan = scan

        # Create database record if persistence enabled
        if config.persist_to_db:
            await self._create_scan_record(config)

        progress = ScanProgress(
            scan_id=self._current_scan_record.id if self._current_scan_record else 0,
            status=ScanStatus.RUNNING,
            progress_pct=0,
            current_step="Initializing...",
            current_phase="init",
            candidates_found=0,
            candidates_analyzed=0,
            candidates_passed=0,
        )
        self._scan_progress = progress

        results: list[dict] = []
        errors: list[str] = []
        processed_wallets: list[str] = []

        try:
            # ================================================================
            # PHASE 0: Collect candidate wallets from ALL leaderboards
            # ================================================================
            await self._update_progress(
                progress, 5, f"Collecting candidates from {len(config.categories)} categories...",
                "collect", progress_callback
            )

            leaderboard_entries = await self._collect_leaderboard_candidates(config)

            if self._scan_cancelled or self._scan_paused:
                scan.status = ScanStatus.CANCELLED if self._scan_cancelled else ScanStatus.RUNNING
                return results, scan

            progress.candidates_found = len(leaderboard_entries)
            scan.candidates_found = len(leaderboard_entries)

            logger.info("candidates_collected", count=len(leaderboard_entries))

            # Persist collected accounts to database
            if config.persist_to_db and self._db_session:
                scan_id = self._current_scan_record.id if self._current_scan_record else None
                for i, entry in enumerate(leaderboard_entries):
                    await self._persist_collected_account(entry, scan_id)
                    if i % config.checkpoint_interval == 0:
                        await self._commit_batch()
                await self._commit_batch()

                if self._current_scan_record:
                    self._current_scan_record.collected_count = len(leaderboard_entries)

            await self._update_progress(
                progress, 10,
                f"Collected {len(leaderboard_entries)} candidates (stored in DB)",
                "collect", progress_callback
            )

            # ================================================================
            # PHASE 1: Quick Filter (no API calls) - VERY PERMISSIVE
            # ================================================================
            await self._update_progress(
                progress, 15, "Phase 1: Quick filtering (permissive)...",
                "phase1", progress_callback
            )

            phase1_passed, phase1_rejected = self._analyzer.batch_quick_filter(leaderboard_entries)

            # Track leaderboard P/L for enrichment after Phase 3
            # This is the authoritative P/L from the leaderboard (not computed)
            leaderboard_pnl_map = {r.wallet_address.lower(): r.total_pnl for r in phase1_passed}

            progress.phase1_total = len(leaderboard_entries)
            progress.phase1_passed = len(phase1_passed)

            logger.info("phase1_complete", passed=len(phase1_passed), rejected=len(phase1_rejected))

            if self._scan_cancelled:
                scan.status = ScanStatus.CANCELLED
                return results, scan

            # Check for pause
            if self._scan_paused:
                await self._save_checkpoint("phase1", [], progress.progress_pct)
                scan.status = ScanStatus.RUNNING
                scan.current_step = "Paused after Phase 1"
                return results, scan

            await self._update_progress(
                progress, 25,
                f"Phase 1: {len(phase1_passed)}/{len(leaderboard_entries)} passed quick filter",
                "phase1", progress_callback
            )

            # Early exit if no candidates passed Phase 1
            if len(phase1_passed) == 0:
                logger.warning("phase1_zero_passed", total=len(leaderboard_entries))
                await self._update_progress(
                    progress, 100,
                    "No candidates passed Phase 1 quick filter",
                    "complete", progress_callback
                )
                scan.status = ScanStatus.COMPLETED
                scan.completed_at = datetime.utcnow()
                return results, scan

            # ================================================================
            # PHASE 2: Light Scan (1 API call per candidate)
            # ================================================================
            phase2_candidates = [r.wallet_address for r in phase1_passed[:config.max_phase2]]

            await self._update_progress(
                progress, 30, f"Phase 2: Light scanning {len(phase2_candidates)} candidates...",
                "phase2", progress_callback
            )

            async def phase2_progress_cb(completed: int, total: int):
                pct = 30 + int((completed / total) * 30)
                await self._update_progress(
                    progress, pct,
                    f"Phase 2: Scanning {completed}/{total}...",
                    "phase2", progress_callback
                )

                # Checkpoint every N accounts
                if completed % config.checkpoint_interval == 0 and config.persist_to_db:
                    await self._save_checkpoint("phase2", processed_wallets, pct)

                # Check for pause
                if self._scan_paused:
                    return  # Will be handled in main loop

            phase2_passed, phase2_rejected = await self._analyzer.batch_light_scan(
                phase2_candidates,
                max_concurrent=5,
                progress_callback=phase2_progress_cb,
            )

            # Persist light scan results
            if config.persist_to_db:
                for result in phase2_passed + phase2_rejected:
                    await self._update_account_light_scan(result.wallet_address, result)
                    processed_wallets.append(result.wallet_address)
                await self._commit_batch()

                if self._current_scan_record:
                    self._current_scan_record.light_scanned_count = len(phase2_candidates)

            progress.phase2_total = len(phase2_candidates)
            progress.phase2_passed = len(phase2_passed)
            progress.api_calls_made = self._analyzer.get_stats()["api_calls"]

            logger.info("phase2_complete", passed=len(phase2_passed), rejected=len(phase2_rejected))

            if self._scan_cancelled:
                scan.status = ScanStatus.CANCELLED
                return results, scan

            if self._scan_paused:
                await self._save_checkpoint("phase2", processed_wallets, progress.progress_pct)
                scan.status = ScanStatus.RUNNING
                scan.current_step = "Paused after Phase 2"
                return results, scan

            await self._update_progress(
                progress, 60,
                f"Phase 2: {len(phase2_passed)}/{len(phase2_candidates)} passed light scan",
                "phase2", progress_callback
            )

            # Early exit if no candidates passed Phase 2
            if len(phase2_passed) == 0:
                logger.warning("phase2_zero_passed", total=len(phase2_candidates))
                await self._update_progress(
                    progress, 100,
                    "No candidates passed Phase 2 light scan",
                    "complete", progress_callback
                )
                scan.status = ScanStatus.COMPLETED
                scan.completed_at = datetime.utcnow()
                return results, scan

            # ================================================================
            # PHASE 3: Deep Analysis (2-3 API calls per candidate)
            # ================================================================
            phase3_candidates = [r.wallet_address for r in phase2_passed[:config.max_phase3]]

            await self._update_progress(
                progress, 65, f"Phase 3: Deep analyzing {len(phase3_candidates)} candidates...",
                "phase3", progress_callback
            )

            async def phase3_progress_cb(completed: int, total: int, wallet: str):
                pct = 65 + int((completed / total) * 30)
                await self._update_progress(
                    progress, pct,
                    f"Phase 3: Analyzing {completed}/{total} ({wallet[:8]}...)",
                    "phase3", progress_callback
                )

                # Checkpoint
                if completed % 10 == 0 and config.persist_to_db:
                    await self._save_checkpoint("phase3", processed_wallets, pct)

            # For wide_net mode, use extended lookback to capture older profitable accounts
            # Need full history for accurate P/L and win rate metrics
            effective_lookback = config.lookback_days
            if config.mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
                effective_lookback = max(config.lookback_days, 1825)  # 5 years - full Polymarket history

            deep_results = await self._analyzer.batch_deep_analysis(
                phase3_candidates,
                lookback_days=effective_lookback,
                max_concurrent=3,
                progress_callback=phase3_progress_cb,
                max_trades=config.analyze_top_n,
            )

            # Persist deep analysis results
            if config.persist_to_db:
                for result in deep_results:
                    await self._update_account_deep_analysis(result.wallet_address, result)
                await self._commit_batch()

                if self._current_scan_record:
                    self._current_scan_record.deep_analyzed_count = len(deep_results)

            # Filter to passing results
            # DEBUG: Log what we're filtering
            has_scoring = sum(1 for r in deep_results if r.scoring_result)
            passes_threshold = sum(1 for r in deep_results if r.scoring_result and r.scoring_result.passes_threshold)
            logger.info("phase3_filter_debug",
                        total_results=len(deep_results),
                        has_scoring_result=has_scoring,
                        passes_threshold=passes_threshold)

            # Log first few results for debugging
            for i, r in enumerate(deep_results[:5]):
                if r.scoring_result:
                    logger.info("phase3_sample_result",
                                idx=i,
                                wallet=r.wallet_address[:10],
                                score=r.scoring_result.composite_score,
                                passes=r.scoring_result.passes_threshold,
                                hard_passed=r.scoring_result.hard_filter_passed)
                else:
                    logger.info("phase3_sample_no_score",
                                idx=i,
                                wallet=r.wallet_address[:10],
                                error=r.error)

            passed_results = [
                r for r in deep_results
                if r.scoring_result and r.scoring_result.passes_threshold
            ]

            progress.phase3_total = len(phase3_candidates)
            progress.phase3_passed = len(passed_results)
            progress.candidates_analyzed = len(deep_results)
            progress.candidates_passed = len(passed_results)
            progress.api_calls_made = self._analyzer.get_stats()["api_calls"]

            logger.info("phase3_complete", analyzed=len(deep_results), passed=len(passed_results))

            # Enrich results with authoritative leaderboard P/L
            # Use the mapping we built in Phase 1 - this is more reliable than DB lookup
            for result in passed_results:
                wallet_lower = result.wallet_address.lower()
                if wallet_lower in leaderboard_pnl_map:
                    result.total_pnl = leaderboard_pnl_map[wallet_lower]
                elif config.persist_to_db and self._db_session:
                    # Fallback to DB lookup if not in map
                    try:
                        db_account = await self._db_session.execute(
                            select(DiscoveredAccount).where(
                                DiscoveredAccount.wallet_address == result.wallet_address
                            )
                        )
                        account = db_account.scalar_one_or_none()
                        if account and account.total_pnl:
                            result.total_pnl = account.total_pnl
                    except Exception:
                        pass  # Continue without leaderboard P/L if DB lookup fails

            # Convert to dict format for API response
            for result in passed_results:
                results.append(result.to_dict())

            # Apply P/L range filters from config (post-processing)
            if config.max_profit is not None or hasattr(config, 'min_profit'):
                min_profit = getattr(config, 'min_profit', None)
                max_profit = config.max_profit
                filtered_results = []
                for r in results:
                    pnl = float(r.get("total_pnl") or 0)
                    if min_profit is not None and pnl < min_profit:
                        continue
                    if max_profit is not None and pnl > max_profit:
                        continue
                    filtered_results.append(r)
                results = filtered_results
                logger.info("profit_filter_applied", min=min_profit, max=max_profit, filtered_count=len(results))

            # Track errors
            for result in deep_results:
                if result.error:
                    errors.append(f"{result.wallet_address[:10]}: {result.error}")

            # Sort by composite score
            results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

            # Finalize scan
            scan.status = ScanStatus.COMPLETED
            scan.completed_at = datetime.utcnow()
            scan.duration_seconds = int((scan.completed_at - scan.started_at).total_seconds())
            scan.candidates_analyzed = len(deep_results)
            scan.candidates_passed = len(passed_results)
            scan.errors = errors if errors else None

            # Update database record
            if config.persist_to_db and self._current_scan_record:
                self._current_scan_record.status = DiscoveryScanStatus.COMPLETED
                self._current_scan_record.completed_at = datetime.utcnow()
                self._current_scan_record.passed_count = len(passed_results)
                self._current_scan_record.progress_pct = 100
                await self._commit_batch()

            # Clear checkpoint file on successful completion
            self.clear_checkpoint_file()

            await self._update_progress(
                progress, 100,
                f"Complete: {len(passed_results)} accounts found (all stored in DB)",
                "complete", progress_callback
            )

        except Exception as e:
            logger.error("scan_failed", error=str(e))
            scan.status = ScanStatus.FAILED
            scan.errors = [str(e)]

            if config.persist_to_db and self._current_scan_record:
                self._current_scan_record.status = DiscoveryScanStatus.FAILED
                self._current_scan_record.error_message = str(e)
                await self._commit_batch()

        self._current_scan = None
        return results, scan

    async def _collect_leaderboard_candidates(
        self,
        config: ScanConfig,
    ) -> list[LeaderboardEntry]:
        """Collect candidates from leaderboard API."""
        all_entries = []
        seen_wallets = set()

        # Exclude wallets to set for quick lookup
        exclude_set = set(w.lower() for w in config.exclude_wallets)

        # For small scans, don't divide by categories - fetch more from each
        # to ensure we get enough unique accounts after deduplication
        if config.max_candidates <= 100:
            # Small scan: fetch max_candidates from each category, dedupe later
            per_category_limit = config.max_candidates
        else:
            # Large scan: divide evenly but ensure minimum of 50 per category
            per_category_limit = max(50, config.max_candidates // len(config.categories))

        for category in config.categories:
            try:
                entries = await self._leaderboard_client.get_leaderboard_full(
                    category=category,
                    limit=per_category_limit,
                )

                for entry in entries:
                    if entry.wallet_address not in seen_wallets:
                        if entry.wallet_address not in exclude_set:
                            seen_wallets.add(entry.wallet_address)
                            all_entries.append(entry)

            except Exception as e:
                logger.warning("leaderboard_collection_failed", category=category, error=str(e))

        return all_entries[:config.max_candidates]

    async def _update_progress(
        self,
        progress: ScanProgress,
        pct: int,
        step: str,
        phase: str,
        callback: Optional[Callable],
    ) -> None:
        """Update scan progress and notify callback."""
        progress.progress_pct = pct
        progress.current_step = step
        progress.current_phase = phase

        if self._current_scan:
            self._current_scan.progress_pct = pct
            self._current_scan.current_step = step
            self._current_scan.current_phase = phase

        if callback:
            try:
                result = callback(progress)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("progress_callback_failed", error=str(e))

    def cancel_scan(self) -> None:
        """Cancel the current running scan."""
        self._scan_cancelled = True
        if self._current_scan:
            self._current_scan.status = ScanStatus.CANCELLED

    async def analyze_single_account(
        self,
        wallet_address: str,
        mode: DiscoveryMode = DiscoveryMode.NICHE_SPECIALIST,
        lookback_days: int = 90,
    ) -> dict:
        """Analyze a single account with full deep analysis.

        Useful for deep-diving into a specific account or
        re-analyzing with different settings.
        """
        self._analyzer.set_mode(mode)

        result = await self._analyzer.deep_analysis(
            wallet_address,
            lookback_days=lookback_days,
            include_insider_signals=True,
        )

        if result.error:
            return {"error": result.error, "wallet_address": wallet_address}

        return result.to_dict()

    def get_scan_stats(self) -> dict:
        """Get statistics from the last scan."""
        return self._analyzer.get_stats() if self._analyzer else {}

    @staticmethod
    def get_available_modes() -> list[dict]:
        """Get list of available discovery modes with configurations."""
        return ScoringEngine.get_all_modes()

    @staticmethod
    def get_leaderboard_categories() -> list[str]:
        """Get available leaderboard categories."""
        return LEADERBOARD_CATEGORIES

    @staticmethod
    def get_mode_config(mode: DiscoveryMode) -> dict:
        """Get configuration for a specific mode."""
        config = MODE_CONFIGS.get(mode)
        return config.to_dict() if config else {}
