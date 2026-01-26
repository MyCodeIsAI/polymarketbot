"""REST API routes for Account Discovery.

Provides endpoints for:
- Running discovery scans with multi-phase filtering
- Viewing and managing discovered accounts
- Configuring scoring thresholds (UI-editable)
- Deep-diving into specific accounts
- Scan history and management
"""

import asyncio
import re
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiohttp
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from pydantic import BaseModel, Field

from ..utils.logging import get_logger
from .models import DiscoveryMode, ScanStatus
from .service import DiscoveryService, ScanConfig, ScanProgress

logger = get_logger(__name__)

# Project root - works on both local dev and VPS
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

router = APIRouter(prefix="/discovery", tags=["discovery"])


# =============================================================================
# Analyzed Accounts Cache (avoid reloading 500MB+ file on each request)
# =============================================================================
_analyzed_accounts_cache: Optional[Dict[str, Any]] = None
_analyzed_accounts_mtime: float = 0


def _get_analyzed_accounts_cached() -> Dict[str, Any]:
    """Load analyzed accounts with caching to avoid repeated disk reads.

    Uses slim file (6MB) instead of full file (578MB) for low-memory VPS compatibility.
    The slim file has all filter/column data but excludes heavy fields like pl_curve_data.
    """
    global _analyzed_accounts_cache, _analyzed_accounts_mtime
    import json

    # Prefer slim file (6MB) over full file (578MB) for memory efficiency
    slim_file = DATA_DIR / "analyzed_accounts_slim.json"
    full_file = DATA_DIR / "analyzed_accounts.json"

    data_file = slim_file if slim_file.exists() else full_file
    if not data_file.exists():
        return {"accounts": [], "total_analyzed": 0}

    # Check if file has changed
    current_mtime = data_file.stat().st_mtime
    if _analyzed_accounts_cache is None or current_mtime != _analyzed_accounts_mtime:
        logger.info(f"Loading analyzed accounts from {data_file.name}...")
        with open(data_file) as f:
            _analyzed_accounts_cache = json.load(f)
        _analyzed_accounts_mtime = current_mtime
        logger.info(f"Loaded {len(_analyzed_accounts_cache.get('accounts', []))} accounts into cache")

    return _analyzed_accounts_cache


def _get_analyzed_accounts_meta() -> Dict[str, Any]:
    """Get just metadata (counts) without loading the full file."""
    import json

    meta_file = DATA_DIR / "analyzed_accounts_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)

    # Fallback: load full file (slow)
    data = _get_analyzed_accounts_cached()
    return {
        "total_analyzed": data.get("total_analyzed", len(data.get("accounts", []))),
        "generated_at": data.get("generated_at"),
    }


# =============================================================================
# Profile Views Fetcher
# =============================================================================

@dataclass
class ProfileStats:
    """Profile statistics from Polymarket."""
    wallet: str
    views: int
    username: Optional[str] = None


async def fetch_profile_views(wallet: str, session: aiohttp.ClientSession) -> Optional[ProfileStats]:
    """Fetch profile views for a single wallet."""
    url = f"https://polymarket.com/profile/{wallet.lower()}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html',
    }

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return ProfileStats(wallet=wallet.lower(), views=0)

            html = await resp.text()

            views = 0
            username = None

            views_match = re.search(r'"views":(\d+)', html)
            if views_match:
                views = int(views_match.group(1))

            name_match = re.search(r'"username":"([^"]*)"', html)
            if name_match and name_match.group(1):
                username = name_match.group(1)

            return ProfileStats(wallet=wallet.lower(), views=views, username=username)

    except Exception:
        return ProfileStats(wallet=wallet.lower(), views=0)


async def fetch_all_profile_views(
    wallets: list[str],
    max_concurrent: int = 10,
    progress_callback: Optional[callable] = None,
) -> Dict[str, ProfileStats]:
    """Fetch profile views for all wallets in parallel with rate limiting."""
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = [0]
    total = len(wallets)

    async with aiohttp.ClientSession() as session:
        async def fetch_one(wallet: str):
            async with semaphore:
                stats = await fetch_profile_views(wallet, session)
                if stats:
                    results[wallet.lower()] = stats
                completed[0] += 1
                if progress_callback and completed[0] % 50 == 0:
                    progress_callback(completed[0], total)
                await asyncio.sleep(0.1)

        await asyncio.gather(*[fetch_one(w) for w in wallets])

    return results


# =============================================================================
# In-memory state for scan management (would use DB in production)
# =============================================================================

_scan_results: dict[int, list[dict]] = {}
_scan_records: dict[int, dict] = {}
_discovered_accounts: dict[str, dict] = {}  # wallet -> account data
_scan_counter = 0
_active_scan_task: Optional[asyncio.Task] = None
_discovery_service: Optional[DiscoveryService] = None


async def get_discovery_service() -> DiscoveryService:
    """Get or create discovery service instance."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
        await _discovery_service.__aenter__()
    return _discovery_service


# =============================================================================
# Pydantic Models
# =============================================================================

class ScanConfigRequest(BaseModel):
    """Scan configuration request.

    For wide-net collection (recommended), use:
    - mode: "wide_net_profitability"
    - max_candidates: 5000-10000
    - min_score_threshold: 15-20

    Or use preset="wide_net" for automatic configuration.
    """

    mode: str = Field(default="wide_net_profitability", description="Discovery mode")
    preset: Optional[str] = Field(None, description="Use preset: 'wide_net', 'niche', or 'quick'")
    categories: Optional[List[str]] = Field(
        default=None,  # None means all categories
        description="Leaderboard categories to scan (null for all)"
    )
    market_ids: List[str] = Field(default=[], description="Market IDs for holder scans")
    reference_wallet: Optional[str] = Field(None, description="Reference wallet for similar-to mode")
    lookback_days: int = Field(90, ge=7, le=365, description="Days of history to analyze")
    min_score_threshold: float = Field(20.0, ge=0, le=100, description="Minimum score to include")
    max_candidates: int = Field(5000, ge=10, le=20000, description="Max candidates to collect (Phase 1)")
    max_phase2: int = Field(1000, ge=5, le=10000, description="Max candidates for Phase 2 light scan")
    max_phase3: int = Field(200, ge=5, le=5000, description="Max candidates for Phase 3 deep analysis")
    min_trades: int = Field(3, ge=1, description="Minimum trades required")
    max_trades: Optional[int] = Field(None, ge=1, description="Maximum trades (filter out high-frequency bots)")
    min_profit: Optional[float] = Field(None, ge=0, description="Minimum profit USD (from leaderboard)")
    max_profit: Optional[float] = Field(None, ge=0, description="Maximum profit USD (filter out whales)")
    analyze_top_n: int = Field(500, ge=10, le=5000, description="Number of trades to scrape per account")
    max_avg_position_size: float = Field(100000.0, ge=1, description="Max average position size USD")
    persist_to_db: bool = Field(True, description="Store results in database")
    track_profile_views: bool = Field(True, description="Fetch profile view counts after scan")
    # Alias for frontend compatibility
    trades_limit: Optional[int] = Field(None, description="Alias for analyze_top_n (deprecated)")


class ScanResponse(BaseModel):
    """Scan status response with phase details."""

    scan_id: int
    status: str
    progress_pct: int
    current_step: str
    current_phase: str = "init"
    candidates_found: int
    candidates_analyzed: int
    candidates_passed: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Phase breakdown
    phase1_total: int = 0
    phase1_passed: int = 0
    phase2_total: int = 0
    phase2_passed: int = 0
    phase3_total: int = 0
    phase3_passed: int = 0
    api_calls_made: int = 0


class FilterConfigUpdate(BaseModel):
    """Update for a single filter configuration."""
    threshold: Optional[float] = None
    enabled: Optional[bool] = None
    weight: Optional[float] = None
    pass_bonus: Optional[float] = None
    fail_penalty: Optional[float] = None


class ScoringConfigUpdate(BaseModel):
    """Update request for scoring configuration."""
    hard_filters: Optional[Dict[str, FilterConfigUpdate]] = None
    soft_filters: Optional[Dict[str, FilterConfigUpdate]] = None
    pl_consistency_weight: Optional[float] = None
    pattern_match_weight: Optional[float] = None
    specialization_weight: Optional[float] = None
    risk_weight: Optional[float] = None
    min_composite_score: Optional[float] = None


class DiscoveredAccountResponse(BaseModel):
    """Discovered account response."""

    wallet_address: str
    composite_score: float
    score_breakdown: dict
    red_flag_count: int
    total_trades: Optional[int] = None
    total_pnl: Optional[str] = None
    avg_position_size: Optional[str] = None
    win_rate: Optional[float] = None
    market_categories: dict
    is_favorite: bool = False
    is_hidden: bool = False


class AccountDetailResponse(BaseModel):
    """Detailed account analysis response."""

    wallet_address: str
    composite_score: float
    score_breakdown: dict
    passes_threshold: bool
    red_flags: list
    pl_metrics: Optional[dict] = None
    pattern_metrics: Optional[dict] = None
    insider_signals: Optional[dict] = None
    market_categories: dict
    recent_trades_sample: list
    pl_curve_data: list


class DiscoveryModesResponse(BaseModel):
    """Available discovery modes response."""

    modes: list


# =============================================================================
# Configuration Endpoints (NEW - for UI threshold editing)
# =============================================================================

@router.get("/config/{mode}")
async def get_mode_config(mode: str):
    """Get scoring configuration for a discovery mode.

    Returns all hard filters, soft filters, and weights that can be
    edited in the UI.
    """
    try:
        discovery_mode = DiscoveryMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    service = await get_discovery_service()
    config = service.get_scoring_config(discovery_mode)

    return {
        "mode": mode,
        "config": config,
    }


@router.put("/config/{mode}")
async def update_mode_config(mode: str, updates: ScoringConfigUpdate):
    """Update scoring configuration for a discovery mode.

    Allows UI to adjust thresholds, weights, and filter settings.
    """
    try:
        discovery_mode = DiscoveryMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    service = await get_discovery_service()

    # Convert Pydantic model to dict, filtering out None values
    update_dict = {}

    if updates.hard_filters:
        update_dict["hard_filters"] = {
            k: {kk: vv for kk, vv in v.model_dump().items() if vv is not None}
            for k, v in updates.hard_filters.items()
        }

    if updates.soft_filters:
        update_dict["soft_filters"] = {
            k: {kk: vv for kk, vv in v.model_dump().items() if vv is not None}
            for k, v in updates.soft_filters.items()
        }

    # Add weight updates
    for key in ["pl_consistency_weight", "pattern_match_weight",
                "specialization_weight", "risk_weight", "min_composite_score"]:
        value = getattr(updates, key, None)
        if value is not None:
            update_dict[key] = value

    # Apply updates
    service._analyzer.set_mode(discovery_mode)
    updated_config = service.update_scoring_config(update_dict)

    return {
        "mode": mode,
        "config": updated_config,
        "message": "Configuration updated",
    }


@router.post("/config/{mode}/reset")
async def reset_mode_config(mode: str):
    """Reset scoring configuration to defaults for a mode."""
    try:
        discovery_mode = DiscoveryMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    # Get default config by re-importing
    from .scoring import (
        MODE_CONFIGS, create_micro_bet_hunter_config, create_niche_specialist_config,
        create_insider_detection_config, create_similar_to_config, create_wide_net_profitability_config
    )

    # Recreate default config
    if discovery_mode == DiscoveryMode.WIDE_NET_PROFITABILITY:
        MODE_CONFIGS[discovery_mode] = create_wide_net_profitability_config()
    elif discovery_mode == DiscoveryMode.MICRO_BET_HUNTER:
        MODE_CONFIGS[discovery_mode] = create_micro_bet_hunter_config()
    elif discovery_mode == DiscoveryMode.NICHE_SPECIALIST:
        MODE_CONFIGS[discovery_mode] = create_niche_specialist_config()
    elif discovery_mode == DiscoveryMode.INSIDER_DETECTION:
        MODE_CONFIGS[discovery_mode] = create_insider_detection_config()
    elif discovery_mode == DiscoveryMode.SIMILAR_TO:
        MODE_CONFIGS[discovery_mode] = create_similar_to_config()

    service = await get_discovery_service()
    service._analyzer.set_mode(discovery_mode)
    config = service.get_scoring_config(discovery_mode)

    return {
        "mode": mode,
        "config": config,
        "message": "Configuration reset to defaults",
    }


# =============================================================================
# Scan Endpoints
# =============================================================================

@router.post("/scan/start", response_model=ScanResponse)
async def start_scan(config: ScanConfigRequest, background_tasks: BackgroundTasks):
    """Start a new multi-phase discovery scan."""
    global _scan_counter, _active_scan_task

    # Check if scan is already running
    if _active_scan_task and not _active_scan_task.done():
        raise HTTPException(status_code=409, detail="A scan is already in progress")

    # Validate mode
    try:
        mode = DiscoveryMode(config.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {config.mode}")

    # Create scan config (use preset if specified, otherwise manual config)
    if config.preset:
        preset_map = {
            "wide_net": ScanConfig.wide_net_preset,
            "niche": ScanConfig.niche_focused_preset,
            "quick": ScanConfig.quick_scan_preset,
        }
        preset_func = preset_map.get(config.preset.lower())
        if not preset_func:
            raise HTTPException(status_code=400, detail=f"Invalid preset: {config.preset}")
        scan_config = preset_func()
        # Override with any explicit values
        if config.categories is not None:
            scan_config.categories = config.categories
        if config.market_ids:
            scan_config.market_ids = config.market_ids
        if config.reference_wallet:
            scan_config.reference_wallet = config.reference_wallet
        scan_config.lookback_days = config.lookback_days
        scan_config.persist_to_db = config.persist_to_db
    else:
        # Handle trades_limit alias for analyze_top_n
        analyze_top_n = config.analyze_top_n
        if config.trades_limit is not None:
            analyze_top_n = config.trades_limit

        scan_config = ScanConfig(
            mode=mode,
            categories=config.categories,
            market_ids=config.market_ids,
            reference_wallet=config.reference_wallet,
            lookback_days=config.lookback_days,
            min_score_threshold=config.min_score_threshold,
            max_candidates=config.max_candidates,
            max_phase2=config.max_phase2,
            max_phase3=config.max_phase3,
            min_trades=config.min_trades,
            max_trades=config.max_trades,
            min_profit=config.min_profit,
            max_profit=config.max_profit,
            analyze_top_n=analyze_top_n,
            max_avg_position_size=config.max_avg_position_size,
            persist_to_db=config.persist_to_db,
        )

    # Create scan record
    _scan_counter += 1
    scan_id = _scan_counter

    _scan_records[scan_id] = {
        "scan_id": scan_id,
        "status": ScanStatus.RUNNING.value,
        "progress_pct": 0,
        "current_step": "Initializing...",
        "current_phase": "init",
        "candidates_found": 0,
        "candidates_analyzed": 0,
        "candidates_passed": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "config": config.model_dump(),
        # Phase stats
        "phase1_total": 0,
        "phase1_passed": 0,
        "phase2_total": 0,
        "phase2_passed": 0,
        "phase3_total": 0,
        "phase3_passed": 0,
        "api_calls_made": 0,
    }

    # Start scan in background
    async def run_scan():
        try:
            service = await get_discovery_service()

            def progress_callback(progress: ScanProgress):
                _scan_records[scan_id].update({
                    "status": progress.status.value,
                    "progress_pct": progress.progress_pct,
                    "current_step": progress.current_step,
                    "current_phase": progress.current_phase,
                    "candidates_found": progress.candidates_found,
                    "candidates_analyzed": progress.candidates_analyzed,
                    "candidates_passed": progress.candidates_passed,
                    "phase1_total": progress.phase1_total,
                    "phase1_passed": progress.phase1_passed,
                    "phase2_total": progress.phase2_total,
                    "phase2_passed": progress.phase2_passed,
                    "phase3_total": progress.phase3_total,
                    "phase3_passed": progress.phase3_passed,
                    "api_calls_made": progress.api_calls_made,
                })

            results, scan = await service.run_scan(scan_config, progress_callback)

            # Fetch profile views if enabled
            if config.track_profile_views and results:
                _scan_records[scan_id].update({
                    "current_step": f"Fetching profile views for {len(results)} accounts...",
                    "current_phase": "phase4_views",
                })

                def views_progress(completed: int, total: int):
                    _scan_records[scan_id]["current_step"] = f"Fetching views: {completed}/{total}..."

                wallets = [r.get("wallet_address", "") for r in results if r.get("wallet_address")]
                profile_views = await fetch_all_profile_views(wallets, max_concurrent=10, progress_callback=views_progress)

                # Attach views to results
                for result in results:
                    wallet_lower = result.get("wallet_address", "").lower()
                    if wallet_lower in profile_views:
                        result["profile_views"] = profile_views[wallet_lower].views
                        result["profile_username"] = profile_views[wallet_lower].username
                    else:
                        result["profile_views"] = 0
                        result["profile_username"] = None

                logger.info("profile_views_fetched", count=len(profile_views))

            # Store results
            _scan_results[scan_id] = results
            _scan_records[scan_id].update({
                "status": scan.status.value,
                "progress_pct": 100,
                "current_step": f"Complete: {len(results)} accounts found",
                "current_phase": "complete",
                "completed_at": datetime.utcnow().isoformat(),
                "candidates_found": scan.candidates_found,
                "candidates_analyzed": scan.candidates_analyzed,
                "candidates_passed": scan.candidates_passed,
            })

            # Add to discovered accounts
            for account in results:
                wallet = account["wallet_address"]
                _discovered_accounts[wallet] = {
                    **account,
                    "is_favorite": False,
                    "is_hidden": False,
                    "discovered_at": datetime.utcnow().isoformat(),
                }

            logger.info("scan_completed", scan_id=scan_id, passed=len(results))

        except Exception as e:
            logger.error("scan_failed", scan_id=scan_id, error=str(e))
            _scan_records[scan_id].update({
                "status": ScanStatus.FAILED.value,
                "current_step": f"Error: {str(e)}",
            })

    _active_scan_task = asyncio.create_task(run_scan())

    return ScanResponse(
        scan_id=scan_id,
        status=ScanStatus.RUNNING.value,
        progress_pct=0,
        current_step="Initializing...",
        current_phase="init",
        candidates_found=0,
        candidates_analyzed=0,
        candidates_passed=0,
        started_at=_scan_records[scan_id]["started_at"],
    )


@router.get("/scan/{scan_id}", response_model=ScanResponse)
async def get_scan_status(scan_id: int):
    """Get status of a discovery scan with phase details."""
    if scan_id not in _scan_records:
        raise HTTPException(status_code=404, detail="Scan not found")

    record = _scan_records[scan_id]
    return ScanResponse(
        scan_id=scan_id,
        status=record["status"],
        progress_pct=record["progress_pct"],
        current_step=record["current_step"],
        current_phase=record.get("current_phase", ""),
        candidates_found=record["candidates_found"],
        candidates_analyzed=record["candidates_analyzed"],
        candidates_passed=record["candidates_passed"],
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        phase1_total=record.get("phase1_total", 0),
        phase1_passed=record.get("phase1_passed", 0),
        phase2_total=record.get("phase2_total", 0),
        phase2_passed=record.get("phase2_passed", 0),
        phase3_total=record.get("phase3_total", 0),
        phase3_passed=record.get("phase3_passed", 0),
        api_calls_made=record.get("api_calls_made", 0),
    )


@router.post("/scan/{scan_id}/cancel")
async def cancel_scan(scan_id: int):
    """Cancel a running scan."""
    if scan_id not in _scan_records:
        raise HTTPException(status_code=404, detail="Scan not found")

    if _scan_records[scan_id]["status"] != ScanStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Scan is not running")

    service = await get_discovery_service()
    service.cancel_scan()

    _scan_records[scan_id]["status"] = ScanStatus.CANCELLED.value
    _scan_records[scan_id]["current_step"] = "Cancelled by user"

    return {"message": "Scan cancelled"}


@router.get("/checkpoint")
async def get_checkpoint():
    """Get current checkpoint data for resume capability.

    Returns checkpoint data if a scan was interrupted, or null if no checkpoint exists.
    Use this to offer resume functionality after a crash or browser close.
    """
    checkpoint = DiscoveryService.load_checkpoint_from_file()

    if not checkpoint:
        return {"checkpoint": None, "message": "No checkpoint found"}

    return {
        "checkpoint": checkpoint,
        "can_resume": True,
        "processed_count": checkpoint.get("processed_count", 0),
        "phase": checkpoint.get("phase"),
        "timestamp": checkpoint.get("timestamp"),
    }


@router.post("/scan/resume")
async def resume_scan(config: ScanConfigRequest, background_tasks: BackgroundTasks):
    """Resume a scan from checkpoint.

    Loads checkpoint data and starts a new scan that skips already-processed wallets.
    The exclude_wallets in the scan config will be populated from the checkpoint.
    """
    global _scan_counter, _active_scan_task

    # Check if scan is already running
    if _active_scan_task and not _active_scan_task.done():
        raise HTTPException(status_code=409, detail="A scan is already in progress")

    # Load checkpoint
    checkpoint = DiscoveryService.load_checkpoint_from_file()
    if not checkpoint:
        raise HTTPException(status_code=404, detail="No checkpoint found to resume from")

    processed_wallets = checkpoint.get("processed_wallets", [])

    # Validate mode
    try:
        mode = DiscoveryMode(config.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {config.mode}")

    # Create scan config with exclude_wallets from checkpoint
    scan_config = ScanConfig(
        mode=mode,
        categories=config.categories,
        market_ids=config.market_ids,
        reference_wallet=config.reference_wallet,
        lookback_days=config.lookback_days,
        min_score_threshold=config.min_score_threshold,
        max_candidates=config.max_candidates,
        max_phase2=config.max_phase2,
        max_phase3=config.max_phase3,
        min_trades=config.min_trades,
        max_trades=config.max_trades,
        max_profit=config.max_profit,
        max_avg_position_size=config.max_avg_position_size,
        persist_to_db=config.persist_to_db,
        exclude_wallets=processed_wallets,  # Skip already-processed wallets
    )

    # Create scan record
    _scan_counter += 1
    scan_id = _scan_counter

    _scan_records[scan_id] = {
        "scan_id": scan_id,
        "status": ScanStatus.RUNNING.value,
        "progress_pct": 0,
        "current_step": f"Resuming from checkpoint ({len(processed_wallets)} already processed)...",
        "current_phase": "init",
        "candidates_found": 0,
        "candidates_analyzed": 0,
        "candidates_passed": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "config": config.model_dump(),
        "resumed_from_checkpoint": True,
        "skipped_wallets": len(processed_wallets),
        # Phase stats
        "phase1_total": 0,
        "phase1_passed": 0,
        "phase2_total": 0,
        "phase2_passed": 0,
        "phase3_total": 0,
        "phase3_passed": 0,
        "api_calls_made": 0,
    }

    # Start scan in background (same as start_scan)
    async def run_scan():
        try:
            service = await get_discovery_service()

            def progress_callback(progress: ScanProgress):
                _scan_records[scan_id].update({
                    "status": progress.status.value,
                    "progress_pct": progress.progress_pct,
                    "current_step": progress.current_step,
                    "current_phase": progress.current_phase,
                    "candidates_found": progress.candidates_found,
                    "candidates_analyzed": progress.candidates_analyzed,
                    "candidates_passed": progress.candidates_passed,
                    "phase1_total": progress.phase1_total,
                    "phase1_passed": progress.phase1_passed,
                    "phase2_total": progress.phase2_total,
                    "phase2_passed": progress.phase2_passed,
                    "phase3_total": progress.phase3_total,
                    "phase3_passed": progress.phase3_passed,
                    "api_calls_made": progress.api_calls_made,
                })

            results, scan = await service.run_scan(scan_config, progress_callback)

            # Store results
            _scan_results[scan_id] = results
            _scan_records[scan_id].update({
                "status": scan.status.value,
                "progress_pct": 100,
                "current_step": f"Complete: {len(results)} accounts found",
                "current_phase": "complete",
                "completed_at": datetime.utcnow().isoformat(),
                "candidates_found": scan.candidates_found,
                "candidates_analyzed": scan.candidates_analyzed,
                "candidates_passed": scan.candidates_passed,
            })

            # Add to discovered accounts
            for account in results:
                wallet = account["wallet_address"]
                _discovered_accounts[wallet] = {
                    **account,
                    "is_favorite": False,
                    "is_hidden": False,
                    "discovered_at": datetime.utcnow().isoformat(),
                }

            logger.info("resumed_scan_completed", scan_id=scan_id, passed=len(results),
                       skipped=len(processed_wallets))

        except Exception as e:
            logger.error("resumed_scan_failed", scan_id=scan_id, error=str(e))
            _scan_records[scan_id].update({
                "status": ScanStatus.FAILED.value,
                "current_step": f"Error: {str(e)}",
            })

    _active_scan_task = asyncio.create_task(run_scan())

    return ScanResponse(
        scan_id=scan_id,
        status=ScanStatus.RUNNING.value,
        progress_pct=0,
        current_step=f"Resuming from checkpoint ({len(processed_wallets)} wallets skipped)...",
        current_phase="init",
        candidates_found=0,
        candidates_analyzed=0,
        candidates_passed=0,
        started_at=_scan_records[scan_id]["started_at"],
    )


@router.delete("/checkpoint")
async def clear_checkpoint():
    """Clear the checkpoint file.

    Use this when you want to start fresh instead of resuming.
    """
    DiscoveryService.clear_checkpoint_file()
    return {"message": "Checkpoint cleared"}


@router.get("/scan/{scan_id}/results")
async def get_scan_results(
    scan_id: int,
    min_score: float = Query(0, ge=0, le=100),
    limit: int = Query(50, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    hide_flagged: bool = Query(False),
    sort_by: str = Query("composite_score"),
):
    """Get results from a completed scan."""
    if scan_id not in _scan_results:
        if scan_id in _scan_records:
            status = _scan_records[scan_id]["status"]
            if status == ScanStatus.RUNNING.value:
                raise HTTPException(status_code=400, detail="Scan still running")
            elif status == ScanStatus.FAILED.value:
                raise HTTPException(status_code=400, detail="Scan failed")
        raise HTTPException(status_code=404, detail="Scan results not found")

    results = _scan_results[scan_id]

    # Filter
    filtered = [
        r for r in results
        if r["composite_score"] >= min_score
        and (not hide_flagged or r.get("red_flag_count", 0) == 0)
    ]

    # Sort
    if sort_by == "composite_score":
        filtered.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    elif sort_by == "total_pnl":
        # Prefer authoritative total_pnl (from leaderboard) over computed pl_metrics value
        filtered.sort(
            key=lambda x: float(x.get("total_pnl") or x.get("pl_metrics", {}).get("total_realized_pnl", "0") or "0"),
            reverse=True
        )
    elif sort_by == "red_flag_count":
        filtered.sort(key=lambda x: x.get("red_flag_count", 0))

    # Paginate
    total = len(filtered)
    paginated = filtered[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": paginated,
    }


@router.get("/scans")
async def list_scans(limit: int = Query(20, ge=1, le=100)):
    """List recent scans with phase details."""
    scans = list(_scan_records.values())
    scans.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return {"scans": scans[:limit]}


# =============================================================================
# Account Endpoints
# =============================================================================

@router.get("/accounts")
async def list_discovered_accounts(
    min_score: float = Query(0, ge=0, le=100),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    favorites_only: bool = Query(False),
    hide_hidden: bool = Query(True),
    sort_by: str = Query("composite_score"),
):
    """List all discovered accounts."""
    accounts = list(_discovered_accounts.values())

    # Filter
    if favorites_only:
        accounts = [a for a in accounts if a.get("is_favorite")]
    if hide_hidden:
        accounts = [a for a in accounts if not a.get("is_hidden")]

    accounts = [a for a in accounts if a.get("composite_score", 0) >= min_score]

    # Sort
    reverse = True
    if sort_by == "composite_score":
        accounts.sort(key=lambda x: x.get("composite_score", 0), reverse=reverse)
    elif sort_by == "total_pnl":
        # Prefer authoritative total_pnl (from leaderboard) over computed pl_metrics value
        accounts.sort(key=lambda x: float(x.get("total_pnl") or x.get("pl_metrics", {}).get("total_realized_pnl", "0") or "0"), reverse=reverse)
    elif sort_by == "total_trades":
        accounts.sort(key=lambda x: x.get("pattern_metrics", {}).get("total_trades", 0) or 0, reverse=reverse)
    elif sort_by == "red_flag_count":
        accounts.sort(key=lambda x: x.get("red_flag_count", 0))

    # Paginate
    total = len(accounts)
    paginated = accounts[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "accounts": paginated,
    }


@router.get("/accounts/{wallet_address}")
async def get_account_detail(wallet_address: str):
    """Get detailed analysis for a specific account."""
    wallet = wallet_address.lower()

    # Check if already analyzed
    if wallet in _discovered_accounts:
        return _discovered_accounts[wallet]

    # Analyze on demand
    try:
        service = await get_discovery_service()
        result = await service.analyze_single_account(wallet)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Cache result
        _discovered_accounts[wallet] = {
            **result,
            "is_favorite": False,
            "is_hidden": False,
            "discovered_at": datetime.utcnow().isoformat(),
        }

        return _discovered_accounts[wallet]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("account_analysis_failed", wallet=wallet[:10], error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/accounts/{wallet_address}/analyze")
async def reanalyze_account(
    wallet_address: str,
    mode: str = Query("wide_net_profitability"),
    lookback_days: int = Query(1825, ge=7, le=1825),
):
    """Reanalyze a specific account with full trade history.

    Default lookback is 1825 days (5 years) to capture complete trade history
    for accurate P/L and win rate metrics.
    """
    wallet = wallet_address.lower()

    try:
        discovery_mode = DiscoveryMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    try:
        service = await get_discovery_service()
        result = await service.analyze_single_account(
            wallet,
            mode=discovery_mode,
            lookback_days=lookback_days,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Update cache
        if wallet in _discovered_accounts:
            _discovered_accounts[wallet].update(result)
        else:
            _discovered_accounts[wallet] = {
                **result,
                "is_favorite": False,
                "is_hidden": False,
                "discovered_at": datetime.utcnow().isoformat(),
            }

        return _discovered_accounts[wallet]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("account_reanalysis_failed", wallet=wallet[:10], error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/accounts/{wallet_address}")
async def update_account(
    wallet_address: str,
    is_favorite: Optional[bool] = None,
    is_hidden: Optional[bool] = None,
    notes: Optional[str] = None,
):
    """Update account metadata (favorite, hidden, notes)."""
    wallet = wallet_address.lower()

    if wallet not in _discovered_accounts:
        raise HTTPException(status_code=404, detail="Account not found")

    if is_favorite is not None:
        _discovered_accounts[wallet]["is_favorite"] = is_favorite
    if is_hidden is not None:
        _discovered_accounts[wallet]["is_hidden"] = is_hidden
    if notes is not None:
        _discovered_accounts[wallet]["notes"] = notes

    return _discovered_accounts[wallet]


@router.delete("/accounts/{wallet_address}")
async def remove_account(wallet_address: str):
    """Remove an account from discovered list."""
    wallet = wallet_address.lower()

    if wallet not in _discovered_accounts:
        raise HTTPException(status_code=404, detail="Account not found")

    del _discovered_accounts[wallet]
    return {"message": "Account removed"}


# =============================================================================
# Utility Endpoints
# =============================================================================

@router.get("/modes", response_model=DiscoveryModesResponse)
async def get_discovery_modes():
    """Get available discovery modes with their configurations."""
    modes = DiscoveryService.get_available_modes()
    return DiscoveryModesResponse(modes=modes)


@router.get("/categories")
async def get_leaderboard_categories():
    """Get available leaderboard categories."""
    return {"categories": DiscoveryService.get_leaderboard_categories()}


@router.get("/stats")
async def get_discovery_stats():
    """Get discovery statistics including scan efficiency."""
    total_accounts = len(_discovered_accounts)
    favorites = len([a for a in _discovered_accounts.values() if a.get("is_favorite")])
    hidden = len([a for a in _discovered_accounts.values() if a.get("is_hidden")])

    # Score distribution
    scores = [a.get("composite_score", 0) for a in _discovered_accounts.values()]
    score_ranges = {
        "90+": len([s for s in scores if s >= 90]),
        "80-89": len([s for s in scores if 80 <= s < 90]),
        "70-79": len([s for s in scores if 70 <= s < 80]),
        "60-69": len([s for s in scores if 60 <= s < 70]),
        "50-59": len([s for s in scores if 50 <= s < 60]),
        "<50": len([s for s in scores if s < 50]),
    }

    # Scan efficiency stats
    service = await get_discovery_service()
    scan_stats = service.get_scan_stats()

    return {
        "total_accounts": total_accounts,
        "favorites": favorites,
        "hidden": hidden,
        "total_scans": len(_scan_records),
        "score_distribution": score_ranges,
        "scan_efficiency": scan_stats,
    }


@router.get("/scan-stats")
async def get_scan_efficiency_stats():
    """Get detailed scan efficiency statistics."""
    service = await get_discovery_service()
    return service.get_scan_stats()


@router.get("/analyzed-accounts/meta")
async def get_analyzed_accounts_meta():
    """Get just metadata (counts) for analyzed accounts - fast, no heavy file load."""
    meta = _get_analyzed_accounts_meta()
    return {
        "total_analyzed": meta.get("total_analyzed", 0),
        "generated_at": meta.get("generated_at"),
        "active_count": meta.get("active_count", 0),
        "quality_count": meta.get("quality_count", 0),
    }


@router.get("/analyzed-accounts")
async def get_analyzed_accounts(
    min_score: float = Query(0, ge=0, le=100),
    max_score: float = Query(100, ge=0, le=100),
    min_pnl: float = Query(0),
    min_trades: int = Query(0, ge=0),
    min_markets: int = Query(0, ge=0),
    max_recency_days: int = Query(9999, ge=0),
    max_profile_views: int = Query(None, ge=0, description="Exclude accounts with more than N profile views"),
    min_profile_views: int = Query(0, ge=0, description="Only include accounts with at least N profile views"),
    hide_dismissed: bool = Query(True),
    hide_watchlisted: bool = Query(False),
    limit: int = Query(50, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("systematic_score"),
    sort_order: str = Query("desc"),
):
    """Get pre-analyzed systematic trader accounts with flexible filtering.

    Filter parameters:
    - min_score/max_score: Score range filter
    - min_pnl: Minimum total PnL
    - min_trades: Minimum trade count
    - min_markets: Minimum unique markets traded
    - max_recency_days: Only accounts active within N days
    - max_profile_views: Exclude accounts with too many profile views (find hidden gems)
    - min_profile_views: Only include accounts with minimum profile visibility
    """
    data_file = DATA_DIR / "analyzed_accounts.json"

    if not data_file.exists():
        return {
            "error": "Analyzed accounts data not found. Run analyze_profitable.py first.",
            "total": 0,
            "accounts": [],
        }

    # Use cached data to avoid reloading 500MB+ file on each request
    data = _get_analyzed_accounts_cached()
    accounts = data.get("accounts", [])

    # Ensure wallet_address is present (some older data only has "wallet")
    for a in accounts:
        if "wallet_address" not in a and "wallet" in a:
            a["wallet_address"] = a["wallet"]

    # Apply filters
    filtered = []
    for a in accounts:
        wallet = a.get("wallet_address", a.get("wallet", "")).lower()

        # Filter dismissed accounts
        if hide_dismissed and wallet in _dismissed:
            continue

        # Filter watchlisted accounts (optional)
        if hide_watchlisted and wallet in _watchlist:
            continue

        score = a.get("systematic_score", 0)
        if score < min_score or score > max_score:
            continue
        if a.get("total_pnl", 0) < min_pnl:
            continue
        if a.get("num_trades", 0) < min_trades:
            continue
        if a.get("unique_markets", 0) < min_markets:
            continue
        if a.get("activity_recency_days", 9999) > max_recency_days:
            continue

        # Profile views filter
        views = a.get("profile_views", 0)
        if max_profile_views is not None and views > max_profile_views:
            continue
        if views < min_profile_views:
            continue

        # Add status flags
        a["is_watchlisted"] = wallet in _watchlist
        a["is_dismissed"] = wallet in _dismissed

        filtered.append(a)

    # Sort
    reverse = sort_order == "desc"
    sort_keys = {
        "systematic_score": lambda x: x.get("systematic_score", 0),
        "total_pnl": lambda x: x.get("total_pnl", 0),
        "num_trades": lambda x: x.get("num_trades", 0),
        "pnl_per_trade": lambda x: x.get("pnl_per_trade", 0),
        "unique_markets": lambda x: x.get("unique_markets", 0),
        "account_age_days": lambda x: x.get("account_age_days", 0),
        "activity_recency_days": lambda x: -x.get("activity_recency_days", 9999),  # Lower is better
        "trades_last_7d": lambda x: x.get("trades_last_7d", 0),
        "trades_last_30d": lambda x: x.get("trades_last_30d", 0),
        "profile_views": lambda x: x.get("profile_views", 0),  # Sort by popularity
    }
    filtered.sort(key=sort_keys.get(sort_by, sort_keys["systematic_score"]), reverse=reverse)

    # Paginate
    total = len(filtered)
    paginated = filtered[offset:offset + limit]

    return {
        "generated_at": data.get("generated_at"),
        "total_collected": data.get("total_collected", 0),
        "total_analyzed": data.get("total_analyzed", 0),
        "pnl_distribution": data.get("pnl_distribution", {}),
        "views_distribution": data.get("views_distribution", {}),
        "total": total,
        "total_dismissed": len(_dismissed),
        "total_watchlisted": len(_watchlist),
        "offset": offset,
        "limit": limit,
        "accounts": paginated,
    }


@router.get("/insider-suspects")
async def get_insider_suspects(
    min_score: float = Query(0, ge=0, le=100),
    priority: str = Query(None, description="Filter by priority: critical, high, medium, low"),
    sort_by: str = Query("insider_score", description="Sort field"),
    reverse: bool = Query(True, description="Descending order"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=10000),
):
    """
    Get accounts flagged as insider suspects from the insider probe results.

    These are accounts that match patterns documented in known insider trading cases.
    """
    import json
    from pathlib import Path

    data_file = DATA_DIR / "insider_probe_results.json"

    if not data_file.exists():
        return {
            "error": "Insider probe results not found. Run scripts/insider_probe.py first.",
            "total": 0,
            "accounts": [],
        }

    with open(data_file) as f:
        data = json.load(f)

    accounts = data.get("accounts", [])

    # Apply filters
    filtered = []
    for acc in accounts:
        # Min score filter
        score = acc.get("insider_score", 0)
        if score < min_score:
            continue

        # Priority filter
        if priority:
            acc_priority = acc.get("priority", "").lower()
            if acc_priority != priority.lower():
                continue

        # Skip dismissed accounts
        wallet = acc.get("wallet_address", "").lower()
        if wallet in _dismissed:
            continue

        filtered.append(acc)

    # Sort
    sort_keys = {
        "insider_score": lambda x: x.get("insider_score", 0),
        "total_pnl": lambda x: x.get("total_pnl", 0),
        "win_rate": lambda x: x.get("win_rate", 0),
        "account_age_days": lambda x: x.get("account_age_days", 0),
        "num_trades": lambda x: x.get("total_trades", 0),  # Data uses total_trades
        "total_trades": lambda x: x.get("total_trades", 0),
    }
    filtered.sort(key=sort_keys.get(sort_by, sort_keys["insider_score"]), reverse=reverse)

    # Paginate
    total = len(filtered)
    paginated = filtered[offset:offset + limit]

    return {
        "generated_at": data.get("generated_at"),
        "config": data.get("config", {}),
        "stats": data.get("stats", {}),
        "total": total,
        "offset": offset,
        "limit": limit,
        "accounts": paginated,
    }


@router.post("/enrich-profile-views")
async def enrich_profile_views(
    wallets: list[str] = Body(..., description="List of wallet addresses to enrich"),
):
    """
    Fetch profile views for a list of wallets.

    Returns a dict mapping wallet -> {views, username}
    """
    if not wallets:
        return {"results": {}}

    # Limit to prevent abuse
    if len(wallets) > 500:
        wallets = wallets[:500]

    results = await fetch_all_profile_views(wallets, max_concurrent=15)

    return {
        "results": {
            wallet: {"views": stats.views, "username": stats.username}
            for wallet, stats in results.items()
        }
    }



# =============================================================================
# Preset Scan Runners
# =============================================================================

# Track running preset scans
_preset_scans: Dict[str, Dict] = {}


class PresetScanStatus(BaseModel):
    """Status of a preset scan."""
    scan_type: str
    status: str  # pending, running, completed, error
    progress_pct: float = 0
    message: str = ""
    accounts_found: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PresetScanRequest(BaseModel):
    """Request to run a preset scan."""
    use_cache: bool = False  # True = load from cache, False = run new scan
    import_to_scanner: bool = True  # Auto-import results to scanner watchlist


def _get_cache_info(result_file: str) -> Optional[Dict]:
    """Get information about cached scan results."""
    from pathlib import Path
    import json
    import os

    path = Path(result_file)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        stat = os.stat(path)
        return {
            "exists": True,
            "generated_at": data.get("generated_at"),
            "accounts_count": len(data.get("accounts", [])) or data.get("stats", {}).get("total_flagged", 0),
            "file_size_kb": stat.st_size // 1024,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except Exception:
        return None


@router.get("/preset/cache-info")
async def get_preset_cache_info():
    """Get information about cached preset scan results.

    Returns the last run time and account count for each preset,
    helping users decide whether to use cache or run a new scan.
    """
    return {
        "mass_scan": _get_cache_info(str(DATA_DIR / "analyzed_accounts.json")),
        "insider_probe": _get_cache_info(str(DATA_DIR / "insider_probe_results.json")),
    }


@router.post("/preset/mass-scan")
async def run_mass_scan_preset(
    background_tasks: BackgroundTasks,
    request: PresetScanRequest = PresetScanRequest()
):
    """Run the mass scan preset to find profitable traders.

    Options:
    - use_cache=true: Load from existing cached results (fast)
    - use_cache=false: Run a new scan (slow, ~5-10 min)

    This executes the scripts/mass_scan.py logic in the background.
    """
    import json
    from pathlib import Path

    result_file = DATA_DIR / "analyzed_accounts.json"
    scan_id = f"mass_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle cache mode
    if request.use_cache:
        if not result_file.exists():
            raise HTTPException(status_code=404, detail="No cached mass scan results found. Run a new scan first.")

        try:
            with open(result_file) as f:
                data = json.load(f)
            accounts_found = len(data.get("accounts", []))

            _preset_scans[scan_id] = {
                "scan_type": "mass_scan",
                "status": "completed",
                "progress_pct": 100,
                "message": f"Loaded {accounts_found} accounts from cache",
                "accounts_found": accounts_found,
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "error": None,
                "from_cache": True,
                "cache_date": data.get("generated_at"),
            }

            return {"scan_id": scan_id, "message": f"Loaded {accounts_found} accounts from cache", "from_cache": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load cache: {str(e)}")

    # Check if already running
    if any(s["status"] == "running" and s["scan_type"] == "mass_scan" for s in _preset_scans.values()):
        raise HTTPException(status_code=409, detail="Mass scan already running")

    _preset_scans[scan_id] = {
        "scan_type": "mass_scan",
        "status": "running",
        "progress_pct": 0,
        "message": "Starting mass scan...",
        "accounts_found": 0,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "from_cache": False,
    }

    async def run_scan():
        try:
            script_path = SCRIPTS_DIR / "mass_scan.py"
            if not script_path.exists():
                _preset_scans[scan_id]["status"] = "error"
                _preset_scans[scan_id]["error"] = "mass_scan.py not found"
                return

            _preset_scans[scan_id]["message"] = "Running mass scan..."
            _preset_scans[scan_id]["progress_pct"] = 10

            # Run the script
            process = await asyncio.create_subprocess_exec(
                "python3", str(script_path),
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Load results
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    _preset_scans[scan_id]["accounts_found"] = len(data.get("accounts", []))

                _preset_scans[scan_id]["status"] = "completed"
                _preset_scans[scan_id]["progress_pct"] = 100
                _preset_scans[scan_id]["message"] = f"Complete! Found {_preset_scans[scan_id]['accounts_found']} accounts"
            else:
                _preset_scans[scan_id]["status"] = "error"
                _preset_scans[scan_id]["error"] = stderr.decode()[:500]
                _preset_scans[scan_id]["message"] = "Scan failed"

            _preset_scans[scan_id]["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            _preset_scans[scan_id]["status"] = "error"
            _preset_scans[scan_id]["error"] = str(e)
            _preset_scans[scan_id]["completed_at"] = datetime.now().isoformat()

    background_tasks.add_task(run_scan)

    return {"scan_id": scan_id, "message": "Mass scan started"}


@router.post("/preset/insider-probe")
async def run_insider_probe_preset(
    background_tasks: BackgroundTasks,
    request: PresetScanRequest = PresetScanRequest()
):
    """Run the insider probe preset to find suspicious accounts.

    Options:
    - use_cache=true: Load from existing cached results (fast)
    - use_cache=false: Run a new scan with --historical flag (slow, ~8-10 min)

    This executes the scripts/insider_probe.py logic in the background.
    """
    import json
    from pathlib import Path

    result_file = DATA_DIR / "insider_probe_results.json"
    scan_id = f"insider_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _count_flagged(data: dict) -> int:
        """Count total flagged accounts from probe results."""
        total = 0
        for priority in ['critical', 'high', 'medium', 'low']:
            total += len(data.get('priority_distribution', {}).get(priority, []))
        return total

    # Handle cache mode
    if request.use_cache:
        if not result_file.exists():
            raise HTTPException(status_code=404, detail="No cached insider probe results found. Run a new scan first.")

        try:
            with open(result_file) as f:
                data = json.load(f)
            accounts_found = _count_flagged(data)

            _preset_scans[scan_id] = {
                "scan_type": "insider_probe",
                "status": "completed",
                "progress_pct": 100,
                "message": f"Loaded {accounts_found} flagged accounts from cache",
                "accounts_found": accounts_found,
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "error": None,
                "from_cache": True,
                "cache_date": data.get("generated_at"),
                "mode": data.get("mode", "unknown"),
            }

            return {"scan_id": scan_id, "message": f"Loaded {accounts_found} flagged accounts from cache", "from_cache": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load cache: {str(e)}")

    # Check if already running
    if any(s["status"] == "running" and s["scan_type"] == "insider_probe" for s in _preset_scans.values()):
        raise HTTPException(status_code=409, detail="Insider probe already running")

    _preset_scans[scan_id] = {
        "scan_type": "insider_probe",
        "status": "running",
        "progress_pct": 0,
        "message": "Starting insider probe...",
        "accounts_found": 0,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "from_cache": False,
    }

    async def run_probe():
        try:
            script_path = SCRIPTS_DIR / "insider_probe.py"
            if not script_path.exists():
                _preset_scans[scan_id]["status"] = "error"
                _preset_scans[scan_id]["error"] = "insider_probe.py not found"
                return

            _preset_scans[scan_id]["message"] = "Running insider probe (historical mode)..."
            _preset_scans[scan_id]["progress_pct"] = 10

            # Run the script with --historical flag for retroactive detection
            process = await asyncio.create_subprocess_exec(
                "python3", str(script_path), "--historical",
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Load results
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    _preset_scans[scan_id]["accounts_found"] = _count_flagged(data)

                _preset_scans[scan_id]["status"] = "completed"
                _preset_scans[scan_id]["progress_pct"] = 100
                _preset_scans[scan_id]["message"] = f"Complete! Flagged {_preset_scans[scan_id]['accounts_found']} suspects"
            else:
                _preset_scans[scan_id]["status"] = "error"
                _preset_scans[scan_id]["error"] = stderr.decode()[:500]
                _preset_scans[scan_id]["message"] = "Probe failed"

            _preset_scans[scan_id]["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            _preset_scans[scan_id]["status"] = "error"
            _preset_scans[scan_id]["error"] = str(e)
            _preset_scans[scan_id]["completed_at"] = datetime.now().isoformat()

    background_tasks.add_task(run_probe)

    return {"scan_id": scan_id, "message": "Insider probe started"}


@router.get("/preset/{scan_id}")
async def get_preset_scan_status(scan_id: str):
    """Get the status of a preset scan."""
    if scan_id not in _preset_scans:
        raise HTTPException(status_code=404, detail="Scan not found")

    return _preset_scans[scan_id]


@router.get("/preset/status/all")
async def get_all_preset_status():
    """Get status of all preset scans."""
    return {
        "scans": list(_preset_scans.values()),
        "mass_scan_running": any(
            s["status"] == "running" and s["scan_type"] == "mass_scan"
            for s in _preset_scans.values()
        ),
        "insider_probe_running": any(
            s["status"] == "running" and s["scan_type"] == "insider_probe"
            for s in _preset_scans.values()
        ),
    }


@router.get("/preset/insider-probe/results")
async def get_insider_probe_results(
    priority: Optional[str] = None,
    limit: int = Query(default=100, le=500)
):
    """Get the latest insider probe results.

    Returns flagged accounts from the most recent probe run.

    Args:
        priority: Filter by priority (critical, high, medium, low)
        limit: Max number of results to return
    """
    import json
    from pathlib import Path

    result_file = DATA_DIR / "insider_probe_results.json"
    if not result_file.exists():
        return {"results": [], "total": 0, "message": "No probe results found. Run an insider probe first."}

    try:
        with open(result_file) as f:
            data = json.load(f)

        # Collect accounts from all priorities
        all_accounts = []
        for p in ['critical', 'high', 'medium', 'low']:
            accounts = data.get('priority_distribution', {}).get(p, [])
            for acc in accounts:
                acc['priority'] = p  # Ensure priority is set
            all_accounts.extend(accounts)

        # Filter by priority if specified
        if priority:
            all_accounts = [a for a in all_accounts if a.get('priority') == priority]

        # Apply limit
        all_accounts = all_accounts[:limit]

        return {
            "results": all_accounts,
            "total": len(all_accounts),
            "generated_at": data.get("generated_at"),
            "mode": data.get("mode"),
            "stats": data.get("stats", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")


@router.get("/preset/mass-scan/results")
async def get_mass_scan_results(
    min_profit: Optional[float] = None,
    limit: int = Query(default=100, le=500)
):
    """Get the latest mass scan results.

    Returns profitable accounts from the most recent scan.

    Args:
        min_profit: Filter by minimum profit USD
        limit: Max number of results to return
    """
    import json
    from pathlib import Path

    result_file = DATA_DIR / "analyzed_accounts.json"
    if not result_file.exists():
        return {"results": [], "total": 0, "message": "No scan results found. Run a mass scan first."}

    try:
        with open(result_file) as f:
            data = json.load(f)

        accounts = data.get("accounts", [])

        # Filter by min profit if specified
        if min_profit:
            accounts = [a for a in accounts if a.get("profit_usd", 0) >= min_profit]

        # Apply limit
        accounts = accounts[:limit]

        return {
            "results": accounts,
            "total": len(accounts),
            "generated_at": data.get("generated_at"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")


# =============================================================================
# Watchlist Management
# =============================================================================

# In-memory watchlist (would be database in production)
_watchlist: dict[str, dict] = {}


@router.get("/watchlist")
async def get_watchlist():
    """Get all accounts in the watchlist."""
    accounts = list(_watchlist.values())
    accounts.sort(key=lambda x: x.get("added_at", ""), reverse=True)
    return {
        "total": len(accounts),
        "accounts": accounts,
    }


@router.post("/watchlist/{wallet}")
async def add_to_watchlist(wallet: str, notes: str = ""):
    """Add an account to the watchlist."""
    import json
    from pathlib import Path
    from datetime import datetime

    wallet = wallet.lower()

    # Get account data from analyzed accounts if available
    data_file = DATA_DIR / "analyzed_accounts.json"
    account_data = None

    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
        for a in data.get("accounts", []):
            if a.get("wallet", "").lower() == wallet:
                account_data = a
                break

    _watchlist[wallet] = {
        "wallet": wallet,
        "added_at": datetime.utcnow().isoformat(),
        "notes": notes,
        "data": account_data,
    }

    # Remove from dismissed if it was there
    if wallet in _dismissed:
        del _dismissed[wallet]

    # Persist to unified prefs file
    _save_user_prefs()

    return {"message": "Added to watchlist", "wallet": wallet}


@router.delete("/watchlist/{wallet}")
async def remove_from_watchlist(wallet: str):
    """Remove an account from the watchlist."""
    wallet = wallet.lower()

    if wallet not in _watchlist:
        raise HTTPException(status_code=404, detail="Account not in watchlist")

    del _watchlist[wallet]

    # Persist to unified prefs file
    _save_user_prefs()

    return {"message": "Removed from watchlist", "wallet": wallet}


@router.patch("/watchlist/{wallet}")
async def update_watchlist_notes(wallet: str, notes: str):
    """Update notes for a watchlist account."""
    wallet = wallet.lower()

    if wallet not in _watchlist:
        raise HTTPException(status_code=404, detail="Account not in watchlist")

    _watchlist[wallet]["notes"] = notes

    # Persist to unified prefs file
    _save_user_prefs()

    return {"message": "Notes updated", "wallet": wallet}


# =============================================================================
# Dismissed Accounts (accounts user has reviewed and doesn't want to see)
# =============================================================================

_dismissed: dict[str, dict] = {}


@router.get("/dismissed")
async def get_dismissed():
    """Get all dismissed accounts."""
    accounts = list(_dismissed.values())
    accounts.sort(key=lambda x: x.get("dismissed_at", ""), reverse=True)
    return {
        "total": len(accounts),
        "accounts": accounts,
    }


@router.post("/dismissed/{wallet}")
async def dismiss_account(wallet: str, reason: str = ""):
    """Mark an account as dismissed (reviewed and not interested)."""
    import json
    from pathlib import Path
    from datetime import datetime

    wallet = wallet.lower()

    _dismissed[wallet] = {
        "wallet": wallet,
        "dismissed_at": datetime.utcnow().isoformat(),
        "reason": reason,
    }

    # Persist to file
    _save_user_prefs()

    return {"message": "Account dismissed", "wallet": wallet}


@router.delete("/dismissed/{wallet}")
async def undismiss_account(wallet: str):
    """Remove an account from the dismissed list."""
    wallet = wallet.lower()

    if wallet not in _dismissed:
        raise HTTPException(status_code=404, detail="Account not in dismissed list")

    del _dismissed[wallet]

    # Persist
    _save_user_prefs()

    return {"message": "Account undismissed", "wallet": wallet}


@router.get("/user-prefs")
async def get_user_prefs():
    """Get all user preferences (watchlist, dismissed, etc.)."""
    return {
        "watchlist": list(_watchlist.keys()),
        "dismissed": list(_dismissed.keys()),
        "watchlist_count": len(_watchlist),
        "dismissed_count": len(_dismissed),
    }


def _save_user_prefs():
    """Save user preferences (watchlist and dismissed) to persistent storage."""
    import json
    from pathlib import Path
    from datetime import datetime

    prefs_file = DATA_DIR / "user_prefs.json"

    # Only save if there's actual data to save
    if not _watchlist and not _dismissed:
        logger.debug("No user prefs to save (both empty)")
        return

    data = {
        "watchlist": list(_watchlist.values()),
        "dismissed": list(_dismissed.values()),
        "last_saved": datetime.utcnow().isoformat(),
    }

    try:
        with open(prefs_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("user_prefs_saved", watchlist_count=len(_watchlist), dismissed_count=len(_dismissed))
    except Exception as e:
        logger.error("user_prefs_save_error", error=str(e))


def _load_user_prefs():
    """Load user preferences from persistent storage."""
    import json
    from pathlib import Path

    prefs_file = DATA_DIR / "user_prefs.json"

    # Try new unified prefs file first
    if prefs_file.exists():
        try:
            with open(prefs_file) as f:
                data = json.load(f)

            loaded_watchlist = 0
            loaded_dismissed = 0

            for item in data.get("watchlist", []):
                if isinstance(item, dict):
                    wallet = item.get("wallet", "").lower()
                    if wallet:
                        _watchlist[wallet] = item
                        loaded_watchlist += 1
                elif isinstance(item, str):
                    # Handle old format where it was just wallet strings
                    _watchlist[item.lower()] = {"wallet": item.lower()}
                    loaded_watchlist += 1

            for item in data.get("dismissed", []):
                if isinstance(item, dict):
                    wallet = item.get("wallet", "").lower()
                    if wallet:
                        _dismissed[wallet] = item
                        loaded_dismissed += 1
                elif isinstance(item, str):
                    _dismissed[item.lower()] = {"wallet": item.lower()}
                    loaded_dismissed += 1

            logger.info("user_prefs_loaded", watchlist=loaded_watchlist, dismissed=loaded_dismissed)
            return
        except Exception as e:
            logger.error("user_prefs_load_error", error=str(e))

    # Fall back to old watchlist.json for migration
    watchlist_file = DATA_DIR / "watchlist.json"
    if watchlist_file.exists():
        try:
            with open(watchlist_file) as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    wallet = item.get("wallet", "").lower()
                    if wallet:
                        _watchlist[wallet] = item
                elif isinstance(item, str):
                    _watchlist[item.lower()] = {"wallet": item.lower()}
            # Migrate to new format
            _save_user_prefs()
            logger.info("user_prefs_migrated", watchlist=len(_watchlist))
        except Exception as e:
            logger.error("watchlist_migration_error", error=str(e))


# Load on module import
_load_user_prefs()


@router.get("/analyzed-stats")
async def get_analyzed_stats():
    """Get statistics from pre-analyzed accounts."""
    import json
    from pathlib import Path

    data_file = DATA_DIR / "analyzed_accounts.json"

    if not data_file.exists():
        return {"error": "Analyzed accounts data not found"}

    with open(data_file) as f:
        data = json.load(f)

    accounts = data.get("accounts", [])

    # Calculate score distribution from actual accounts
    score_dist = data.get("score_distribution") or {
        "90-100": len([a for a in accounts if a.get("systematic_score", 0) >= 90]),
        "80-89": len([a for a in accounts if 80 <= a.get("systematic_score", 0) < 90]),
        "70-79": len([a for a in accounts if 70 <= a.get("systematic_score", 0) < 80]),
        "65-69": len([a for a in accounts if 65 <= a.get("systematic_score", 0) < 70]),
    }

    # PnL tiers
    pnl_tiers = {
        "$1M+": len([a for a in accounts if a.get("total_pnl", 0) >= 1000000]),
        "$500k+": len([a for a in accounts if a.get("total_pnl", 0) >= 500000]),
        "$100k+": len([a for a in accounts if a.get("total_pnl", 0) >= 100000]),
        "$50k+": len([a for a in accounts if a.get("total_pnl", 0) >= 50000]),
    }

    # Activity metrics from new fields
    active_7d = len([a for a in accounts if a.get("is_currently_active", False) or a.get("activity_recency_days", 999) <= 7])
    active_14d = len([a for a in accounts if a.get("activity_recency_days", 999) <= 14])
    high_volume = len([a for a in accounts if a.get("num_trades", 0) >= 200])

    # Drawdown distribution
    low_drawdown = len([a for a in accounts if a.get("estimated_drawdown_pct", 0) < 10])
    medium_drawdown = len([a for a in accounts if 10 <= a.get("estimated_drawdown_pct", 0) < 20])
    high_drawdown = len([a for a in accounts if a.get("estimated_drawdown_pct", 0) >= 20])

    # Small fish: profitable but under radar ($5k-$50k)
    # Note: If collecting from leaderboard only, these may all be whales
    # In that case, show accounts in the lower PnL tier of the dataset
    small_fish = len([a for a in accounts if 5000 <= a.get("total_pnl", 0) < 50000])
    if small_fish == 0:
        # Fallback: count accounts in bottom 25% by PnL
        pnl_sorted = sorted([a.get("total_pnl", 0) for a in accounts])
        if pnl_sorted:
            p25 = pnl_sorted[len(pnl_sorted) // 4] if len(pnl_sorted) >= 4 else pnl_sorted[0]
            small_fish = len([a for a in accounts if a.get("total_pnl", 0) <= p25])

    return {
        "generated_at": data.get("generated_at"),
        "total_collected": data.get("total_collected", 0),
        "total_analyzed": len(accounts),
        "quality_count": data.get("quality_count", len(accounts)),
        "filters_applied": data.get("filters_applied", {}),
        "score_distribution": score_dist,
        "pnl_tiers": pnl_tiers,
        "pnl_distribution": data.get("pnl_distribution", {}),
        "activity_distribution": data.get("activity_distribution", {
            "active_7d": active_7d,
            "active_14d": active_14d,
        }),
        # Stats for simplified UI
        "score_90_plus": score_dist.get("90-100", 0),
        "score_80_89": score_dist.get("80-89", 0),
        "pnl_1m_plus": pnl_tiers.get("$1M+", 0),
        "active_7d": active_7d,
        "small_fish": small_fish,
        # Legacy fields
        "active_last_7_days": active_7d,
        "active_last_14_days": active_14d,
        "high_volume_traders": high_volume,
        "drawdown_distribution": {
            "low_0_10": low_drawdown,
            "medium_10_20": medium_drawdown,
            "high_20_plus": high_drawdown,
        },
    }


class FetchMoreTradesRequest(BaseModel):
    """Request to fetch more trades for an account."""
    wallet: str
    limit: int = Field(2000, ge=100, le=5000, description="Number of additional trades to fetch")


@router.post("/fetch-more-trades")
async def fetch_more_trades(request: FetchMoreTradesRequest):
    """Fetch more trades for an account and recalculate metrics.

    When an account hits the initial trade fetch limit (e.g., 500),
    this endpoint allows fetching a larger batch (e.g., 2000) and
    recalculates all metrics with the full dataset.
    """
    import json
    from pathlib import Path
    import sys

    # Add project path for imports
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.api.data import DataAPIClient, ActivityType

    wallet = request.wallet.lower()
    limit = request.limit

    # Get current account data
    data_file = DATA_DIR / "analyzed_accounts.json"
    if not data_file.exists():
        raise HTTPException(status_code=404, detail="No analyzed accounts found")

    with open(data_file) as f:
        data = json.load(f)

    # Find the account
    accounts = data.get("accounts", [])
    account_idx = None
    account_data = None

    for idx, a in enumerate(accounts):
        if a.get("wallet", "").lower() == wallet or a.get("wallet_address", "").lower() == wallet:
            account_idx = idx
            account_data = a
            break

    if account_data is None:
        raise HTTPException(status_code=404, detail="Account not found in analyzed data")

    # Fetch more trades AND redemptions
    try:
        async with DataAPIClient() as client:
            all_activities = []
            all_redeems = []
            offset = 0
            batch_size = 500

            # Paginate to get all TRADES up to limit
            while offset < limit:
                activities = await client.get_activity(
                    user=wallet,
                    activity_type=ActivityType.TRADE,
                    limit=batch_size,
                    offset=offset,
                )

                if not activities:
                    break

                all_activities.extend(activities)

                if len(activities) < batch_size:
                    break

                offset += batch_size

            # Also fetch REDEEM activities (resolved positions)
            offset = 0
            while offset < limit // 2:  # Fewer pages for redeems
                redeems = await client.get_activity(
                    user=wallet,
                    activity_type=ActivityType.REDEEM,
                    limit=batch_size,
                    offset=offset,
                )

                if not redeems:
                    break

                all_redeems.extend(redeems)

                if len(redeems) < batch_size:
                    break

                offset += batch_size

            if not all_activities:
                return {"account": account_data, "message": "No trades found"}

            # Recalculate metrics with the imported function
            from scripts.analyze_profitable import calculate_drawdown_metrics

            # Build metrics from activities
            from datetime import datetime
            from decimal import Decimal

            num_trades = len(all_activities)
            pnl = account_data.get("total_pnl", 0)

            # Position sizes
            sizes = [float(a.usd_value) for a in all_activities if a.usd_value > 0]
            if not sizes:
                return {"account": account_data, "message": "No valid trade sizes"}

            avg_position = sum(sizes) / len(sizes)
            sorted_sizes = sorted(sizes)
            median_position = sorted_sizes[len(sorted_sizes) // 2]
            max_position = max(sizes)

            # Position consistency
            if avg_position > 0:
                variance = sum((s - avg_position) ** 2 for s in sizes) / len(sizes)
                std_dev = variance ** 0.5
                position_consistency = 1 - min(1, std_dev / avg_position)
            else:
                position_consistency = 0

            # Activity metrics (include both trades AND redeems for timestamps)
            now = datetime.now()
            dates = set()
            timestamps_dt = []

            # Trade timestamps
            for a in all_activities:
                try:
                    dates.add(a.timestamp.strftime("%Y-%m-%d"))
                    timestamps_dt.append(a.timestamp)
                except:
                    pass

            # Also include redeem timestamps for activity recency
            for r in all_redeems:
                try:
                    timestamps_dt.append(r.timestamp)
                except:
                    pass

            active_days = len(dates)  # Active TRADING days only

            last_trade_timestamp = None
            if len(timestamps_dt) >= 2:
                sorted_ts = sorted(timestamps_dt)
                account_age_days = max(1, (sorted_ts[-1] - sorted_ts[0]).days)
                activity_recency_days = (now - sorted_ts[-1]).days
                last_trade_timestamp = sorted_ts[-1].isoformat()
            elif len(timestamps_dt) == 1:
                account_age_days = max(30, active_days * 2)
                activity_recency_days = (now - timestamps_dt[0]).days
                last_trade_timestamp = timestamps_dt[0].isoformat()
            else:
                account_age_days = max(30, active_days * 2)
                activity_recency_days = 999

            trades_per_week = (num_trades / max(1, account_age_days)) * 7

            # Unique markets (from trades only - redeems have incomplete market data)
            markets = set(a.condition_id for a in all_activities if a.condition_id)
            unique_markets = len(markets)

            # Buy/sell ratio (from trades only)
            buys = sum(1 for a in all_activities if a.side and a.side.value == "BUY")
            buy_sell_ratio = buys / num_trades if num_trades > 0 else 0.5

            # Derived metrics
            pnl_per_trade = pnl / num_trades if num_trades > 0 else 0
            pnl_per_market = pnl / unique_markets if unique_markets > 0 else 0

            # Recent activity (trades)
            trades_last_7d = 0
            trades_last_30d = 0
            for a in all_activities:
                try:
                    days_ago = (now - a.timestamp).days
                    if days_ago <= 7:
                        trades_last_7d += 1
                        trades_last_30d += 1
                    elif days_ago <= 30:
                        trades_last_30d += 1
                except:
                    pass

            # Count recent redeems for activity status
            redeems_last_7d = 0
            for r in all_redeems:
                try:
                    days_ago = (now - r.timestamp).days
                    if days_ago <= 7:
                        redeems_last_7d += 1
                except:
                    pass

            # Account is active if they traded OR redeemed in last 7 days
            is_currently_active = (trades_last_7d > 0) or (redeems_last_7d > 0)

            # Calculate drawdowns with full data including redemptions
            drawdown_metrics = calculate_drawdown_metrics(all_activities, pnl, all_redeems)

            # Win rate estimate - more varied based on actual performance metrics
            total_volume = sum(sizes)
            return_on_volume = pnl / total_volume if total_volume > 0 else 0

            if total_pnl > 0:
                # Profitable accounts: estimate win rate 52-68% based on efficiency
                # Higher return on volume = higher estimated win rate
                win_rate = min(0.68, max(0.52, 0.52 + return_on_volume * 2))
                # Bonus for high trade count (more reliable data)
                if num_trades >= 200:
                    win_rate = min(0.70, win_rate + 0.02)
            else:
                # Losing accounts: estimate win rate 35-48%
                win_rate = max(0.35, min(0.48, 0.48 + return_on_volume * 2))

            # Update account data
            updated_account = {
                **account_data,
                "num_trades": num_trades,
                "trades_fetched": num_trades,
                "avg_position": avg_position,
                "avg_position_size": avg_position,
                "median_position": median_position,
                "max_position": max_position,
                "position_consistency": position_consistency,
                "account_age_days": account_age_days,
                "active_days": active_days,
                "unique_markets": unique_markets,
                "trades_per_week": trades_per_week,
                "pnl_per_trade": pnl_per_trade,
                "pnl_per_market": pnl_per_market,
                "activity_recency_days": activity_recency_days,
                "buy_sell_ratio": buy_sell_ratio,
                "win_rate": win_rate,
                "trades_last_7d": trades_last_7d,
                "trades_last_30d": trades_last_30d,
                "is_currently_active": is_currently_active,
                "last_trade_timestamp": last_trade_timestamp,
                # Drawdown metrics
                "max_drawdown_pct": drawdown_metrics["max_drawdown_pct"],
                "max_drawdown_usd": drawdown_metrics["max_drawdown_usd"],
                "avg_drawdown_pct": drawdown_metrics["avg_drawdown_pct"],
                "avg_drawdown_usd": drawdown_metrics["avg_drawdown_usd"],
                "drawdown_count": drawdown_metrics["drawdown_count"],
                "severe_drawdown_count": drawdown_metrics["severe_drawdown_count"],
                "drawdown_frequency": drawdown_metrics["drawdown_frequency"],
                "avg_recovery_trades": drawdown_metrics["avg_recovery_trades"],
                "current_drawdown_pct": drawdown_metrics["current_drawdown_pct"],
                "pl_curve_smoothness": drawdown_metrics["pl_curve_smoothness"],
                "profit_factor": drawdown_metrics["profit_factor"],
            }

            # Update in the data file
            accounts[account_idx] = updated_account
            data["accounts"] = accounts

            with open(data_file, "w") as f:
                json.dump(data, f, indent=2)

            return {
                "account": updated_account,
                "message": f"Updated with {num_trades} trades (was {account_data.get('num_trades', 0)})",
            }

    except Exception as e:
        logger.error("fetch_more_trades_failed", wallet=wallet[:10], error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch trades: {str(e)}")


# =============================================================================
# Profile & Position Endpoints (NEW)
# =============================================================================

# In-memory cache for usernames (avoids repeated API calls)
_username_cache: dict[str, str] = {}


@router.get("/profile/{wallet_address}")
async def get_wallet_profile(wallet_address: str):
    """Fetch Polymarket profile data for a wallet including username.

    Tries multiple API endpoints to get user information.
    Results are cached to avoid repeated API calls.
    """
    import aiohttp

    wallet = wallet_address.lower()

    # Check cache first
    if wallet in _username_cache:
        return {
            "wallet": wallet,
            "username": _username_cache[wallet],
            "cached": True,
        }

    try:
        async with aiohttp.ClientSession() as session:
            # Try the Gamma API endpoint for user profiles
            endpoints = [
                f"https://gamma-api.polymarket.com/profiles/{wallet}",
                f"https://gamma-api.polymarket.com/users/{wallet}",
            ]

            for url in endpoints:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                        if resp.status == 200:
                            content_type = resp.headers.get('content-type', '')
                            if 'application/json' in content_type:
                                data = await resp.json()
                                username = (data.get("username") or data.get("name") or
                                           data.get("displayName") or data.get("display_name"))

                                if username:
                                    _username_cache[wallet] = username
                                    return {
                                        "wallet": wallet,
                                        "username": username,
                                        "profileUrl": data.get("profileImage") or data.get("profile_image"),
                                        "cached": False,
                                    }
                except Exception:
                    continue

        # No username found - cache as None
        _username_cache[wallet] = None
        return {
            "wallet": wallet,
            "username": None,
            "cached": False,
        }

    except Exception as e:
        logger.warning("profile_fetch_failed", wallet=wallet[:10], error=str(e))
        return {
            "wallet": wallet,
            "username": None,
            "error": str(e),
        }


@router.post("/profiles/batch")
async def get_wallet_profiles_batch(wallets: List[str] = Body(...)):
    """Fetch profiles for multiple wallets in batch.

    Returns a dict mapping wallet addresses to usernames.
    Efficiently fetches only non-cached wallets.
    Note: Polymarket's profile API has limited availability.
    """
    import aiohttp
    import asyncio

    results = {}
    wallets_to_fetch = []

    # Check cache first
    for wallet in wallets:
        wallet_lower = wallet.lower()
        if wallet_lower in _username_cache:
            results[wallet_lower] = _username_cache[wallet_lower]
        else:
            wallets_to_fetch.append(wallet_lower)

    # Fetch non-cached wallets (limit concurrency)
    if wallets_to_fetch:
        async def fetch_profile(session: aiohttp.ClientSession, wallet: str):
            # Try multiple endpoints
            endpoints = [
                f"https://gamma-api.polymarket.com/profiles/{wallet}",
                f"https://gamma-api.polymarket.com/users/{wallet}",
            ]
            for url in endpoints:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            content_type = resp.headers.get('content-type', '')
                            if 'application/json' in content_type:
                                data = await resp.json()
                                username = (data.get("username") or data.get("name") or
                                           data.get("displayName") or data.get("display_name"))
                                if username:
                                    _username_cache[wallet] = username
                                    return wallet, username
                except:
                    continue

            _username_cache[wallet] = None
            return wallet, None

        async with aiohttp.ClientSession() as session:
            # Fetch in batches of 5 to avoid overwhelming the API
            for i in range(0, min(len(wallets_to_fetch), 20), 5):  # Max 20 fetches
                batch = wallets_to_fetch[i:i+5]
                tasks = [fetch_profile(session, w) for w in batch]
                batch_results = await asyncio.gather(*tasks)
                for wallet, username in batch_results:
                    results[wallet] = username
                if i + 5 < len(wallets_to_fetch):
                    await asyncio.sleep(0.05)

    return {"profiles": results}


@router.get("/positions/{wallet_address}")
async def get_wallet_positions(wallet_address: str, limit: int = Query(20, ge=1, le=100)):
    """Fetch current positions for a wallet.

    Returns the wallet's current market positions with:
    - Market info
    - Position size
    - Current value
    - Unrealized P/L
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.api.data import DataAPIClient, PositionSortBy
    from src.api.gamma import GammaAPIClient

    wallet = wallet_address.lower()

    try:
        async with DataAPIClient() as client:
            positions = await client.get_positions(
                user=wallet,
                sort_by=PositionSortBy.CURRENT,
                limit=limit,
            )

            # Convert to serializable format
            positions_data = []
            for pos in positions:
                positions_data.append({
                    "condition_id": pos.condition_id,
                    "token_id": pos.token_id,
                    "outcome": pos.outcome,
                    "size": float(pos.size),
                    "avg_price": float(pos.average_price),
                    "current_value": float(pos.current_value),
                    "initial_value": float(pos.initial_value),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "market_slug": pos.market_slug,
                    "market_title": pos.market_title,
                })

            return {
                "wallet": wallet,
                "positions": positions_data,
                "count": len(positions_data),
            }

    except Exception as e:
        logger.error("positions_fetch_failed", wallet=wallet[:10], error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")


@router.get("/presets")
async def get_scan_presets():
    """Get available scan presets with their configurations.

    Returns preset configurations for:
    - wide_net: Maximum collection (5000+ candidates, all categories)
    - niche: Focused on niche market specialists
    - quick: Fast testing scan with minimal candidates
    """
    return {
        "presets": [
            {
                "id": "wide_net",
                "name": "Wide Net Collection",
                "description": "Cast the widest net. Collect ALL accounts with $1000+ profit from all leaderboard categories. Minimal filtering - sort later by profitability.",
                "recommended": True,
                "config": {
                    "mode": "wide_net_profitability",
                    "max_candidates": 5000,
                    "max_phase2": 1000,
                    "max_phase3": 200,
                    "min_score_threshold": 20,
                    "categories": None,  # All categories
                    "persist_to_db": True,
                }
            },
            {
                "id": "niche",
                "name": "Niche Focused",
                "description": "Focus on niche market specialists trading in weather, economics, and tech categories.",
                "recommended": False,
                "config": {
                    "mode": "niche_specialist",
                    "max_candidates": 2000,
                    "max_phase2": 500,
                    "max_phase3": 100,
                    "min_score_threshold": 50,
                    "categories": ["WEATHER", "ECONOMICS", "TECH", "SCIENCE"],
                    "persist_to_db": True,
                }
            },
            {
                "id": "quick",
                "name": "Quick Test Scan",
                "description": "Fast scan for testing with minimal candidates. Good for verifying setup.",
                "recommended": False,
                "config": {
                    "mode": "wide_net_profitability",
                    "max_candidates": 200,
                    "max_phase2": 50,
                    "max_phase3": 15,
                    "min_score_threshold": 20,
                    "categories": ["ECONOMICS", "TECH"],
                    "persist_to_db": False,
                }
            },
        ]
    }


# =============================================================================
# Lazy Categorization Endpoint
# =============================================================================

class CategorizeRequest(BaseModel):
    """Request to categorize accounts lazily."""
    wallet_addresses: List[str] = Field(..., min_length=1, max_length=50)
    sample_trades: int = Field(50, ge=10, le=200)


@router.post("/categorize")
async def categorize_accounts(request: CategorizeRequest):
    """Lazily categorize accounts by sampling their recent trades.

    This is a lightweight categorization that:
    1. Fetches a sample of recent trades (default 50)
    2. Looks up market titles for those trades
    3. Categorizes based on keywords
    4. Returns category breakdown for each account

    Use this after scanning to categorize accounts without
    slowing down the main scan process.
    """
    from src.api.data import ActivityType

    try:
        service = await get_discovery_service()
        analyzer = service._analyzer

        results = {}

        for wallet in request.wallet_addresses:
            try:
                # Fetch sample of recent trades - try TRADE first, then all activities
                activities = await analyzer._data_client.get_activity(
                    user=wallet.lower(),
                    activity_type=ActivityType.TRADE,
                    limit=request.sample_trades,
                )

                # If no TRADE activities, try fetching all activities (including REDEEM)
                # This helps categorize accounts that mostly hold positions to redemption
                if not activities:
                    activities = await analyzer._data_client.get_activity(
                        user=wallet.lower(),
                        limit=request.sample_trades,
                    )


                if not activities:
                    results[wallet] = {
                        "primary_category": "Unknown",
                        "category_breakdown": {},
                        "trades_sampled": 0,
                    }
                    continue

                # Categorize each trade
                from collections import defaultdict
                category_counts = defaultdict(int)

                for activity in activities:
                    try:
                        # Get market title - use cache or fetch
                        condition_id = activity.condition_id
                        market_title = await analyzer._get_market_title(condition_id)
                        category = analyzer._categorize_market(market_title)
                        category_counts[category] += 1
                    except Exception:
                        category_counts["other"] += 1

                # Calculate breakdown percentages
                total = sum(category_counts.values())
                breakdown = {
                    cat: round(count / total * 100, 1)
                    for cat, count in category_counts.items()
                }

                # Find primary category
                primary = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else "other"
                concentration = breakdown.get(primary, 0)

                # If no single category dominates (>60%), mark as Diversified
                if concentration < 60:
                    primary = "Diversified"

                results[wallet] = {
                    "primary_category": primary.title() if primary != "Diversified" else "Diversified",
                    "category_concentration": concentration,
                    "category_breakdown": breakdown,
                    "trades_sampled": len(activities),
                }

            except Exception as e:
                logger.warning("categorize_failed", wallet=wallet[:10], error=str(e))
                results[wallet] = {
                    "primary_category": "Unknown",
                    "category_breakdown": {},
                    "trades_sampled": 0,
                    "error": str(e),
                }

        return {
            "categorized": len([r for r in results.values() if r.get("trades_sampled", 0) > 0]),
            "failed": len([r for r in results.values() if r.get("trades_sampled", 0) == 0]),
            "results": results,
        }

    except Exception as e:
        logger.error("categorize_batch_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
