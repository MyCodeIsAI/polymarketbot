"""REST API routes for Insider Scanner dashboard.

Provides endpoints for:
- Watchlist management (flagged wallets)
- Detection record viewing
- Scoring analysis
- Alert configuration
- Audit trail export
- Scanner control
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from contextlib import contextmanager

from ...utils.logging import get_logger
from .dependencies import get_db, get_session

logger = get_logger(__name__)


@contextmanager
def session_wrapper(session):
    """Wrap a session to work with SybilDetector's context manager pattern."""
    try:
        yield session
    finally:
        pass  # Don't close - FastAPI manages the session lifecycle


def make_session_factory(session):
    """Create a session factory compatible with SybilDetector."""
    return lambda: session_wrapper(session)

router = APIRouter(prefix="/insider", tags=["insider-scanner"])


# =============================================================================
# Pydantic Models
# =============================================================================


class WalletSummary(BaseModel):
    """Summary of a flagged wallet."""

    id: int
    wallet_address: str
    flag_type: str = "unknown"  # insider, sybil, unknown
    linked_trader_id: Optional[int] = None
    funding_match_address: Optional[str] = None
    insider_score: float
    priority: str
    status: str
    signal_count: int
    active_dimensions: int
    account_age_days: Optional[int] = None
    total_position_usd: Optional[float] = None
    win_rate: Optional[float] = None
    detected_at: Optional[str] = None


class WalletDetail(BaseModel):
    """Full details of a flagged wallet."""

    id: int
    wallet_address: str
    flag_type: str = "unknown"  # insider, sybil, unknown
    linked_trader_id: Optional[int] = None
    linked_trader_wallet: Optional[str] = None  # The profitable trader's wallet
    linked_trader_profit: Optional[float] = None  # The profitable trader's profit
    funding_match_address: Optional[str] = None
    insider_score: float
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    priority: str
    status: str
    dimensions: dict
    signals: Optional[List[dict]] = None
    signal_count: int
    active_dimensions: int
    downgraded: bool = False
    downgrade_reason: Optional[str] = None
    account_age_days: Optional[int] = None
    transaction_count: Optional[int] = None
    total_position_usd: Optional[float] = None
    largest_position_usd: Optional[float] = None
    win_rate: Optional[float] = None
    cluster_id: Optional[int] = None
    notes: Optional[str] = None
    detected_at: Optional[str] = None
    last_scored_at: Optional[str] = None


class WalletCreate(BaseModel):
    """Request to manually add a wallet to watchlist."""

    wallet_address: str = Field(..., pattern="^0x[a-fA-F0-9]{40}$")
    priority: str = Field("medium", pattern="^(critical|high|medium|low|normal)$")
    notes: Optional[str] = None
    auto_score: bool = Field(True, description="Run scoring algorithm on wallet")


class WalletUpdate(BaseModel):
    """Request to update wallet status/notes."""

    status: Optional[str] = Field(None, pattern="^(new|monitoring|escalated|confirmed|cleared|archived)$")
    notes: Optional[str] = None
    priority: Optional[str] = Field(None, pattern="^(critical|high|medium|low|normal)$")


class DetectionRecordResponse(BaseModel):
    """Detection record for audit trail."""

    id: int
    record_hash: str
    wallet_address: str
    detected_at: str
    insider_score: float
    priority: str
    signal_count: int
    market_ids: Optional[List[str]] = None
    anchor_tx_id: Optional[str] = None
    anchor_type: Optional[str] = None


class InvestmentThesisCreate(BaseModel):
    """Create an investment thesis before acting."""

    detection_record_ids: List[int]
    reasoning: str
    intended_action: Optional[str] = None
    market_id: Optional[str] = None
    position_side: Optional[str] = Field(None, pattern="^(YES|NO)$")
    position_size: Optional[float] = None


class InvestmentThesisResponse(BaseModel):
    """Investment thesis record."""

    id: int
    thesis_hash: str
    created_at: str
    detection_record_ids: List[int]
    reasoning: str
    intended_action: Optional[str] = None
    market_id: Optional[str] = None
    position_side: Optional[str] = None
    position_size: Optional[float] = None
    action_taken: bool = False


class ScannerStatusResponse(BaseModel):
    """Status of the insider scanner."""

    is_running: bool
    mode: str  # websocket, polling, stopped
    connected_markets: int = 0
    wallets_monitored: int = 0
    alerts_today: int = 0
    last_event_at: Optional[str] = None
    uptime_seconds: Optional[float] = None


class AlertConfigResponse(BaseModel):
    """Alert configuration for insider scanner."""

    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None
    email_enabled: bool = False
    email_recipients: Optional[List[str]] = None
    min_priority: str = "medium"
    quiet_hours_start: Optional[int] = None
    quiet_hours_end: Optional[int] = None


class ScannerStats(BaseModel):
    """Statistics for the insider scanner."""

    total_wallets_flagged: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    clusters_detected: int
    funding_sources_flagged: int
    detections_today: int
    alerts_sent_today: int


# =============================================================================
# Watchlist Endpoints
# =============================================================================


@router.get("/watchlist", response_model=List[WalletSummary])
async def list_flagged_wallets(
    priority: Optional[str] = Query(None, description="Filter by priority"),
    status: Optional[str] = Query(None, description="Filter by status"),
    flag_type: Optional[str] = Query(None, description="Filter by flag type (insider, sybil)"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum score"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db=Depends(get_session),
):
    """List all flagged wallets in the watchlist."""
    from ...insider_scanner.models import FlaggedWallet, InsiderPriority, WalletStatus, FlagType

    query = db.query(FlaggedWallet)

    if priority:
        try:
            query = query.filter(FlaggedWallet.priority == InsiderPriority(priority))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

    if status:
        try:
            query = query.filter(FlaggedWallet.status == WalletStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if flag_type:
        try:
            query = query.filter(FlaggedWallet.flag_type == FlagType(flag_type))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid flag_type: {flag_type}")

    if min_score is not None:
        query = query.filter(FlaggedWallet.insider_score >= min_score)

    wallets = (
        query.order_by(FlaggedWallet.insider_score.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        WalletSummary(
            id=w.id,
            wallet_address=w.wallet_address,
            flag_type=w.flag_type.value if w.flag_type else "unknown",
            linked_trader_id=w.linked_trader_id,
            funding_match_address=w.funding_match_address,
            insider_score=float(w.insider_score) if w.insider_score else 0,
            priority=w.priority.value,
            status=w.status.value,
            signal_count=w.signal_count or 0,
            active_dimensions=w.active_dimensions or 0,
            account_age_days=w.account_age_days,
            total_position_usd=float(w.total_position_usd) if w.total_position_usd else None,
            win_rate=float(w.win_rate) if w.win_rate else None,
            detected_at=w.detected_at.isoformat() if w.detected_at else None,
        )
        for w in wallets
    ]


@router.get("/watchlist/{wallet_id}", response_model=WalletDetail)
async def get_wallet_detail(wallet_id: int, db=Depends(get_session)):
    """Get detailed information for a flagged wallet."""
    from ...insider_scanner.models import FlaggedWallet, KnownProfitableTrader

    wallet = db.query(FlaggedWallet).filter(FlaggedWallet.id == wallet_id).first()
    if not wallet:
        raise HTTPException(status_code=404, detail="Wallet not found")

    # Get linked trader info if this is a sybil wallet
    linked_trader_wallet = None
    linked_trader_profit = None
    if wallet.linked_trader_id:
        linked_trader = db.query(KnownProfitableTrader).filter(
            KnownProfitableTrader.id == wallet.linked_trader_id
        ).first()
        if linked_trader:
            linked_trader_wallet = linked_trader.wallet_address
            linked_trader_profit = float(linked_trader.profit_usd) if linked_trader.profit_usd else None

    return WalletDetail(
        id=wallet.id,
        wallet_address=wallet.wallet_address,
        flag_type=wallet.flag_type.value if wallet.flag_type else "unknown",
        linked_trader_id=wallet.linked_trader_id,
        linked_trader_wallet=linked_trader_wallet,
        linked_trader_profit=linked_trader_profit,
        funding_match_address=wallet.funding_match_address,
        insider_score=float(wallet.insider_score) if wallet.insider_score else 0,
        confidence_low=float(wallet.confidence_low) if wallet.confidence_low else None,
        confidence_high=float(wallet.confidence_high) if wallet.confidence_high else None,
        priority=wallet.priority.value,
        status=wallet.status.value,
        dimensions={
            "account": float(wallet.score_account) if wallet.score_account else 0,
            "trading": float(wallet.score_trading) if wallet.score_trading else 0,
            "behavioral": float(wallet.score_behavioral) if wallet.score_behavioral else 0,
            "contextual": float(wallet.score_contextual) if wallet.score_contextual else 0,
            "cluster": float(wallet.score_cluster) if wallet.score_cluster else 0,
        },
        signals=wallet.signals_json,
        signal_count=wallet.signal_count or 0,
        active_dimensions=wallet.active_dimensions or 0,
        downgraded=wallet.downgraded or False,
        downgrade_reason=wallet.downgrade_reason,
        account_age_days=wallet.account_age_days,
        transaction_count=wallet.transaction_count,
        total_position_usd=float(wallet.total_position_usd) if wallet.total_position_usd else None,
        largest_position_usd=float(wallet.largest_position_usd) if wallet.largest_position_usd else None,
        win_rate=float(wallet.win_rate) if wallet.win_rate else None,
        cluster_id=wallet.cluster_id,
        notes=wallet.notes,
        detected_at=wallet.detected_at.isoformat() if wallet.detected_at else None,
        last_scored_at=wallet.last_scored_at.isoformat() if wallet.last_scored_at else None,
    )


@router.post("/watchlist", response_model=WalletSummary, status_code=201)
async def add_wallet_to_watchlist(
    request: WalletCreate,
    background_tasks: BackgroundTasks,
    db=Depends(get_session),
):
    """Manually add a wallet to the watchlist."""
    from ...insider_scanner.models import FlaggedWallet, InsiderPriority, WalletStatus

    # Check if already exists
    existing = (
        db.query(FlaggedWallet)
        .filter(FlaggedWallet.wallet_address == request.wallet_address.lower())
        .first()
    )

    if existing:
        raise HTTPException(status_code=400, detail="Wallet already in watchlist")

    # Create wallet
    wallet = FlaggedWallet(
        wallet_address=request.wallet_address.lower(),
        insider_score=Decimal("0"),
        priority=InsiderPriority(request.priority),
        status=WalletStatus.NEW,
        notes=request.notes,
    )

    db.add(wallet)
    db.commit()
    db.refresh(wallet)

    logger.info("wallet_added_to_watchlist", wallet=request.wallet_address, priority=request.priority)

    # Optionally score wallet in background
    if request.auto_score:
        background_tasks.add_task(_score_wallet_background, wallet.id)

    return WalletSummary(
        id=wallet.id,
        wallet_address=wallet.wallet_address,
        insider_score=float(wallet.insider_score) if wallet.insider_score else 0,
        priority=wallet.priority.value,
        status=wallet.status.value,
        signal_count=wallet.signal_count or 0,
        active_dimensions=wallet.active_dimensions or 0,
        detected_at=wallet.detected_at.isoformat() if wallet.detected_at else None,
    )


@router.patch("/watchlist/{wallet_id}", response_model=WalletSummary)
async def update_wallet(wallet_id: int, update: WalletUpdate, db=Depends(get_session)):
    """Update a flagged wallet's status or notes."""
    from ...insider_scanner.models import FlaggedWallet, InsiderPriority, WalletStatus

    wallet = db.query(FlaggedWallet).filter(FlaggedWallet.id == wallet_id).first()
    if not wallet:
        raise HTTPException(status_code=404, detail="Wallet not found")

    if update.status:
        wallet.status = WalletStatus(update.status)
    if update.notes is not None:
        wallet.notes = update.notes
    if update.priority:
        wallet.priority = InsiderPriority(update.priority)

    db.commit()
    db.refresh(wallet)

    logger.info("wallet_updated", wallet_id=wallet_id)

    return WalletSummary(
        id=wallet.id,
        wallet_address=wallet.wallet_address,
        insider_score=float(wallet.insider_score) if wallet.insider_score else 0,
        priority=wallet.priority.value,
        status=wallet.status.value,
        signal_count=wallet.signal_count or 0,
        active_dimensions=wallet.active_dimensions or 0,
        detected_at=wallet.detected_at.isoformat() if wallet.detected_at else None,
    )


@router.delete("/watchlist/{wallet_id}", status_code=204)
async def remove_wallet(wallet_id: int, db=Depends(get_session)):
    """Remove a wallet from the watchlist."""
    from ...insider_scanner.models import FlaggedWallet

    wallet = db.query(FlaggedWallet).filter(FlaggedWallet.id == wallet_id).first()
    if not wallet:
        raise HTTPException(status_code=404, detail="Wallet not found")

    db.delete(wallet)
    db.commit()

    logger.info("wallet_removed_from_watchlist", wallet_id=wallet_id)


# =============================================================================
# Detection Records (Audit Trail)
# =============================================================================


@router.get("/detections", response_model=List[DetectionRecordResponse])
async def list_detection_records(
    wallet_address: Optional[str] = Query(None, description="Filter by wallet"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_session),
):
    """List detection records for audit trail."""
    from ...insider_scanner.models import DetectionRecord

    query = db.query(DetectionRecord)

    if wallet_address:
        query = query.filter(DetectionRecord.wallet_address == wallet_address.lower())
    if priority:
        query = query.filter(DetectionRecord.priority == priority)

    records = query.order_by(DetectionRecord.detected_at.desc()).limit(limit).all()

    return [
        DetectionRecordResponse(
            id=r.id,
            record_hash=r.record_hash,
            wallet_address=r.wallet_address,
            detected_at=r.detected_at.isoformat() if r.detected_at else "",
            insider_score=float(r.insider_score),
            priority=r.priority,
            signal_count=len(r.signals_snapshot) if r.signals_snapshot else 0,
            market_ids=r.market_ids,
            anchor_tx_id=r.anchor_tx_id,
            anchor_type=r.anchor_type,
        )
        for r in records
    ]


@router.get("/detections/{record_id}")
async def get_detection_record(record_id: int, db=Depends(get_session)):
    """Get full detection record with signals."""
    from ...insider_scanner.models import DetectionRecord
    from ...insider_scanner.audit import AuditTrailManager

    record = db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Detection record not found")

    # Verify integrity
    audit = AuditTrailManager(db)
    verification = audit.verify_detection_record(record_id)

    return {
        "record": record.to_dict(),
        "verification": verification,
    }


@router.get("/detections/{record_id}/export")
async def export_detection_record(record_id: int, db=Depends(get_session)):
    """Export a detection record for legal purposes."""
    from ...insider_scanner.audit import AuditTrailManager

    audit = AuditTrailManager(db)
    export = audit.export_detection_record(record_id)

    if "error" in export:
        raise HTTPException(status_code=404, detail=export["error"])

    return export


# =============================================================================
# Investment Thesis
# =============================================================================


@router.post("/thesis", response_model=InvestmentThesisResponse, status_code=201)
async def create_investment_thesis(request: InvestmentThesisCreate, db=Depends(get_session)):
    """Create an investment thesis before acting on detection."""
    from ...insider_scanner.audit import AuditTrailManager

    audit = AuditTrailManager(db)

    thesis = audit.create_investment_thesis(
        detection_record_ids=request.detection_record_ids,
        reasoning=request.reasoning,
        intended_action=request.intended_action,
        market_id=request.market_id,
        position_side=request.position_side,
        position_size=request.position_size,
    )

    db.commit()

    return InvestmentThesisResponse(
        id=thesis.id,
        thesis_hash=thesis.thesis_hash,
        created_at=thesis.created_at.isoformat() if thesis.created_at else "",
        detection_record_ids=thesis.detection_record_ids,
        reasoning=thesis.reasoning,
        intended_action=thesis.intended_action,
        market_id=thesis.market_id,
        position_side=thesis.position_side,
        position_size=float(thesis.position_size) if thesis.position_size else None,
        action_taken=thesis.action_taken or False,
    )


@router.get("/thesis")
async def list_investment_theses(
    limit: int = Query(50, ge=1, le=500),
    db=Depends(get_session),
):
    """List investment theses."""
    from ...insider_scanner.models import InvestmentThesis

    theses = (
        db.query(InvestmentThesis)
        .order_by(InvestmentThesis.created_at.desc())
        .limit(limit)
        .all()
    )

    return [t.to_dict() for t in theses]


# =============================================================================
# Audit Trail
# =============================================================================


@router.get("/audit/export")
async def export_audit_trail(
    wallet_address: Optional[str] = Query(None, description="Filter by wallet"),
    db=Depends(get_session),
):
    """Export full audit trail for legal purposes."""
    from ...insider_scanner.audit import AuditTrailManager

    audit = AuditTrailManager(db)
    export = audit.export_full_audit_trail(wallet_address=wallet_address)

    return export


@router.get("/audit/verify")
async def verify_chain_integrity(db=Depends(get_session)):
    """Verify the integrity of the audit chain."""
    from ...insider_scanner.audit import AuditTrailManager

    audit = AuditTrailManager(db)
    result = audit.verify_chain_integrity()

    return result


# =============================================================================
# Scanner Control
# =============================================================================

# Global scanner service instance (initialized on first start)
_scanner_service = None
_scanner_lock = None


def _get_scanner_lock():
    """Get or create the scanner lock."""
    global _scanner_lock
    import asyncio
    if _scanner_lock is None:
        _scanner_lock = asyncio.Lock()
    return _scanner_lock


async def _get_or_create_scanner(session_factory):
    """Get or create the scanner service instance."""
    global _scanner_service
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from ...insider_scanner.scanner_service import InsiderScannerService

    # Load .env file for API keys
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    async with _get_scanner_lock():
        if _scanner_service is None:
            # Get API key from environment for blockchain monitoring
            api_key = os.getenv("POLYGONSCAN_API_KEY") or os.getenv("ETHERSCAN_API_KEY")

            logger.info("scanner_init", api_key_set=bool(api_key))

            _scanner_service = InsiderScannerService(
                session_factory=session_factory,
                polygonscan_api_key=api_key,
                auto_populate=True,
            )
    return _scanner_service


@router.get("/status", response_model=ScannerStatusResponse)
async def get_scanner_status(db=Depends(get_session)):
    """Get current scanner status including real-time statistics.

    Checks systemd service status and queries scanner API if running.
    """
    import asyncio
    import subprocess

    # Check if the insider-scanner systemd service is running
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "insider-scanner"],
            capture_output=True,
            text=True,
            timeout=5
        )
        service_running = result.stdout.strip() == "active"
    except Exception:
        service_running = False

    if service_running:
        # Service is running - try to get detailed stats via internal API
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://127.0.0.1:8080/api/stats",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        uptime = None
                        if data.get("started_at"):
                            started = datetime.fromisoformat(data["started_at"])
                            uptime = (datetime.utcnow() - started).total_seconds()

                        bm = data.get("blockchain_monitor", {})

                        return ScannerStatusResponse(
                            is_running=True,
                            mode=data.get("mode", "full"),
                            connected_markets=bm.get("blocks_scanned", 0),
                            wallets_monitored=data.get("insider", {}).get("wallets_scored", 0),
                            alerts_today=data.get("alerts", {}).get("total", 0),
                            last_event_at=bm.get("last_scan_at"),  # Keep as string
                            uptime_seconds=uptime,
                        )
        except Exception as e:
            logger.debug("scanner_api_check_failed", error=str(e))

        # Service running but can't get stats - return basic running status
        return ScannerStatusResponse(
            is_running=True,
            mode="full",
            connected_markets=0,
            wallets_monitored=0,
            alerts_today=0,
            last_event_at=None,
            uptime_seconds=None,
        )

    # Check if we have in-process scanner
    global _scanner_service
    if _scanner_service is not None and _scanner_service.is_running:
        stats = _scanner_service.stats
        uptime = None
        if stats.started_at:
            uptime = (datetime.utcnow() - stats.started_at).total_seconds()

        return ScannerStatusResponse(
            is_running=True,
            mode=stats.mode.value,
            connected_markets=0,
            wallets_monitored=stats.insider_wallets_scored,
            alerts_today=stats.alerts_generated,
            last_event_at=None,
            uptime_seconds=uptime,
        )

    # No scanner running
    return ScannerStatusResponse(
        is_running=False,
        mode="stopped",
        connected_markets=0,
        wallets_monitored=0,
        alerts_today=0,
        last_event_at=None,
        uptime_seconds=None,
    )


class ScannerStartRequest(BaseModel):
    """Request to start the scanner."""
    mode: str = Field("full", pattern="^(full|insider_only|sybil_only)$")


@router.post("/control/start")
async def start_scanner(
    request: Optional[ScannerStartRequest] = None,
    db=Depends(get_session),
):
    """Start the insider scanner.

    Controls the insider-scanner systemd service.
    """
    import subprocess

    # Check if already running
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "insider-scanner"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip() == "active":
            return {
                "status": "already_running",
                "mode": "full",
                "message": "Scanner service is already running",
            }
    except Exception as e:
        logger.warning("systemctl_check_failed", error=str(e))

    # Start the systemd service
    try:
        result = subprocess.run(
            ["systemctl", "start", "insider-scanner"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info("insider_scanner_started_via_systemd")
            return {
                "status": "started",
                "mode": "full",
                "message": "Scanner service started",
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to start scanner: {result.stderr}",
            }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Failed to start scanner: {str(e)}",
        }


@router.post("/control/stop")
async def stop_scanner():
    """Stop the insider scanner.

    Controls the insider-scanner systemd service.
    """
    import subprocess

    # Check if running
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "insider-scanner"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip() != "active":
            return {
                "status": "not_running",
                "message": "Scanner service is not running",
            }
    except Exception as e:
        logger.warning("systemctl_check_failed", error=str(e))

    # Stop the systemd service
    try:
        result = subprocess.run(
            ["systemctl", "stop", "insider-scanner"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info("insider_scanner_stopped_via_systemd")
            return {
                "status": "stopped",
                "message": "Scanner service stopped",
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to stop scanner: {result.stderr}",
            }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Failed to stop scanner: {str(e)}",
        }


@router.get("/control/stats")
async def get_scanner_runtime_stats():
    """Get detailed runtime statistics from the scanner."""
    global _scanner_service

    if _scanner_service is None:
        return {
            "running": False,
            "message": "Scanner not initialized",
        }

    stats = _scanner_service.get_stats()

    # Add monitor stats if available
    if _scanner_service.monitor:
        stats["monitor"] = _scanner_service.monitor.stats.to_dict()

    # Add blockchain monitor stats if available (for automatic sybil detection)
    if _scanner_service.blockchain_monitor:
        stats["blockchain_monitor"] = _scanner_service.blockchain_monitor.get_stats()

    return stats


# =============================================================================
# Statistics
# =============================================================================


@router.get("/stats", response_model=ScannerStats)
async def get_scanner_stats(db=Depends(get_session)):
    """Get scanner statistics."""
    from ...insider_scanner.models import (
        FlaggedWallet,
        FlaggedFundingSource,
        InsiderCluster,
        InsiderPriority,
        DetectionRecord,
    )

    # Count wallets by priority
    critical = db.query(FlaggedWallet).filter(
        FlaggedWallet.priority == InsiderPriority.CRITICAL
    ).count()
    high = db.query(FlaggedWallet).filter(
        FlaggedWallet.priority == InsiderPriority.HIGH
    ).count()
    medium = db.query(FlaggedWallet).filter(
        FlaggedWallet.priority == InsiderPriority.MEDIUM
    ).count()
    low = db.query(FlaggedWallet).filter(
        FlaggedWallet.priority == InsiderPriority.LOW
    ).count()

    total = db.query(FlaggedWallet).count()
    clusters = db.query(InsiderCluster).count()
    funding_sources = db.query(FlaggedFundingSource).count()

    # Today's detections
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    detections_today = db.query(DetectionRecord).filter(
        DetectionRecord.detected_at >= today
    ).count()

    return ScannerStats(
        total_wallets_flagged=total,
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        low_count=low,
        clusters_detected=clusters,
        funding_sources_flagged=funding_sources,
        detections_today=detections_today,
        alerts_sent_today=0,  # Would query alert history
    )


# =============================================================================
# Scoring
# =============================================================================


class ManualScoreRequest(BaseModel):
    """Request for manual wallet scoring."""
    positions: Optional[List[dict]] = None
    account_age_days: Optional[int] = None
    transaction_count: Optional[int] = None
    market_category: Optional[str] = None
    event_hours_away: Optional[float] = None


@router.post("/score/{wallet_address}")
async def score_wallet(
    wallet_address: str,
    request: Optional[ManualScoreRequest] = None,
    db=Depends(get_session),
):
    """Score a wallet for insider behavior.

    If the scanner is running, uses the live scorer.
    Otherwise, creates a temporary scorer for the request.

    You can optionally provide:
    - positions: List of position data (market_id, side, size_usd, etc.)
    - account_age_days: Override account age
    - transaction_count: Override transaction count
    - market_category: One of military_conflict, policy_regulatory, election, corporate, social_culture, crypto, sports
    - event_hours_away: Hours until event resolution
    """
    from ...insider_scanner.scoring import InsiderScorer, MarketCategory

    global _scanner_service

    # Use scanner's scorer if available, otherwise create one
    if _scanner_service and _scanner_service.scorer:
        scorer = _scanner_service.scorer
    else:
        scorer = InsiderScorer()

    # Parse market category if provided
    market_cat = None
    if request and request.market_category:
        try:
            market_cat = MarketCategory(request.market_category)
        except ValueError:
            pass

    # Run scoring
    result = scorer.score_wallet(
        wallet_address=wallet_address,
        account_age_days=request.account_age_days if request else None,
        transaction_count=request.transaction_count if request else None,
        positions=request.positions if request else None,
        market_category=market_cat,
        event_hours_away=request.event_hours_away if request else None,
    )

    return {
        "wallet_address": wallet_address.lower(),
        "score": result.score,
        "priority": result.priority,
        "confidence_low": result.confidence_low,
        "confidence_high": result.confidence_high,
        "dimensions": result.dimensions,
        "signal_count": result.signal_count,
        "active_dimensions": result.active_dimensions,
        "signals": [
            {
                "name": s.name,
                "category": s.category,
                "weight": s.weight,
                "raw_value": str(s.raw_value) if s.raw_value else None,
            }
            for s in result.signals
        ],
        "downgraded": result.downgraded,
        "downgrade_reason": result.downgrade_reason,
    }


# =============================================================================
# Funding Sources
# =============================================================================


@router.get("/funding-sources")
async def list_funding_sources(
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_session),
):
    """List flagged funding sources."""
    from ...insider_scanner.models import FlaggedFundingSource

    sources = (
        db.query(FlaggedFundingSource)
        .order_by(FlaggedFundingSource.associated_wallet_count.desc())
        .limit(limit)
        .all()
    )

    return [s.to_dict() for s in sources]


@router.post("/funding-sources", status_code=201)
async def add_funding_source(
    funding_address: str = Query(..., pattern="^0x[a-fA-F0-9]{40}$"),
    reason: Optional[str] = Query(None),
    risk_level: str = Query("high", pattern="^(low|medium|high|critical)$"),
    db=Depends(get_session),
):
    """Manually flag a funding source."""
    from ...insider_scanner.models import FlaggedFundingSource

    existing = (
        db.query(FlaggedFundingSource)
        .filter(FlaggedFundingSource.funding_address == funding_address.lower())
        .first()
    )

    if existing:
        raise HTTPException(status_code=400, detail="Funding source already flagged")

    source = FlaggedFundingSource(
        funding_address=funding_address.lower(),
        risk_level=risk_level,
        reason=reason,
        manually_flagged=True,
    )

    db.add(source)
    db.commit()
    db.refresh(source)

    logger.info("funding_source_flagged", address=funding_address, risk_level=risk_level)

    return source.to_dict()


# =============================================================================
# Clusters
# =============================================================================


@router.get("/clusters")
async def list_clusters(
    limit: int = Query(50, ge=1, le=500),
    db=Depends(get_session),
):
    """List detected wallet clusters."""
    from ...insider_scanner.models import InsiderCluster

    clusters = (
        db.query(InsiderCluster)
        .order_by(InsiderCluster.total_position_usd.desc())
        .limit(limit)
        .all()
    )

    return [c.to_dict() for c in clusters]


@router.get("/clusters/{cluster_id}")
async def get_cluster_detail(cluster_id: int, db=Depends(get_session)):
    """Get cluster details including member wallets."""
    from ...insider_scanner.models import InsiderCluster, FlaggedWallet

    cluster = db.query(InsiderCluster).filter(InsiderCluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    wallets = db.query(FlaggedWallet).filter(
        FlaggedWallet.cluster_id == cluster_id
    ).all()

    return {
        "cluster": cluster.to_dict(),
        "wallets": [w.to_dict() for w in wallets],
    }


# =============================================================================
# Background Tasks
# =============================================================================


async def _score_wallet_background(wallet_id: int):
    """Background task to score a wallet."""
    # Would implement full scoring with API calls
    logger.info("background_scoring_started", wallet_id=wallet_id)


# =============================================================================
# Sybil Detection (Profitable Trader New Accounts)
# =============================================================================

# Default data directory where extract_funding_sources.py writes output
# Uses project root relative path to work across different deployments
SYBIL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Default file paths for sybil detection data
SYBIL_DEFAULT_PATHS = {
    "profitable_wallets": SYBIL_DATA_DIR / "profitable_wallets_full.json",
    "funding_sources": SYBIL_DATA_DIR / "funding_sources.json",
    "withdrawal_destinations": SYBIL_DATA_DIR / "withdrawal_destinations.json",
    "fund_flow_graph": SYBIL_DATA_DIR / "fund_flow_graph.json",
}

# Checkpoint file for in-progress extractions
SYBIL_CHECKPOINT_PATH = SYBIL_DATA_DIR / "extraction_checkpoint.json"


class SybilImportRequest(BaseModel):
    """Request to import sybil detection data."""
    file_type: str = Field(..., description="Type of file: profitable_wallets, funding_sources, withdrawal_destinations, fund_flow_graph")
    file_path: Optional[str] = Field(None, description="Path to the JSON file to import. If not provided, uses default data directory.")


class SybilCheckRequest(BaseModel):
    """Request to check if a wallet/funding is sybil."""
    new_wallet_address: str = Field(..., pattern="^0x[a-fA-F0-9]{40}$")
    funding_address: str = Field(..., pattern="^0x[a-fA-F0-9]{40}$")
    funding_amount_matic: Optional[float] = None
    funding_tx_hash: Optional[str] = None


@router.get("/sybil/stats")
async def get_sybil_stats(db=Depends(get_session)):
    """Get statistics about the sybil detection database.

    Returns comprehensive statistics including:
    - Raw database counts (total records in each table)
    - Exclusion breakdown (what's filtered out and why)
    - Indexed counts (what's actually used for sybil matching)

    Exclusion categories:
    - always_excluded: Hardcoded exchanges, null addresses, bridges
    - high_volume_excluded: Dynamically excluded based on connection threshold
    """
    from ...insider_scanner.models import (
        KnownProfitableTrader,
        KnownFundingSource,
        KnownWithdrawalDest,
        FundFlowEdge,
        SybilDetection,
        FlaggedWallet,
        FlagType,
    )
    from ...insider_scanner.sybil_detector import SybilDetector

    # Get raw database counts
    raw_stats = {
        "known_profitable_traders": db.query(KnownProfitableTrader).count(),
        "known_funding_sources": db.query(KnownFundingSource).count(),
        "known_withdrawal_dests": db.query(KnownWithdrawalDest).count(),
        "fund_flow_edges": db.query(FundFlowEdge).count(),
        "sybil_detections": db.query(SybilDetection).count(),
        "sybil_flagged_wallets": db.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.SYBIL
        ).count(),
        "insider_flagged_wallets": db.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.INSIDER
        ).count(),
    }

    # Get exclusion stats from detector
    detector = SybilDetector(make_session_factory(db))
    detector.rebuild_index()
    exclusion_stats = detector.get_exclusion_stats()

    # Extract nested values from exclusion_stats structure
    indexed = exclusion_stats.get("indexed", {})
    always_excluded = exclusion_stats.get("always_excluded", {})
    dynamic_excluded = exclusion_stats.get("dynamic_excluded", {})

    return {
        **raw_stats,
        "exclusion_breakdown": exclusion_stats,
        "indexed_funding_sources": indexed.get("funding_sources", 0),
        "indexed_withdrawal_dests": indexed.get("withdrawal_dests", 0),
        "total_excluded": exclusion_stats.get("total_excluded", 0),
        "always_excluded": always_excluded.get("total", 0),
        "high_volume_excluded": dynamic_excluded.get("high_volume_sources", 0),
    }


@router.post("/sybil/import")
async def import_sybil_data(request: SybilImportRequest, db=Depends(get_session)):
    """Import sybil detection data from JSON files.

    Accepts:
    - profitable_wallets: profitable_wallets_full.json
    - funding_sources: funding_sources.json
    - withdrawal_destinations: withdrawal_destinations.json
    - fund_flow_graph: fund_flow_graph.json (edges only)

    If file_path is not provided, uses default paths from data directory.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    # Use default path if not provided
    file_path = request.file_path
    if not file_path:
        if request.file_type not in SYBIL_DEFAULT_PATHS:
            raise HTTPException(status_code=400, detail=f"Invalid file_type: {request.file_type}")
        file_path = str(SYBIL_DEFAULT_PATHS[request.file_type])

    detector = SybilDetector(make_session_factory(db))

    try:
        if request.file_type == "profitable_wallets":
            stats = detector.import_profitable_wallets(file_path)
        elif request.file_type == "funding_sources":
            stats = detector.import_funding_sources(file_path)
        elif request.file_type == "withdrawal_destinations":
            stats = detector.import_withdrawal_destinations(file_path)
        elif request.file_type == "fund_flow_graph":
            stats = detector.import_fund_flow_edges(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid file_type: {request.file_type}")

        # Rebuild indices after import
        index_stats = detector.rebuild_index()

        logger.info("sybil_data_imported", file_type=request.file_type, stats=stats)

        return {
            "success": True,
            "file_type": request.file_type,
            "import_stats": stats,
            "index_stats": index_stats,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("sybil_import_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/sybil/import-checkpoint")
async def import_from_checkpoint(db=Depends(get_session)):
    """Import sybil data from in-progress extraction checkpoint.

    Use this when extract_funding_sources.py is still running.
    The checkpoint file contains processed wallets even before
    the final JSON files are generated.

    Imports directly from extraction_checkpoint.json which is
    written every 50 wallets during the extraction process.
    """
    from ...insider_scanner.models import KnownProfitableTrader, KnownFundingSource, KnownWithdrawalDest
    import json
    from decimal import Decimal
    from collections import defaultdict

    if not SYBIL_CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint file not found: {SYBIL_CHECKPOINT_PATH}. Run extract_funding_sources.py first."
        )

    with open(SYBIL_CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)

    results = checkpoint.get("results", [])
    if not results:
        return {
            "success": False,
            "message": "Checkpoint exists but has no results yet",
            "checkpoint_timestamp": checkpoint.get("timestamp"),
        }

    stats = {
        "traders_imported": 0,
        "traders_updated": 0,
        "funding_sources_imported": 0,
        "withdrawal_dests_imported": 0,
    }

    # Build funding source and withdrawal dest aggregations
    funding_sources = defaultdict(lambda: {"funded_wallets": [], "total_profit": Decimal("0")})
    withdrawal_dests = defaultdict(lambda: {"received_from": [], "total_profit": Decimal("0")})

    # Import traders and build aggregations
    for wallet_data in results:
        wallet_addr = wallet_data.get("wallet", "").lower()
        if not wallet_addr:
            continue

        profit = Decimal(str(wallet_data.get("profit_usd", 0)))
        funding_source = (wallet_data.get("funding_source") or "").lower()
        primary_wd = (wallet_data.get("primary_withdrawal_dest") or "").lower()

        # Upsert trader
        existing = db.query(KnownProfitableTrader).filter(
            KnownProfitableTrader.wallet_address == wallet_addr
        ).first()

        if existing:
            existing.profit_usd = profit
            existing.funding_source = funding_source if funding_source else None
            existing.funding_source_type = wallet_data.get("funding_source_type")
            existing.primary_withdrawal_dest = primary_wd if primary_wd else None
            stats["traders_updated"] += 1
        else:
            trader = KnownProfitableTrader(
                wallet_address=wallet_addr,
                profit_usd=profit,
                funding_source=funding_source if funding_source else None,
                funding_source_type=wallet_data.get("funding_source_type"),
                funding_amount_matic=Decimal(str(wallet_data.get("funding_amount_matic", 0))) if wallet_data.get("funding_amount_matic") else None,
                primary_withdrawal_dest=primary_wd if primary_wd else None,
                total_withdrawn_matic=Decimal(str(wallet_data.get("total_withdrawn_matic", 0))) if wallet_data.get("total_withdrawn_matic") else None,
                data_source="extraction_checkpoint.json",
            )
            db.add(trader)
            stats["traders_imported"] += 1

        # Aggregate funding sources
        if funding_source:
            funding_sources[funding_source]["funded_wallets"].append(wallet_addr)
            funding_sources[funding_source]["total_profit"] += profit

        # Aggregate withdrawal destinations
        if primary_wd:
            withdrawal_dests[primary_wd]["received_from"].append(wallet_addr)
            withdrawal_dests[primary_wd]["total_profit"] += profit

    db.commit()

    # Import funding sources
    for addr, data in funding_sources.items():
        existing = db.query(KnownFundingSource).filter(
            KnownFundingSource.address == addr
        ).first()

        if not existing:
            fs = KnownFundingSource(
                address=addr,
                funded_trader_wallets=data["funded_wallets"],
                funded_trader_count=len(data["funded_wallets"]),
                total_profit_funded=data["total_profit"],
            )
            db.add(fs)
            stats["funding_sources_imported"] += 1

    # Import withdrawal destinations
    for addr, data in withdrawal_dests.items():
        existing = db.query(KnownWithdrawalDest).filter(
            KnownWithdrawalDest.address == addr
        ).first()

        if not existing:
            wd = KnownWithdrawalDest(
                address=addr,
                received_from_traders=data["received_from"],
                received_from_count=len(data["received_from"]),
                total_profit_source=data["total_profit"],
            )
            db.add(wd)
            stats["withdrawal_dests_imported"] += 1

    db.commit()

    # Rebuild detector index
    from ...insider_scanner.sybil_detector import SybilDetector
    detector = SybilDetector(make_session_factory(db))
    index_stats = detector.rebuild_index()

    return {
        "success": True,
        "checkpoint_timestamp": checkpoint.get("timestamp"),
        "checkpoint_wallets": len(checkpoint.get("processed", [])),
        "import_stats": stats,
        "index_stats": index_stats,
        "message": f"Imported {stats['traders_imported'] + stats['traders_updated']} traders from in-progress checkpoint",
    }


@router.post("/sybil/import-all")
async def import_all_sybil_data(db=Depends(get_session)):
    """Import all sybil detection data from default data directory.

    Imports in order:
    1. profitable_wallets_full.json
    2. funding_sources.json
    3. withdrawal_destinations.json
    4. fund_flow_graph.json

    Uses files from the project's data/ directory
    which are produced by extract_funding_sources.py

    TIP: If extraction is still in progress, use /sybil/import-checkpoint
    to import from the checkpoint file instead.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))
    results = {
        "success": True,
        "data_directory": str(SYBIL_DATA_DIR),
        "files_imported": [],
        "files_missing": [],
        "import_stats": {},
    }

    # Import order matters - traders first, then sources, then destinations
    import_order = [
        ("profitable_wallets", "profitable_wallets_full.json"),
        ("funding_sources", "funding_sources.json"),
        ("withdrawal_destinations", "withdrawal_destinations.json"),
        ("fund_flow_graph", "fund_flow_graph.json"),
    ]

    for file_type, filename in import_order:
        file_path = SYBIL_DEFAULT_PATHS[file_type]

        if not file_path.exists():
            results["files_missing"].append(filename)
            logger.warning("sybil_file_missing", file=filename)
            continue

        try:
            if file_type == "profitable_wallets":
                stats = detector.import_profitable_wallets(str(file_path))
            elif file_type == "funding_sources":
                stats = detector.import_funding_sources(str(file_path))
            elif file_type == "withdrawal_destinations":
                stats = detector.import_withdrawal_destinations(str(file_path))
            elif file_type == "fund_flow_graph":
                stats = detector.import_fund_flow_edges(str(file_path))

            results["files_imported"].append(filename)
            results["import_stats"][file_type] = stats
            logger.info("sybil_file_imported", file=filename, stats=stats)

        except Exception as e:
            logger.error("sybil_file_import_failed", file=filename, error=str(e))
            results["import_stats"][file_type] = {"error": str(e)}

    # Rebuild indices after all imports
    index_stats = detector.rebuild_index()
    results["index_stats"] = index_stats

    # Get final counts
    results["total_stats"] = detector.get_stats()

    if not results["files_imported"]:
        results["success"] = False
        results["message"] = "No data files found. Run extract_funding_sources.py first."

    return results


@router.post("/sybil/check")
async def check_sybil_wallet(request: SybilCheckRequest, db=Depends(get_session)):
    """Check if a new wallet is a sybil (funded by known profitable trader).

    Returns match details if found, including the linked trader and chain path.

    Validates that the new wallet address is not a test/placeholder address.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))
    detector.rebuild_index()  # Ensure indices are current

    # Validate that the new wallet is not a test/null address
    new_wallet_lower = request.new_wallet_address.lower()
    is_excluded, reason = detector.is_excluded(new_wallet_lower)
    if is_excluded:
        logger.warning(
            "sybil_check_rejected_test_address",
            wallet=new_wallet_lower,
            reason=reason,
        )
        return {
            "is_sybil": False,
            "rejected": True,
            "rejection_reason": reason,
            "message": f"New wallet address rejected: {reason}. This appears to be a test or invalid address.",
        }

    match = detector.check_funding_source(request.funding_address)

    if match.matched:
        # Flag the wallet
        detector.flag_sybil_wallet(
            new_wallet=request.new_wallet_address,
            funding_address=request.funding_address,
            funding_amount_matic=Decimal(str(request.funding_amount_matic)) if request.funding_amount_matic else None,
            funding_tx_hash=request.funding_tx_hash,
        )

        # Query for flagged wallet ID within current session
        from ...insider_scanner.models import FlaggedWallet
        flagged_wallet = db.query(FlaggedWallet).filter(
            FlaggedWallet.wallet_address == request.new_wallet_address.lower()
        ).first()

        return {
            "is_sybil": True,
            "match_type": match.match_type,
            "match_address": match.match_address,
            "linked_trader_wallet": match.linked_trader_wallet,
            "linked_trader_id": match.linked_trader_id,
            "linked_trader_profit": float(match.linked_trader_profit) if match.linked_trader_profit else None,
            "chain_path": match.chain_path,
            "confidence": match.confidence,
            "flagged_wallet_id": flagged_wallet.id if flagged_wallet else None,
        }

    return {
        "is_sybil": False,
        "match_type": None,
        "linked_trader_wallet": None,
    }


@router.get("/sybil/detections")
async def list_sybil_detections(
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_session),
):
    """List recent sybil detection events."""
    from ...insider_scanner.models import SybilDetection

    detections = (
        db.query(SybilDetection)
        .order_by(SybilDetection.detected_at.desc())
        .limit(limit)
        .all()
    )

    return [d.to_dict() for d in detections]


@router.get("/sybil/profitable-traders")
async def list_known_profitable_traders(
    min_profit: Optional[float] = Query(None, description="Minimum profit in USD"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_session),
):
    """List known profitable traders from imported data."""
    from ...insider_scanner.models import KnownProfitableTrader

    query = db.query(KnownProfitableTrader)

    if min_profit is not None:
        query = query.filter(KnownProfitableTrader.profit_usd >= min_profit)

    traders = (
        query.order_by(KnownProfitableTrader.profit_usd.desc())
        .limit(limit)
        .all()
    )

    return [t.to_dict() for t in traders]


@router.get("/sybil/funding-sources")
async def list_known_funding_sources(
    is_exchange: Optional[bool] = Query(None, description="Filter by exchange status"),
    min_traders: Optional[int] = Query(None, description="Minimum traders funded"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_session),
):
    """List known funding sources from imported data."""
    from ...insider_scanner.models import KnownFundingSource

    query = db.query(KnownFundingSource)

    if is_exchange is not None:
        query = query.filter(KnownFundingSource.is_exchange == is_exchange)

    if min_traders is not None:
        query = query.filter(KnownFundingSource.funded_trader_count >= min_traders)

    sources = (
        query.order_by(KnownFundingSource.total_profit_funded.desc())
        .limit(limit)
        .all()
    )

    return [s.to_dict() for s in sources]


@router.get("/sybil/withdrawal-destinations")
async def list_known_withdrawal_destinations(
    is_bridge: Optional[bool] = Query(None, description="Filter by bridge wallet status"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_session),
):
    """List known withdrawal destinations from imported data.

    Bridge wallets (is_bridge=True) are addresses that received withdrawals
    from profitable traders AND then funded new traders - key for chain detection.
    """
    from ...insider_scanner.models import KnownWithdrawalDest

    query = db.query(KnownWithdrawalDest)

    if is_bridge is not None:
        query = query.filter(KnownWithdrawalDest.is_bridge_wallet == is_bridge)

    dests = (
        query.order_by(KnownWithdrawalDest.total_profit_source.desc())
        .limit(limit)
        .all()
    )

    return [d.to_dict() for d in dests]


@router.post("/sybil/rebuild-index")
async def rebuild_sybil_index(db=Depends(get_session)):
    """Rebuild the in-memory sybil detection indices.

    Call this after importing new data to update the lookup indices.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))
    stats = detector.rebuild_index()

    logger.info("sybil_index_rebuilt", stats=stats)

    return {
        "success": True,
        "indexed": stats,
    }


@router.get("/sybil/high-volume-sources")
async def get_high_volume_sources(db=Depends(get_session)):
    """Get list of high-volume funding sources excluded from sybil detection.

    High-volume sources are addresses that funded more than the threshold
    number of traders (default: 10). These are typically:
    - Exchanges (Binance, Coinbase, etc.)
    - OTC desks
    - Market makers
    - Bridge protocols

    These are excluded because matching them would create many false positives.
    Personal wallet sybil detection focuses on sources that fund only a few traders.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))
    detector.rebuild_index()

    high_volume = detector.get_high_volume_sources()

    return {
        "threshold": detector.max_funded_traders_threshold,
        "count": len(high_volume),
        "sources": high_volume,
        "message": f"Sources funding >{detector.max_funded_traders_threshold} traders are excluded from sybil detection",
    }


@router.get("/sybil/excluded-sources")
async def get_excluded_sources(db=Depends(get_session)):
    """Get complete breakdown of ALL excluded sources from sybil detection.

    This endpoint shows both always-excluded addresses and dynamically excluded
    high-volume sources, with reasons for each exclusion.

    Categories:
    - always_excluded: Hardcoded lists that never change
      - exchanges: Known exchange hot wallets (Binance, Bybit, etc.)
      - null_addresses: Zero address, dead address
      - bridges: Known bridge protocols (Polygon, LayerSwap, etc.)
    - high_volume: Dynamically excluded based on connection threshold
      - Sources with > threshold total connections (funding + withdrawal)

    The scanner only matches against sources NOT in this list.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))
    detector.rebuild_index()

    exclusion_stats = detector.get_exclusion_stats()

    # Build detailed breakdown
    always_excluded = {
        "exchanges": list(detector.KNOWN_EXCHANGE_ADDRESSES),
        "null_addresses": list(detector.NULL_ADDRESSES),
        "bridges": list(detector.BRIDGE_ADDRESSES),
    }

    # Get high-volume with details
    high_volume_sources = detector.get_high_volume_sources()

    return {
        "summary": {
            "total_excluded": exclusion_stats.get("total_excluded", 0),
            "always_excluded_count": exclusion_stats.get("always_excluded", 0),
            "high_volume_count": exclusion_stats.get("high_volume_excluded", 0),
            "indexed_funding_sources": exclusion_stats.get("indexed_funding_sources", 0),
            "indexed_withdrawal_dests": exclusion_stats.get("indexed_withdrawal_dests", 0),
        },
        "always_excluded": always_excluded,
        "always_excluded_total": len(detector.KNOWN_EXCHANGE_ADDRESSES) + len(detector.NULL_ADDRESSES) + len(detector.BRIDGE_ADDRESSES),
        "high_volume_threshold": detector.max_funded_traders_threshold,
        "high_volume_sources": high_volume_sources,
        "message": "Sources in always_excluded are hardcoded (exchanges, nulls, bridges). High-volume sources are dynamically excluded based on connection count threshold.",
    }


@router.get("/sybil/check-exclusion/{address}")
async def check_address_exclusion(address: str, db=Depends(get_session)):
    """Check if a specific address is excluded from sybil detection and why.

    Returns detailed information about whether the address is excluded,
    the reason for exclusion, and connection counts if applicable.
    """
    from ...insider_scanner.sybil_detector import SybilDetector
    from ...insider_scanner.models import KnownFundingSource, KnownWithdrawalDest

    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address format")

    detector = SybilDetector(make_session_factory(db))
    detector.rebuild_index()

    addr_lower = address.lower()

    # Check exclusion status with reason
    is_excluded, reason = detector.is_excluded(addr_lower)

    # Get connection details from database
    fs = db.query(KnownFundingSource).filter(
        KnownFundingSource.address == addr_lower
    ).first()

    wd = db.query(KnownWithdrawalDest).filter(
        KnownWithdrawalDest.address == addr_lower
    ).first()

    funded_count = fs.funded_trader_count if fs else 0
    received_count = wd.received_from_count if wd else 0
    total_connections = funded_count + received_count

    return {
        "address": addr_lower,
        "is_excluded": is_excluded,
        "exclusion_reason": reason,
        "funded_trader_count": funded_count,
        "received_from_count": received_count,
        "total_connections": total_connections,
        "high_volume_threshold": detector.max_funded_traders_threshold,
        "in_index": addr_lower in detector._funding_source_index or addr_lower in detector._withdrawal_dest_index,
    }


@router.get("/sybil/threshold")
async def get_sybil_threshold(db=Depends(get_session)):
    """Get the current max funded traders threshold.

    Sources that have funded more than this many traders are considered
    high-volume (exchanges, OTC, market makers) and excluded from sybil detection.
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))

    return {
        "max_funded_traders_threshold": detector.max_funded_traders_threshold,
        "description": "Sources funding more than this many traders are excluded as likely exchanges/OTC",
    }


class SybilThresholdUpdate(BaseModel):
    """Request to update the max funded traders threshold."""
    threshold: int = Field(..., ge=1, le=1000, description="New threshold value")


@router.post("/sybil/threshold")
async def update_sybil_threshold(request: SybilThresholdUpdate, db=Depends(get_session)):
    """Update the max funded traders threshold and rebuild the index.

    Sources that have funded more than this many traders will be excluded
    from sybil detection as likely exchanges, OTC desks, or market makers.

    Recommended values:
    - 5: Very strict (excludes more sources, fewer sybil matches)
    - 10: Balanced (default, good for most cases)
    - 20: Permissive (only excludes largest exchanges)
    """
    from ...insider_scanner.sybil_detector import SybilDetector

    detector = SybilDetector(make_session_factory(db))
    old_threshold = detector.max_funded_traders_threshold

    stats = detector.set_max_funded_traders_threshold(request.threshold)

    logger.info(
        "sybil_threshold_updated",
        old_threshold=old_threshold,
        new_threshold=request.threshold,
        high_volume_excluded=stats.get("high_volume_excluded", 0),
    )

    return {
        "success": True,
        "old_threshold": old_threshold,
        "new_threshold": request.threshold,
        "index_stats": stats,
        "message": f"Updated threshold from {old_threshold} to {request.threshold}",
    }


@router.get("/sybil/check-high-volume/{address}")
async def check_if_high_volume(address: str, db=Depends(get_session)):
    """Check if a specific address is a high-volume source.

    High-volume sources are excluded from sybil detection.
    """
    from ...insider_scanner.sybil_detector import SybilDetector
    from ...insider_scanner.models import KnownFundingSource, KnownWithdrawalDest

    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address format")

    detector = SybilDetector(make_session_factory(db))
    detector.rebuild_index()

    addr_lower = address.lower()
    is_high_volume = detector.is_high_volume_source(addr_lower)

    # Get details from database
    fs = db.query(KnownFundingSource).filter(
        KnownFundingSource.address == addr_lower
    ).first()

    wd = db.query(KnownWithdrawalDest).filter(
        KnownWithdrawalDest.address == addr_lower
    ).first()

    funded_count = fs.funded_trader_count if fs else 0
    received_count = wd.received_from_count if wd else 0

    return {
        "address": addr_lower,
        "is_high_volume": is_high_volume,
        "threshold": detector.max_funded_traders_threshold,
        "funded_trader_count": funded_count,
        "received_from_count": received_count,
        "total_connections": max(funded_count, received_count),
        "reason": "Exceeds threshold - likely exchange/OTC/market maker" if is_high_volume else "Below threshold - personal wallet",
    }


# =============================================================================
# Documented Insiders Management
# =============================================================================

DOCUMENTED_INSIDERS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "documented_insiders.json"


def _load_documented_insiders() -> dict:
    """Load documented insiders from JSON file."""
    if not DOCUMENTED_INSIDERS_PATH.exists():
        return {"metadata": {}, "insiders": [], "funding_sources_to_monitor": [], "scanner_integration": {}}
    with open(DOCUMENTED_INSIDERS_PATH) as f:
        return json.load(f)


def _save_documented_insiders(data: dict):
    """Save documented insiders to JSON file."""
    data["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(DOCUMENTED_INSIDERS_PATH, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/documented-insiders")
async def list_documented_insiders():
    """List all documented insider accounts from historical research."""
    data = _load_documented_insiders()
    return {
        "metadata": data.get("metadata", {}),
        "total": len(data.get("insiders", [])),
        "insiders": data.get("insiders", []),
        "funding_sources": data.get("funding_sources_to_monitor", []),
        "scanner_integration": data.get("scanner_integration", {}),
    }


@router.get("/documented-insiders/{insider_id}")
async def get_documented_insider(insider_id: str):
    """Get a specific documented insider by ID."""
    data = _load_documented_insiders()
    for insider in data.get("insiders", []):
        if insider.get("id") == insider_id:
            return insider
    raise HTTPException(status_code=404, detail=f"Insider {insider_id} not found")


class DocumentedInsiderCreate(BaseModel):
    """Create a new documented insider entry."""
    wallet_address: str = Field(..., pattern="^0x[a-fA-F0-9]{40}$")
    username: Optional[str] = None
    category: str = Field(..., description="Category: military, awards, corporate, etc.")
    subcategory: Optional[str] = None
    documented_profit_usd: Optional[float] = None
    documented_win_rate: Optional[float] = None
    signals: Optional[List[str]] = None
    notes: Optional[str] = None
    sources: Optional[List[str]] = None


@router.post("/documented-insiders", status_code=201)
async def add_documented_insider(request: DocumentedInsiderCreate):
    """Add a new documented insider to the database.

    This is called when the scanner flags a wallet and the user confirms it as an insider.
    The wallet's funding sources are then monitored for future alt account detection.
    """
    data = _load_documented_insiders()

    # Generate ID from username or wallet
    insider_id = request.username or request.wallet_address[:12]
    insider_id = insider_id.lower().replace("-", "_").replace(" ", "_")

    # Check if already exists
    for existing in data.get("insiders", []):
        if existing.get("wallet_address", "").lower() == request.wallet_address.lower():
            raise HTTPException(status_code=400, detail="Insider already documented")

    new_insider = {
        "id": insider_id,
        "wallet_address": request.wallet_address.lower(),
        "wallet_verified": True,
        "username": request.username,
        "category": request.category,
        "subcategory": request.subcategory,
        "documented_profit_usd": request.documented_profit_usd,
        "documented_win_rate": request.documented_win_rate,
        "signals": request.signals or [],
        "notes": request.notes,
        "sources": request.sources or [],
        "added_at": datetime.utcnow().isoformat() + "Z",
        "added_by": "scanner_auto_probe",
    }

    data["insiders"].append(new_insider)

    # Update scanner integration list
    if "scanner_integration" not in data:
        data["scanner_integration"] = {"known_insider_wallets": [], "known_insider_usernames": []}

    if request.wallet_address.lower() not in data["scanner_integration"].get("known_insider_wallets", []):
        data["scanner_integration"]["known_insider_wallets"].append(request.wallet_address.lower())

    if request.username and request.username not in data["scanner_integration"].get("known_insider_usernames", []):
        data["scanner_integration"]["known_insider_usernames"].append(request.username)

    _save_documented_insiders(data)

    logger.info("documented_insider_added", wallet=request.wallet_address, category=request.category)

    return new_insider


class InsiderFundingInfo(BaseModel):
    """Funding information for a documented insider."""
    funding_source: Optional[str] = None
    funding_amount: Optional[float] = None
    funding_tx: Optional[str] = None
    withdrawal_dest: Optional[str] = None
    withdrawal_amount: Optional[float] = None
    withdrawal_tx: Optional[str] = None


@router.patch("/documented-insiders/{insider_id}/funding")
async def update_insider_funding(insider_id: str, funding: InsiderFundingInfo):
    """Update funding information for a documented insider.

    This adds the funding source to the monitored list for alt detection.
    """
    data = _load_documented_insiders()

    insider_found = False
    for insider in data.get("insiders", []):
        if insider.get("id") == insider_id:
            insider["funding_info"] = funding.dict(exclude_none=True)
            insider_found = True
            break

    if not insider_found:
        raise HTTPException(status_code=404, detail=f"Insider {insider_id} not found")

    # Add funding source to monitored list
    if funding.funding_source:
        existing_sources = {s.get("address") for s in data.get("funding_sources_to_monitor", [])}
        if funding.funding_source.lower() not in existing_sources:
            data["funding_sources_to_monitor"].append({
                "type": "insider_funding",
                "address": funding.funding_source.lower(),
                "associated_insiders": [insider_id],
                "risk_level": "critical",
                "note": f"Funded documented insider {insider_id}",
            })

    _save_documented_insiders(data)

    logger.info("insider_funding_updated", insider_id=insider_id)

    return {"success": True, "insider_id": insider_id}


@router.post("/auto-probe/{wallet_address}")
async def auto_probe_wallet(
    wallet_address: str,
    background_tasks: BackgroundTasks,
    db=Depends(get_session),
):
    """Auto-probe a flagged wallet and add to documented insiders.

    This endpoint:
    1. Fetches the wallet's full profile from Polymarket
    2. Extracts funding sources via Etherscan
    3. Adds the wallet to documented_insiders.json
    4. Registers funding sources for alt detection

    Use this when a flagged wallet is confirmed as an insider.
    """
    from ...insider_scanner.profile import ProfileFetcher
    from ...insider_scanner.blockchain_monitor import BlockchainMonitor
    from ...insider_scanner.models import FlaggedWallet

    if not wallet_address.startswith("0x") or len(wallet_address) != 42:
        raise HTTPException(status_code=400, detail="Invalid wallet address")

    wallet_lower = wallet_address.lower()

    # Check if already in flagged wallets
    flagged = db.query(FlaggedWallet).filter(
        FlaggedWallet.wallet_address == wallet_lower
    ).first()

    # Get profile data
    profile_data = {}
    try:
        async with ProfileFetcher() as fetcher:
            profile = await fetcher.get_profile(wallet_lower)
            if profile:
                profile_data = {
                    "username": profile.get("name") or profile.get("pseudonym"),
                    "account_created_at": profile.get("createdAt"),
                }
    except Exception as e:
        logger.warning("profile_fetch_failed", wallet=wallet_lower, error=str(e))

    # Extract funding info
    funding_info = {}
    try:
        monitor = BlockchainMonitor()
        funding = await monitor.extract_funding_sources(wallet_lower)
        if funding:
            funding_info = {
                "funding_source": funding.get("funding_source"),
                "funding_amount": funding.get("funding_amount"),
                "funding_tx": funding.get("funding_tx"),
                "withdrawal_dest": funding.get("withdrawal_dest"),
                "withdrawal_amount": funding.get("withdrawal_amount"),
                "withdrawal_tx": funding.get("withdrawal_tx"),
            }
    except Exception as e:
        logger.warning("funding_extraction_failed", wallet=wallet_lower, error=str(e))

    # Build insider entry
    signals = []
    category = "unknown"
    subcategory = None
    documented_profit = None
    documented_win_rate = None

    if flagged:
        signals = [s.get("name", s.get("signal_name", "unknown")) for s in (flagged.signals_json or [])]
        documented_profit = float(flagged.total_position_usd) if flagged.total_position_usd else None
        documented_win_rate = float(flagged.win_rate) if flagged.win_rate else None
        # Try to infer category from signals
        if any("military" in s.lower() for s in signals):
            category = "military"
        elif any("award" in s.lower() or "nobel" in s.lower() for s in signals):
            category = "awards"
        elif any("corporate" in s.lower() or "google" in s.lower() for s in signals):
            category = "corporate"
        elif any("election" in s.lower() for s in signals):
            category = "election"

    # Add to documented insiders
    data = _load_documented_insiders()

    insider_id = (profile_data.get("username") or wallet_lower[:12]).lower().replace("-", "_")

    # Check if already exists
    for existing in data.get("insiders", []):
        if existing.get("wallet_address", "").lower() == wallet_lower:
            # Update existing entry
            existing["funding_info"] = funding_info
            existing["last_probed_at"] = datetime.utcnow().isoformat() + "Z"
            _save_documented_insiders(data)
            return {
                "success": True,
                "action": "updated",
                "insider_id": existing.get("id"),
                "wallet_address": wallet_lower,
                "funding_info": funding_info,
            }

    new_insider = {
        "id": insider_id,
        "wallet_address": wallet_lower,
        "wallet_verified": True,
        "username": profile_data.get("username"),
        "account_created_at": profile_data.get("account_created_at"),
        "category": category,
        "subcategory": subcategory,
        "documented_profit_usd": documented_profit,
        "documented_win_rate": documented_win_rate,
        "signals": signals,
        "funding_info": funding_info,
        "status": "active" if flagged else "unknown",
        "investigation_status": "auto_probed",
        "added_at": datetime.utcnow().isoformat() + "Z",
        "added_by": "scanner_auto_probe",
        "probed_at": datetime.utcnow().isoformat() + "Z",
    }

    data["insiders"].append(new_insider)

    # Update scanner integration
    if "scanner_integration" not in data:
        data["scanner_integration"] = {"known_insider_wallets": [], "known_insider_usernames": []}

    if wallet_lower not in data["scanner_integration"].get("known_insider_wallets", []):
        data["scanner_integration"]["known_insider_wallets"].append(wallet_lower)

    if profile_data.get("username"):
        if profile_data["username"] not in data["scanner_integration"].get("known_insider_usernames", []):
            data["scanner_integration"]["known_insider_usernames"].append(profile_data["username"])

    # Add funding sources to monitor
    if funding_info.get("funding_source"):
        existing_sources = {s.get("address") for s in data.get("funding_sources_to_monitor", [])}
        if funding_info["funding_source"].lower() not in existing_sources:
            data["funding_sources_to_monitor"].append({
                "type": "insider_funding",
                "address": funding_info["funding_source"].lower(),
                "associated_insiders": [insider_id],
                "risk_level": "critical",
                "note": f"Funded documented insider {insider_id}",
            })

    _save_documented_insiders(data)

    # Update flagged wallet status if exists
    if flagged:
        flagged.status = WalletStatus.CONFIRMED
        flagged.notes = (flagged.notes or "") + f"\n[Auto-probed and added to documented insiders at {datetime.utcnow().isoformat()}]"
        db.commit()

    logger.info("wallet_auto_probed", wallet=wallet_lower, insider_id=insider_id)

    return {
        "success": True,
        "action": "created",
        "insider_id": insider_id,
        "wallet_address": wallet_lower,
        "profile": profile_data,
        "funding_info": funding_info,
        "signals": signals,
    }


@router.get("/insider-funding-sources")
async def list_insider_funding_sources():
    """List funding sources associated with documented insiders.

    These are high-risk sources that funded confirmed insider traders.
    New wallets funded from these sources should be flagged immediately.
    """
    data = _load_documented_insiders()

    sources = data.get("funding_sources_to_monitor", [])

    # Also extract from insider funding_info
    insider_sources = []
    for insider in data.get("insiders", []):
        funding = insider.get("funding_info", {})
        if funding.get("funding_source"):
            insider_sources.append({
                "address": funding["funding_source"].lower(),
                "type": "insider_direct",
                "associated_insiders": [insider.get("id")],
                "risk_level": "critical",
                "note": f"Direct funding source for {insider.get('username') or insider.get('id')}",
            })

    # Merge and dedupe
    all_sources = {}
    for src in sources + insider_sources:
        addr = src.get("address", "").lower()
        if addr not in all_sources:
            all_sources[addr] = src
        else:
            # Merge associated insiders
            existing = all_sources[addr].get("associated_insiders", [])
            new_insiders = src.get("associated_insiders", [])
            all_sources[addr]["associated_insiders"] = list(set(existing + new_insiders))

    return {
        "total": len(all_sources),
        "sources": list(all_sources.values()),
    }


@router.post("/check-insider-funding")
async def check_insider_funding(funding_address: str):
    """Check if a funding address is associated with documented insiders.

    This is the insider-specific equivalent of /sybil/check.
    """
    if not funding_address.startswith("0x") or len(funding_address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")

    addr_lower = funding_address.lower()
    data = _load_documented_insiders()

    # Check documented funding sources
    for src in data.get("funding_sources_to_monitor", []):
        if src.get("address", "").lower() == addr_lower:
            return {
                "is_match": True,
                "match_type": "documented_insider_funding",
                "associated_insiders": src.get("associated_insiders", []),
                "risk_level": src.get("risk_level", "critical"),
                "note": src.get("note"),
            }

    # Check insider funding_info
    for insider in data.get("insiders", []):
        funding = insider.get("funding_info", {})
        if funding.get("funding_source", "").lower() == addr_lower:
            return {
                "is_match": True,
                "match_type": "insider_direct_funding",
                "associated_insiders": [insider.get("id")],
                "risk_level": "critical",
                "note": f"Direct funding source for documented insider {insider.get('username') or insider.get('id')}",
                "insider_profit": insider.get("documented_profit_usd"),
            }

        # Also check withdrawal dest (circular funding)
        if funding.get("withdrawal_dest", "").lower() == addr_lower:
            return {
                "is_match": True,
                "match_type": "insider_withdrawal_recycled",
                "associated_insiders": [insider.get("id")],
                "risk_level": "critical",
                "note": f"Withdrawal destination recycled as funding - potential alt of {insider.get('username') or insider.get('id')}",
            }

    return {
        "is_match": False,
        "note": "Address not associated with any documented insiders",
    }
