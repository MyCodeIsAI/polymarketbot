"""REST API routes for PolymarketBot dashboard.

Provides endpoints for:
- Bot status and control
- Account management
- Position viewing
- Trade history
- P&L data
- System health
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...utils.logging import get_logger
from .dependencies import get_db, get_bot_state

logger = get_logger(__name__)

router = APIRouter(tags=["api"])


# =============================================================================
# Pydantic Models
# =============================================================================

class StatusResponse(BaseModel):
    """Bot status response."""

    is_running: bool
    uptime_seconds: Optional[float] = None
    state: str = "stopped"
    health: str = "unknown"

    class Config:
        json_schema_extra = {
            "example": {
                "is_running": True,
                "uptime_seconds": 3600,
                "state": "running",
                "health": "healthy",
            }
        }


class BalanceResponse(BaseModel):
    """Balance information response."""

    total: str
    available: str
    reserved: str
    timestamp: str


class AccountCreate(BaseModel):
    """Account creation request."""

    name: str = Field(..., min_length=1, max_length=100)
    wallet: str = Field(..., pattern="^0x[a-fA-F0-9]{40}$")
    position_ratio: float = Field(0.01, gt=0, le=1)
    max_position_usd: float = Field(500, gt=0)
    slippage_tolerance: float = Field(0.05, gt=0, le=0.5)
    min_position_usd: float = Field(5, ge=0)


class AccountResponse(BaseModel):
    """Account response."""

    id: int
    name: str
    target_wallet: str
    position_ratio: str
    max_position_usd: str
    slippage_tolerance: str
    min_position_usd: str
    enabled: bool
    created_at: Optional[str] = None


class AccountUpdate(BaseModel):
    """Account update request."""

    position_ratio: Optional[float] = Field(None, gt=0, le=1)
    max_position_usd: Optional[float] = Field(None, gt=0)
    slippage_tolerance: Optional[float] = Field(None, gt=0, le=0.5)
    min_position_usd: Optional[float] = Field(None, ge=0)
    enabled: Optional[bool] = None


class PositionResponse(BaseModel):
    """Position response."""

    id: int
    account_id: int
    market_id: str
    token_id: str
    outcome: str
    our_size: str
    target_size: str
    average_price: str
    current_price: Optional[str] = None
    unrealized_pnl: Optional[str] = None
    status: str
    drift_percent: Optional[str] = None


class TradeResponse(BaseModel):
    """Trade response."""

    id: int
    account_id: int
    market_id: str
    side: str
    target_price: str
    target_size: str
    execution_price: Optional[str] = None
    execution_size: Optional[str] = None
    slippage_percent: Optional[str] = None
    status: str
    detected_at: str
    executed_at: Optional[str] = None


class PnLResponse(BaseModel):
    """P&L summary response."""

    realized_pnl: str
    unrealized_pnl: str
    total_pnl: str
    roi_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_percent: float
    total_volume: str


class HealthResponse(BaseModel):
    """Health check response."""

    overall_status: str
    components: List[dict]
    uptime_seconds: float
    timestamp: str


class AlertConfigResponse(BaseModel):
    """Alert configuration response."""

    webhook_url: Optional[str] = None
    enabled: bool = True
    min_severity: str = "warning"


class AlertConfigUpdate(BaseModel):
    """Alert configuration update."""

    webhook_url: Optional[str] = None
    enabled: Optional[bool] = None
    min_severity: Optional[str] = None


# =============================================================================
# Status & Control Endpoints
# =============================================================================

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current bot status."""
    # In full implementation, would check actual bot state
    bot_state = get_bot_state()

    return StatusResponse(
        is_running=bot_state.get("is_running", False),
        uptime_seconds=bot_state.get("uptime_seconds"),
        state=bot_state.get("state", "stopped"),
        health=bot_state.get("health", "unknown"),
    )


@router.post("/control/start")
async def start_bot():
    """Start the copy-trading bot."""
    # In full implementation, would start bot process
    logger.info("api_start_requested")
    return {"status": "start_requested", "message": "Bot start initiated"}


@router.post("/control/stop")
async def stop_bot():
    """Stop the copy-trading bot gracefully."""
    logger.info("api_stop_requested")
    return {"status": "stop_requested", "message": "Bot stop initiated"}


@router.post("/control/pause")
async def pause_bot():
    """Pause trading (keep monitoring)."""
    logger.info("api_pause_requested")
    return {"status": "paused", "message": "Trading paused"}


@router.post("/control/resume")
async def resume_bot():
    """Resume trading after pause."""
    logger.info("api_resume_requested")
    return {"status": "resumed", "message": "Trading resumed"}


# =============================================================================
# Balance Endpoints
# =============================================================================

@router.get("/balance", response_model=BalanceResponse)
async def get_balance(db=Depends(get_db)):
    """Get current balance information."""
    from ...database import BalanceSnapshotRepository

    repo = BalanceSnapshotRepository(db)
    latest = repo.get_latest()

    if not latest:
        return BalanceResponse(
            total="0",
            available="0",
            reserved="0",
            timestamp=datetime.utcnow().isoformat(),
        )

    return BalanceResponse(
        total=str(latest.total_balance),
        available=str(latest.available_balance),
        reserved=str(latest.reserved_balance),
        timestamp=latest.created_at.isoformat() if latest.created_at else "",
    )


# =============================================================================
# Account Endpoints
# =============================================================================

@router.get("/accounts", response_model=List[AccountResponse])
async def list_accounts(
    enabled_only: bool = Query(False, description="Only show enabled accounts"),
    db=Depends(get_db),
):
    """List all tracked accounts."""
    from ...database import AccountRepository

    repo = AccountRepository(db)

    if enabled_only:
        accounts = repo.get_enabled()
    else:
        accounts = repo.get_all()

    return [
        AccountResponse(
            id=a.id,
            name=a.name,
            target_wallet=a.target_wallet,
            position_ratio=str(a.position_ratio),
            max_position_usd=str(a.max_position_usd),
            slippage_tolerance=str(a.slippage_tolerance),
            min_position_usd=str(a.min_position_usd),
            enabled=a.enabled,
            created_at=a.created_at.isoformat() if a.created_at else None,
        )
        for a in accounts
    ]


@router.post("/accounts", response_model=AccountResponse, status_code=201)
async def create_account(account: AccountCreate, db=Depends(get_db)):
    """Create a new tracked account."""
    from ...database import AccountRepository

    repo = AccountRepository(db)

    # Check if name exists
    existing = repo.get_by_name(account.name)
    if existing:
        raise HTTPException(status_code=400, detail="Account name already exists")

    created = repo.create(
        name=account.name,
        target_wallet=account.wallet.lower(),
        position_ratio=Decimal(str(account.position_ratio)),
        max_position_usd=Decimal(str(account.max_position_usd)),
        slippage_tolerance=Decimal(str(account.slippage_tolerance)),
        min_position_usd=Decimal(str(account.min_position_usd)),
        enabled=True,
    )

    logger.info("account_created_via_api", name=account.name)

    return AccountResponse(
        id=created.id,
        name=created.name,
        target_wallet=created.target_wallet,
        position_ratio=str(created.position_ratio),
        max_position_usd=str(created.max_position_usd),
        slippage_tolerance=str(created.slippage_tolerance),
        min_position_usd=str(created.min_position_usd),
        enabled=created.enabled,
        created_at=created.created_at.isoformat() if created.created_at else None,
    )


@router.get("/accounts/{account_id}", response_model=AccountResponse)
async def get_account(account_id: int, db=Depends(get_db)):
    """Get a specific account."""
    from ...database import AccountRepository

    repo = AccountRepository(db)
    account = repo.get_by_id(account_id)

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    return AccountResponse(
        id=account.id,
        name=account.name,
        target_wallet=account.target_wallet,
        position_ratio=str(account.position_ratio),
        max_position_usd=str(account.max_position_usd),
        slippage_tolerance=str(account.slippage_tolerance),
        min_position_usd=str(account.min_position_usd),
        enabled=account.enabled,
        created_at=account.created_at.isoformat() if account.created_at else None,
    )


@router.patch("/accounts/{account_id}", response_model=AccountResponse)
async def update_account(account_id: int, update: AccountUpdate, db=Depends(get_db)):
    """Update an account."""
    from ...database import AccountRepository

    repo = AccountRepository(db)

    updates = {}
    if update.position_ratio is not None:
        updates["position_ratio"] = Decimal(str(update.position_ratio))
    if update.max_position_usd is not None:
        updates["max_position_usd"] = Decimal(str(update.max_position_usd))
    if update.slippage_tolerance is not None:
        updates["slippage_tolerance"] = Decimal(str(update.slippage_tolerance))
    if update.min_position_usd is not None:
        updates["min_position_usd"] = Decimal(str(update.min_position_usd))
    if update.enabled is not None:
        updates["enabled"] = update.enabled

    account = repo.update(account_id, **updates)

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    logger.info("account_updated_via_api", account_id=account_id)

    return AccountResponse(
        id=account.id,
        name=account.name,
        target_wallet=account.target_wallet,
        position_ratio=str(account.position_ratio),
        max_position_usd=str(account.max_position_usd),
        slippage_tolerance=str(account.slippage_tolerance),
        min_position_usd=str(account.min_position_usd),
        enabled=account.enabled,
        created_at=account.created_at.isoformat() if account.created_at else None,
    )


@router.delete("/accounts/{account_id}", status_code=204)
async def delete_account(account_id: int, db=Depends(get_db)):
    """Delete an account."""
    from ...database import AccountRepository

    repo = AccountRepository(db)

    if not repo.delete(account_id):
        raise HTTPException(status_code=404, detail="Account not found")

    logger.info("account_deleted_via_api", account_id=account_id)


# =============================================================================
# Position Endpoints
# =============================================================================

@router.get("/positions", response_model=List[PositionResponse])
async def list_positions(
    account_id: Optional[int] = Query(None, description="Filter by account"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db=Depends(get_db),
):
    """List all positions."""
    from ...database import PositionRepository

    repo = PositionRepository(db)
    positions = repo.get_open_positions(account_id)

    if status:
        positions = [p for p in positions if p.status.value == status]

    return [
        PositionResponse(
            id=p.id,
            account_id=p.account_id,
            market_id=p.market_id,
            token_id=p.token_id,
            outcome=p.outcome,
            our_size=str(p.our_size),
            target_size=str(p.target_size),
            average_price=str(p.average_price),
            current_price=str(p.current_price) if p.current_price else None,
            unrealized_pnl=str(p.unrealized_pnl) if p.unrealized_pnl else None,
            status=p.status.value,
            drift_percent=str(p.drift_percent) if p.drift_percent else None,
        )
        for p in positions
    ]


@router.get("/positions/summary")
async def positions_summary(db=Depends(get_db)):
    """Get positions summary."""
    from ...database import PositionRepository

    repo = PositionRepository(db)
    positions = repo.get_open_positions()

    total_value = sum(
        float(p.our_size) * float(p.current_price or p.average_price)
        for p in positions
    )
    total_pnl = sum(float(p.unrealized_pnl or 0) for p in positions)

    return {
        "open_count": len(positions),
        "total_value": str(round(total_value, 2)),
        "unrealized_pnl": str(round(total_pnl, 2)),
        "by_status": {
            "synced": sum(1 for p in positions if p.status.value == "synced"),
            "drift": sum(1 for p in positions if p.status.value == "drift"),
            "pending": sum(1 for p in positions if p.status.value == "pending"),
        },
    }


# =============================================================================
# Trade Endpoints
# =============================================================================

@router.get("/trades", response_model=List[TradeResponse])
async def list_trades(
    account_id: Optional[int] = Query(None, description="Filter by account"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    db=Depends(get_db),
):
    """List recent trades."""
    from ...database import TradeLogRepository, TradeStatus

    repo = TradeLogRepository(db)

    status_filter = None
    if status:
        try:
            status_filter = TradeStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    trades = repo.get_recent_trades(
        account_id=account_id,
        limit=limit,
        status=status_filter,
    )

    return [
        TradeResponse(
            id=t.id,
            account_id=t.account_id,
            market_id=t.market_id,
            side=t.side,
            target_price=str(t.target_price),
            target_size=str(t.target_size),
            execution_price=str(t.execution_price) if t.execution_price else None,
            execution_size=str(t.execution_size) if t.execution_size else None,
            slippage_percent=str(t.slippage_percent) if t.slippage_percent else None,
            status=t.status.value,
            detected_at=t.detected_at.isoformat() if t.detected_at else "",
            executed_at=t.executed_at.isoformat() if t.executed_at else None,
        )
        for t in trades
    ]


@router.get("/trades/stats")
async def trade_stats(
    days: int = Query(7, ge=1, le=365, description="Number of days"),
    db=Depends(get_db),
):
    """Get trade statistics."""
    from ...database import TradeLogRepository

    repo = TradeLogRepository(db)
    start = datetime.utcnow() - timedelta(days=days)

    stats = repo.get_trade_stats(start)

    return {
        "period_days": days,
        **stats,
    }


# =============================================================================
# P&L Endpoints
# =============================================================================

@router.get("/pnl", response_model=PnLResponse)
async def get_pnl(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    db=Depends(get_db),
):
    """Get P&L summary."""
    from ...database import AuditService, PositionRepository

    audit = AuditService(db)
    pos_repo = PositionRepository(db)

    summary = audit.get_performance_summary(days=days)
    positions = pos_repo.get_open_positions()

    unrealized = sum(float(p.unrealized_pnl or 0) for p in positions)
    realized = float(summary.get("total_realized_pnl", "0").replace(",", ""))

    total_pnl = realized + unrealized

    return PnLResponse(
        realized_pnl=str(round(realized, 2)),
        unrealized_pnl=str(round(unrealized, 2)),
        total_pnl=str(round(total_pnl, 2)),
        roi_percent=0,  # Would calculate from invested
        total_trades=summary.get("total_trades_executed", 0),
        winning_trades=0,
        losing_trades=0,
        win_rate_percent=summary.get("execution_rate", 0) * 100,
        total_volume=summary.get("total_volume_usd", "0"),
    )


@router.get("/pnl/daily")
async def daily_pnl(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    db=Depends(get_db),
):
    """Get daily P&L breakdown."""
    from ...database import DailyStatsRepository

    repo = DailyStatsRepository(db)
    stats = repo.get_history(days)

    return [
        {
            "date": s.date.isoformat() if s.date else None,
            "realized_pnl": str(s.realized_pnl),
            "trades_executed": s.trades_executed,
            "volume": str(s.total_volume_usd),
        }
        for s in stats
    ]


# =============================================================================
# Health Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Get system health status."""
    bot_state = get_bot_state()

    return HealthResponse(
        overall_status=bot_state.get("health", "unknown"),
        components=[],  # Would include actual component health
        uptime_seconds=bot_state.get("uptime_seconds") or 0.0,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/health/circuit-breaker")
async def circuit_breaker_status(db=Depends(get_db)):
    """Get circuit breaker status."""
    from ...database import CircuitBreakerEventRepository

    repo = CircuitBreakerEventRepository(db)
    recent_trips = repo.get_recent_trips(limit=5)

    return {
        "is_open": False,  # Would check actual state
        "recent_trips": [t.to_dict() for t in recent_trips],
    }


# =============================================================================
# Alert Configuration Endpoints
# =============================================================================

@router.get("/alerts/config", response_model=AlertConfigResponse)
async def get_alert_config():
    """Get alert configuration."""
    # In full implementation, would load from config/database
    return AlertConfigResponse(
        webhook_url=None,
        enabled=True,
        min_severity="warning",
    )


@router.patch("/alerts/config", response_model=AlertConfigResponse)
async def update_alert_config(update: AlertConfigUpdate):
    """Update alert configuration."""
    # In full implementation, would save to config/database
    logger.info("alert_config_updated")
    return AlertConfigResponse(
        webhook_url=update.webhook_url,
        enabled=update.enabled if update.enabled is not None else True,
        min_severity=update.min_severity or "warning",
    )


# =============================================================================
# System Events Endpoints
# =============================================================================

@router.get("/events")
async def list_events(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    event_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_db),
):
    """List system events."""
    from ...database import SystemEventRepository, EventSeverity

    repo = SystemEventRepository(db)

    sev = None
    if severity:
        try:
            sev = EventSeverity(severity)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")

    events = repo.get_recent_events(
        limit=limit,
        severity=sev,
        event_type=event_type,
    )

    return [e.to_dict() for e in events]
