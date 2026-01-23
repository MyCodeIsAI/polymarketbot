"""Account management module for PolymarketBot.

This module provides:
- Balance tracking with reservations
- Position management per account
- P&L calculation (realized + unrealized)
- Position drift detection
- Account reconciliation with API state
"""

from .balance import (
    BalanceTracker,
    MultiAccountBalanceTracker,
    BalanceChange,
    BalanceChangeType,
    BalanceSnapshot,
)
from .positions import (
    PositionManager,
    CopyPosition,
    PositionLot,
    PositionStatus,
)
from .pnl import (
    PnLCalculator,
    TradeResult,
    DailyPnL,
    PerformanceMetrics,
    PnLPeriod,
    MarketPnL,
)
from .reconciliation import (
    DriftDetector,
    DriftResult,
    DriftStatus,
    AccountReconciler,
    ReconciliationResult,
    PeriodicReconciler,
)

__all__ = [
    # Balance
    "BalanceTracker",
    "MultiAccountBalanceTracker",
    "BalanceChange",
    "BalanceChangeType",
    "BalanceSnapshot",
    # Positions
    "PositionManager",
    "CopyPosition",
    "PositionLot",
    "PositionStatus",
    # P&L
    "PnLCalculator",
    "TradeResult",
    "DailyPnL",
    "PerformanceMetrics",
    "PnLPeriod",
    "MarketPnL",
    # Reconciliation
    "DriftDetector",
    "DriftResult",
    "DriftStatus",
    "AccountReconciler",
    "ReconciliationResult",
    "PeriodicReconciler",
]
