"""Insider Scanner Module.

Detects suspicious trading patterns that may indicate insider trading.
Implements:
- Multi-dimensional scoring system (Account, Trading, Behavioral, Contextual, Cluster)
- Cumulative position tracking
- Variance-calibrated thresholds
- Immutable audit trail for legal protection
- Real-time trade monitoring with WebSocket/polling
"""

from .models import (
    InsiderPriority,
    FlaggedWallet,
    FlaggedFundingSource,
    InsiderSignal,
    DetectionRecord,
    InvestmentThesis,
    AuditChainEntry,
    CumulativePosition,
)
from .scoring import InsiderScorer
from .audit import AuditTrailManager
from .profile import ProfileFetcher, WalletProfile, PositionSummary, TradeEntry
from .monitor import (
    RealTimeMonitor,
    NewWalletMonitor,
    TradeEvent,
    WalletAccumulation,
    MonitorState,
    MonitorStats,
)
from .alerts import (
    InsiderAlertType,
    InsiderAlertPriority,
    InsiderAlert,
    InsiderAlertChannel,
    LoggingAlertChannel,
    DiscordAlertChannel,
    EmailAlertChannel,
    FileAlertChannel,
    AlertPreferences,
    InsiderAlertManager,
)

__all__ = [
    # Models
    "InsiderPriority",
    "FlaggedWallet",
    "FlaggedFundingSource",
    "InsiderSignal",
    "DetectionRecord",
    "InvestmentThesis",
    "AuditChainEntry",
    "CumulativePosition",
    # Scoring
    "InsiderScorer",
    # Audit
    "AuditTrailManager",
    # Profile
    "ProfileFetcher",
    "WalletProfile",
    "PositionSummary",
    "TradeEntry",
    # Monitor (Stage 4)
    "RealTimeMonitor",
    "NewWalletMonitor",
    "TradeEvent",
    "WalletAccumulation",
    "MonitorState",
    "MonitorStats",
    # Alerts (Stage 5)
    "InsiderAlertType",
    "InsiderAlertPriority",
    "InsiderAlert",
    "InsiderAlertChannel",
    "LoggingAlertChannel",
    "DiscordAlertChannel",
    "EmailAlertChannel",
    "FileAlertChannel",
    "AlertPreferences",
    "InsiderAlertManager",
]
