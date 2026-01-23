"""Position monitoring for PolymarketBot.

This module provides real-time monitoring of target wallet positions:
- Activity polling for fast trade detection
- Position state tracking
- Change detection and event emission
- Periodic reconciliation with full position sync
"""

from .events import (
    EventBus,
    EventType,
    BaseEvent,
    TradeDetectedEvent,
    PositionOpenedEvent,
    PositionIncreasedEvent,
    PositionDecreasedEvent,
    PositionClosedEvent,
    MonitoringStartedEvent,
    MonitoringStoppedEvent,
    SyncCompletedEvent,
    ErrorEvent,
    get_event_bus,
    reset_event_bus,
)
from .state import (
    PositionStateManager,
    TrackedPosition,
    PositionChange,
    ChangeType,
    PositionStatus,
    get_state_manager,
    reset_state_manager,
)
from .poller import (
    ActivityPoller,
    MultiTargetPoller,
    PollerConfig,
    PollerStats,
)
from .detector import (
    PositionChangeDetector,
    TradeAggregator,
)
from .reconciliation import (
    PositionReconciler,
    MultiTargetReconciler,
    ReconcilerConfig,
    ReconciliationResult,
)
from .orchestrator import (
    MonitoringOrchestrator,
    MonitoringStats,
    create_and_start_monitoring,
)
from .latency_monitor import (
    LatencyMonitor,
    LatencyStage,
    LatencyAlert,
    AlertLevel,
    LatencyThresholds,
    StageMetrics,
    get_latency_monitor,
)

__all__ = [
    # Events
    "EventBus",
    "EventType",
    "BaseEvent",
    "TradeDetectedEvent",
    "PositionOpenedEvent",
    "PositionIncreasedEvent",
    "PositionDecreasedEvent",
    "PositionClosedEvent",
    "MonitoringStartedEvent",
    "MonitoringStoppedEvent",
    "SyncCompletedEvent",
    "ErrorEvent",
    "get_event_bus",
    "reset_event_bus",
    # State
    "PositionStateManager",
    "TrackedPosition",
    "PositionChange",
    "ChangeType",
    "PositionStatus",
    "get_state_manager",
    "reset_state_manager",
    # Polling
    "ActivityPoller",
    "MultiTargetPoller",
    "PollerConfig",
    "PollerStats",
    # Detection
    "PositionChangeDetector",
    "TradeAggregator",
    # Reconciliation
    "PositionReconciler",
    "MultiTargetReconciler",
    "ReconcilerConfig",
    "ReconciliationResult",
    # Orchestrator
    "MonitoringOrchestrator",
    "MonitoringStats",
    "create_and_start_monitoring",
    # Latency Monitoring
    "LatencyMonitor",
    "LatencyStage",
    "LatencyAlert",
    "AlertLevel",
    "LatencyThresholds",
    "StageMetrics",
    "get_latency_monitor",
]
