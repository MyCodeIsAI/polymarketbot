"""Safety systems module for PolymarketBot.

This module provides:
- Circuit breaker framework with pluggable conditions
- Position and loss limits
- Health check system
- Alert system (webhook, logging, file)
- Graceful shutdown handling
"""

from .circuit_breaker import (
    BreakerState,
    TripReason,
    TripEvent,
    StateSnapshot,
    BreakerCondition,
    CircuitBreaker,
    CompositeCircuitBreaker,
)
from .limits import (
    MaxDailyLoss,
    MaxDrawdown,
    MaxConsecutiveFailures,
    MaxSlippageEvents,
    APIErrorRate,
    PositionDriftTooHigh,
    BalanceTooLow,
    WebSocketDisconnected,
    RateLimitExceeded,
)
from .health_check import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    APIHealthChecker,
    WebSocketHealthChecker,
    RateLimitHealthChecker,
    SystemResourceChecker,
    QueueDepthChecker,
    HealthCheckManager,
)
from .alerts import (
    AlertSeverity,
    AlertCategory,
    Alert,
    AlertChannel,
    LoggingChannel,
    WebhookChannel,
    FileChannel,
    AlertThrottler,
    AlertManager,
)
from .shutdown import (
    ShutdownReason,
    ShutdownPhase,
    ShutdownTask,
    ShutdownResult,
    GracefulShutdown,
    EmergencyShutdown,
    ShutdownCoordinator,
)

__all__ = [
    # Circuit breaker
    "BreakerState",
    "TripReason",
    "TripEvent",
    "StateSnapshot",
    "BreakerCondition",
    "CircuitBreaker",
    "CompositeCircuitBreaker",
    # Limits
    "MaxDailyLoss",
    "MaxDrawdown",
    "MaxConsecutiveFailures",
    "MaxSlippageEvents",
    "APIErrorRate",
    "PositionDriftTooHigh",
    "BalanceTooLow",
    "WebSocketDisconnected",
    "RateLimitExceeded",
    # Health check
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "APIHealthChecker",
    "WebSocketHealthChecker",
    "RateLimitHealthChecker",
    "SystemResourceChecker",
    "QueueDepthChecker",
    "HealthCheckManager",
    # Alerts
    "AlertSeverity",
    "AlertCategory",
    "Alert",
    "AlertChannel",
    "LoggingChannel",
    "WebhookChannel",
    "FileChannel",
    "AlertThrottler",
    "AlertManager",
    # Shutdown
    "ShutdownReason",
    "ShutdownPhase",
    "ShutdownTask",
    "ShutdownResult",
    "GracefulShutdown",
    "EmergencyShutdown",
    "ShutdownCoordinator",
]
