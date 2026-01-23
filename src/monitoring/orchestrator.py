"""Monitoring orchestrator - coordinates all monitoring components.

This module provides the main entry point for the position monitoring
system, coordinating the poller, detector, reconciler, and state manager.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Coroutine, Any, Optional

from ..api.data import DataAPIClient
from ..config.models import AppConfig, TargetAccount
from ..utils.logging import get_logger
from .events import (
    EventBus,
    BaseEvent,
    EventType,
    MonitoringStartedEvent,
    MonitoringStoppedEvent,
    get_event_bus,
)
from .state import PositionStateManager, get_state_manager
from .poller import MultiTargetPoller, PollerConfig
from .detector import PositionChangeDetector
from .reconciliation import MultiTargetReconciler, ReconcilerConfig

logger = get_logger(__name__)


# Type alias for event handlers
EventHandler = Callable[[BaseEvent], Coroutine[Any, Any, None]]


@dataclass
class MonitoringStats:
    """Aggregate monitoring statistics."""

    started_at: Optional[datetime] = None
    targets_monitored: int = 0
    total_trades_detected: int = 0
    total_positions_opened: int = 0
    total_positions_closed: int = 0
    total_syncs: int = 0
    total_errors: int = 0


class MonitoringOrchestrator:
    """Orchestrates all monitoring components.

    Provides a unified interface for starting, stopping, and managing
    the position monitoring system.

    Example:
        orchestrator = MonitoringOrchestrator(config)
        await orchestrator.initialize()

        # Subscribe to events
        orchestrator.on_trade(handle_trade)
        orchestrator.on_position_opened(handle_position_opened)

        await orchestrator.start()
        # ... bot runs ...
        await orchestrator.stop()
    """

    def __init__(
        self,
        config: AppConfig,
        data_client: Optional[DataAPIClient] = None,
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[PositionStateManager] = None,
    ):
        """Initialize the monitoring orchestrator.

        Args:
            config: Application configuration
            data_client: Data API client (created if not provided)
            event_bus: Event bus (uses global if not provided)
            state_manager: State manager (uses global if not provided)
        """
        self.config = config

        # Use provided or create/get default instances
        self._owns_data_client = data_client is None
        self.data_client = data_client or DataAPIClient(
            timeout_s=config.network.api_timeout_s,
        )
        self.event_bus = event_bus or get_event_bus()
        self.state_manager = state_manager or get_state_manager()

        # Components (initialized in initialize())
        self._poller: Optional[MultiTargetPoller] = None
        self._detector: Optional[PositionChangeDetector] = None
        self._reconciler: Optional[MultiTargetReconciler] = None

        # State
        self._running = False
        self._initialized = False
        self.stats = MonitoringStats()

        # User-registered event handlers
        self._user_handlers: dict[EventType, list[EventHandler]] = {}

    async def initialize(self) -> None:
        """Initialize all monitoring components.

        Must be called before start().
        """
        if self._initialized:
            return

        targets = self.config.enabled_targets
        if not targets:
            logger.warning("no_enabled_targets")

        # Create poller configuration
        poller_config = PollerConfig(
            poll_interval_ms=self.config.polling.activity_interval_ms,
        )

        # Create reconciler configuration
        reconciler_config = ReconcilerConfig(
            sync_interval_s=self.config.polling.positions_sync_interval_s,
        )

        # Initialize components
        self._poller = MultiTargetPoller(
            targets=targets,
            data_client=self.data_client,
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            config=poller_config,
        )

        self._detector = PositionChangeDetector(
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            targets=targets,
        )

        self._reconciler = MultiTargetReconciler(
            targets=targets,
            data_client=self.data_client,
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            config=reconciler_config,
        )

        self._initialized = True
        logger.info(
            "orchestrator_initialized",
            targets=[t.name for t in targets],
        )

    async def start(self) -> None:
        """Start the monitoring system.

        Initializes if not already done, then starts all components.
        """
        if self._running:
            logger.warning("orchestrator_already_running")
            return

        if not self._initialized:
            await self.initialize()

        self._running = True
        self.stats.started_at = datetime.utcnow()
        self.stats.targets_monitored = len(self.config.enabled_targets)

        # Start event bus
        await self.event_bus.start()

        # Start detector (must be before poller to catch events)
        await self._detector.start()

        # Start poller and reconciler
        await self._poller.start_all()
        await self._reconciler.start_all()

        # Emit monitoring started events
        for target in self.config.enabled_targets:
            positions = await self.state_manager.get_all_positions(target.name)
            await self.event_bus.publish(MonitoringStartedEvent(
                target_name=target.name,
                target_wallet=target.wallet,
                position_count=len(positions),
            ))

        logger.info(
            "monitoring_started",
            targets=self.stats.targets_monitored,
        )

    async def stop(self) -> None:
        """Stop the monitoring system gracefully."""
        if not self._running:
            return

        self._running = False

        # Emit monitoring stopped events
        for target in self.config.enabled_targets:
            await self.event_bus.publish(MonitoringStoppedEvent(
                target_name=target.name,
                target_wallet=target.wallet,
                reason="shutdown",
            ))

        # Stop components in reverse order
        if self._reconciler:
            await self._reconciler.stop_all()

        if self._poller:
            await self._poller.stop_all()

        if self._detector:
            await self._detector.stop()

        # Stop event bus (processes remaining events)
        await self.event_bus.stop()

        # Close data client if we own it
        if self._owns_data_client and self.data_client:
            await self.data_client.close()

        logger.info(
            "monitoring_stopped",
            stats=self.stats.__dict__,
        )

    # =========================================================================
    # Event subscription helpers
    # =========================================================================

    def on_event(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe to a specific event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async handler function
        """
        self.event_bus.subscribe(event_type, handler)

    def on_any_event(self, handler: EventHandler) -> None:
        """Subscribe to all events.

        Args:
            handler: Async handler function
        """
        self.event_bus.subscribe(None, handler)

    def on_trade(self, handler: EventHandler) -> None:
        """Subscribe to trade detected events."""
        self.on_event(EventType.TRADE_DETECTED, handler)

    def on_position_opened(self, handler: EventHandler) -> None:
        """Subscribe to position opened events."""
        self.on_event(EventType.POSITION_OPENED, handler)

    def on_position_increased(self, handler: EventHandler) -> None:
        """Subscribe to position increased events."""
        self.on_event(EventType.POSITION_INCREASED, handler)

    def on_position_decreased(self, handler: EventHandler) -> None:
        """Subscribe to position decreased events."""
        self.on_event(EventType.POSITION_DECREASED, handler)

    def on_position_closed(self, handler: EventHandler) -> None:
        """Subscribe to position closed events."""
        self.on_event(EventType.POSITION_CLOSED, handler)

    def on_error(self, handler: EventHandler) -> None:
        """Subscribe to error events."""
        self.on_event(EventType.ERROR_OCCURRED, handler)

    # =========================================================================
    # Control methods
    # =========================================================================

    async def pause_target(self, target_name: str) -> bool:
        """Pause monitoring for a specific target.

        Args:
            target_name: Name of the target to pause

        Returns:
            True if successful
        """
        if self._poller:
            return await self._poller.stop_target(target_name)
        return False

    async def resume_target(self, target_name: str) -> bool:
        """Resume monitoring for a specific target.

        Args:
            target_name: Name of the target to resume

        Returns:
            True if successful
        """
        if self._poller:
            return await self._poller.start_target(target_name)
        return False

    async def force_sync(self, target_name: Optional[str] = None) -> dict:
        """Force a position synchronization.

        Args:
            target_name: Specific target to sync, or None for all

        Returns:
            Sync results
        """
        if not self._reconciler:
            return {}

        if target_name:
            reconciler = self._reconciler.get_reconciler(target_name)
            if reconciler:
                result = await reconciler.force_sync()
                return {target_name: result}
            return {}
        else:
            return await self._reconciler.force_sync_all()

    # =========================================================================
    # Status and statistics
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if monitoring is running."""
        return self._running

    def get_status(self) -> dict:
        """Get current monitoring status."""
        poller_stats = self._poller.get_aggregate_stats() if self._poller else {}
        reconciler_stats = self._reconciler.get_stats() if self._reconciler else {}

        return {
            "running": self._running,
            "initialized": self._initialized,
            "started_at": self.stats.started_at.isoformat() if self.stats.started_at else None,
            "targets_monitored": self.stats.targets_monitored,
            "poller": poller_stats,
            "reconciler": reconciler_stats,
            "event_bus": {
                "queue_size": self.event_bus.queue_size,
                "stats": self.event_bus.stats,
            },
            "state_manager": self.state_manager.stats,
        }

    async def get_positions_summary(self) -> dict:
        """Get summary of all tracked positions."""
        return await self.state_manager.get_summary()


async def create_and_start_monitoring(
    config: AppConfig,
) -> MonitoringOrchestrator:
    """Convenience function to create and start monitoring.

    Args:
        config: Application configuration

    Returns:
        Running MonitoringOrchestrator instance
    """
    orchestrator = MonitoringOrchestrator(config)
    await orchestrator.initialize()
    await orchestrator.start()
    return orchestrator
