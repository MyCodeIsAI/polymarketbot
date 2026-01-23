"""Activity polling for target wallet monitoring.

This module implements the core polling loop that detects new trades
by periodically checking the Polymarket Data API for activity updates.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Coroutine, Any

from ..api.data import DataAPIClient, Activity, ActivityType, TradeSide
from ..config.models import TargetAccount
from ..utils.logging import get_logger
from .events import (
    EventBus,
    TradeDetectedEvent,
    ErrorEvent,
)
from .state import PositionStateManager

logger = get_logger(__name__)


@dataclass
class PollerStats:
    """Statistics for a single poller."""

    polls: int = 0
    trades_detected: int = 0
    errors: int = 0
    last_poll_time: Optional[datetime] = None
    last_poll_latency_ms: float = 0
    avg_poll_latency_ms: float = 0
    total_latency_ms: float = 0


@dataclass
class PollerConfig:
    """Configuration for the activity poller."""

    poll_interval_ms: int = 200
    max_activities_per_poll: int = 50
    error_backoff_ms: int = 1000
    max_consecutive_errors: int = 5


class ActivityPoller:
    """Polls the Data API for new activity on a target wallet.

    Detects trades as quickly as possible by polling at a configurable
    interval and tracking the last seen activity timestamp.

    Example:
        poller = ActivityPoller(
            target=target_account,
            data_client=data_client,
            event_bus=event_bus,
            state_manager=state_manager,
        )
        await poller.start()
        # ... later ...
        await poller.stop()
    """

    def __init__(
        self,
        target: TargetAccount,
        data_client: DataAPIClient,
        event_bus: EventBus,
        state_manager: PositionStateManager,
        config: Optional[PollerConfig] = None,
    ):
        """Initialize the activity poller.

        Args:
            target: Target account configuration
            data_client: Data API client instance
            event_bus: Event bus for publishing events
            state_manager: Position state manager
            config: Poller configuration
        """
        self.target = target
        self.data_client = data_client
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.config = config or PollerConfig()

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_timestamp: Optional[int] = None
        self._seen_activity_ids: set[str] = set()
        self._consecutive_errors = 0

        self.stats = PollerStats()

    @property
    def is_running(self) -> bool:
        """Check if poller is running."""
        return self._running

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            logger.warning("poller_already_running", target=self.target.name)
            return

        self._running = True
        self._consecutive_errors = 0

        # Initialize last timestamp from state manager
        stored_ts = await self.state_manager.get_last_activity_timestamp(self.target.name)
        if stored_ts:
            self._last_timestamp = stored_ts
        else:
            # Start from now (don't process historical activity)
            self._last_timestamp = int(time.time()) - 60  # 1 minute buffer

        self._task = asyncio.create_task(self._poll_loop())

        logger.info(
            "poller_started",
            target=self.target.name,
            wallet=self.target.wallet[:10] + "...",
            interval_ms=self.config.poll_interval_ms,
        )

    async def stop(self) -> None:
        """Stop the polling loop."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Save last timestamp
        if self._last_timestamp:
            await self.state_manager.set_last_activity_timestamp(
                self.target.name,
                self._last_timestamp,
            )

        logger.info(
            "poller_stopped",
            target=self.target.name,
            stats=self.stats.__dict__,
        )

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            start_time = time.perf_counter()

            try:
                await self._poll_once()
                self._consecutive_errors = 0

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.stats.last_poll_latency_ms = latency_ms
                self.stats.total_latency_ms += latency_ms
                self.stats.polls += 1
                self.stats.avg_poll_latency_ms = (
                    self.stats.total_latency_ms / self.stats.polls
                )
                self.stats.last_poll_time = datetime.utcnow()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_errors += 1
                self.stats.errors += 1

                logger.error(
                    "poll_error",
                    target=self.target.name,
                    error=str(e),
                    consecutive_errors=self._consecutive_errors,
                )

                await self.event_bus.publish(ErrorEvent(
                    target_name=self.target.name,
                    error_type="poll_error",
                    error_message=str(e),
                    recoverable=self._consecutive_errors < self.config.max_consecutive_errors,
                ))

                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    logger.error(
                        "max_errors_reached",
                        target=self.target.name,
                        stopping=True,
                    )
                    self._running = False
                    break

                # Backoff on error
                await asyncio.sleep(self.config.error_backoff_ms / 1000)
                continue

            # Wait for next poll interval
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            sleep_ms = max(0, self.config.poll_interval_ms - elapsed_ms)
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000)

    async def _poll_once(self) -> None:
        """Execute a single poll for new activity."""
        activities = await self.data_client.get_activity(
            user=self.target.wallet,
            activity_type=ActivityType.TRADE,
            start_timestamp=self._last_timestamp,
            limit=self.config.max_activities_per_poll,
        )

        if not activities:
            return

        # Process activities (oldest first for correct ordering)
        activities.sort(key=lambda a: a.timestamp)

        for activity in activities:
            await self._process_activity(activity)

    async def _process_activity(self, activity: Activity) -> None:
        """Process a detected activity.

        Args:
            activity: The activity to process
        """
        # Skip if already seen (deduplication)
        if activity.id in self._seen_activity_ids:
            return

        self._seen_activity_ids.add(activity.id)

        # Limit seen IDs set size
        if len(self._seen_activity_ids) > 10000:
            # Remove oldest half
            self._seen_activity_ids = set(list(self._seen_activity_ids)[-5000:])

        # Update last timestamp
        activity_ts = int(activity.timestamp.timestamp())
        if self._last_timestamp is None or activity_ts > self._last_timestamp:
            self._last_timestamp = activity_ts

        # Only process trades
        if activity.type != ActivityType.TRADE:
            return

        self.stats.trades_detected += 1

        # Calculate detection latency
        detection_latency_ms = (datetime.utcnow() - activity.timestamp).total_seconds() * 1000

        logger.info(
            "trade_detected",
            target=self.target.name,
            side=activity.side.value if activity.side else "UNKNOWN",
            size=str(activity.size),
            price=str(activity.price),
            token_id=activity.token_id[:16] + "..." if activity.token_id else "N/A",
            detection_latency_ms=round(detection_latency_ms, 0),
        )

        # Publish trade detected event
        await self.event_bus.publish(TradeDetectedEvent(
            target_name=self.target.name,
            target_wallet=self.target.wallet,
            condition_id=activity.condition_id,
            token_id=activity.token_id,
            outcome=activity.outcome,
            side=activity.side.value if activity.side else "UNKNOWN",
            size=activity.size,
            price=activity.price,
            usd_value=activity.usd_value,
            trade_timestamp=activity.timestamp,
            activity_id=activity.id,
        ))

    async def force_poll(self) -> int:
        """Force an immediate poll.

        Returns:
            Number of new activities detected
        """
        initial_count = self.stats.trades_detected
        await self._poll_once()
        return self.stats.trades_detected - initial_count


class MultiTargetPoller:
    """Manages polling for multiple target accounts.

    Coordinates multiple ActivityPoller instances and provides
    aggregate statistics.

    Example:
        poller = MultiTargetPoller(
            targets=config.enabled_targets,
            data_client=data_client,
            event_bus=event_bus,
            state_manager=state_manager,
        )
        await poller.start_all()
    """

    def __init__(
        self,
        targets: list[TargetAccount],
        data_client: DataAPIClient,
        event_bus: EventBus,
        state_manager: PositionStateManager,
        config: Optional[PollerConfig] = None,
    ):
        """Initialize multi-target poller.

        Args:
            targets: List of target accounts to monitor
            data_client: Shared Data API client
            event_bus: Shared event bus
            state_manager: Shared state manager
            config: Poller configuration
        """
        self.targets = targets
        self.data_client = data_client
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.config = config or PollerConfig()

        self._pollers: dict[str, ActivityPoller] = {}

        # Create poller for each target
        for target in targets:
            self._pollers[target.name] = ActivityPoller(
                target=target,
                data_client=data_client,
                event_bus=event_bus,
                state_manager=state_manager,
                config=config,
            )

    async def start_all(self) -> None:
        """Start polling for all enabled targets."""
        for name, poller in self._pollers.items():
            if poller.target.enabled:
                await poller.start()

        logger.info(
            "multi_poller_started",
            targets=[t.name for t in self.targets if t.enabled],
        )

    async def stop_all(self) -> None:
        """Stop all pollers."""
        for poller in self._pollers.values():
            await poller.stop()

        logger.info("multi_poller_stopped")

    async def start_target(self, target_name: str) -> bool:
        """Start polling for a specific target.

        Args:
            target_name: Name of the target

        Returns:
            True if started, False if target not found
        """
        if target_name not in self._pollers:
            return False

        await self._pollers[target_name].start()
        return True

    async def stop_target(self, target_name: str) -> bool:
        """Stop polling for a specific target.

        Args:
            target_name: Name of the target

        Returns:
            True if stopped, False if target not found
        """
        if target_name not in self._pollers:
            return False

        await self._pollers[target_name].stop()
        return True

    def get_poller(self, target_name: str) -> Optional[ActivityPoller]:
        """Get the poller for a specific target."""
        return self._pollers.get(target_name)

    def get_stats(self) -> dict[str, PollerStats]:
        """Get statistics for all pollers."""
        return {name: poller.stats for name, poller in self._pollers.items()}

    def get_aggregate_stats(self) -> dict:
        """Get aggregate statistics across all pollers."""
        total_polls = 0
        total_trades = 0
        total_errors = 0
        running_count = 0

        for poller in self._pollers.values():
            total_polls += poller.stats.polls
            total_trades += poller.stats.trades_detected
            total_errors += poller.stats.errors
            if poller.is_running:
                running_count += 1

        return {
            "total_targets": len(self._pollers),
            "running_targets": running_count,
            "total_polls": total_polls,
            "total_trades_detected": total_trades,
            "total_errors": total_errors,
        }
