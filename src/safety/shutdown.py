"""Graceful shutdown handling for trading bot.

This module provides:
- Signal handling (SIGINT, SIGTERM)
- Ordered shutdown sequence
- Pending order cancellation
- State persistence before exit
- Timeout handling for stuck components
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Awaitable, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ShutdownReason(str, Enum):
    """Reasons for shutdown."""

    SIGINT = "sigint"  # Ctrl+C
    SIGTERM = "sigterm"  # Kill signal
    CIRCUIT_BREAKER = "circuit_breaker"  # Safety trip
    FATAL_ERROR = "fatal_error"  # Unrecoverable error
    USER_REQUEST = "user_request"  # API/command request
    HEALTH_FAILURE = "health_failure"  # Health check failure
    MAINTENANCE = "maintenance"  # Scheduled maintenance


class ShutdownPhase(str, Enum):
    """Shutdown phases (executed in order)."""

    STOP_NEW_ORDERS = "stop_new_orders"  # Stop accepting new orders
    CANCEL_PENDING = "cancel_pending"  # Cancel pending orders
    CLOSE_WEBSOCKETS = "close_websockets"  # Close WS connections
    STOP_MONITORING = "stop_monitoring"  # Stop target monitoring
    PERSIST_STATE = "persist_state"  # Save state to disk
    CLEANUP = "cleanup"  # Final cleanup


@dataclass
class ShutdownTask:
    """A task to run during shutdown."""

    name: str
    phase: ShutdownPhase
    handler: Callable[[], Awaitable[None]]
    timeout_s: float = 30
    critical: bool = False  # If True, failure aborts shutdown
    order: int = 0  # Lower runs first within phase


@dataclass
class ShutdownResult:
    """Result of shutdown procedure."""

    reason: ShutdownReason
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    phases_completed: list[ShutdownPhase] = field(default_factory=list)
    tasks_completed: list[str] = field(default_factory=list)
    tasks_failed: list[tuple[str, str]] = field(default_factory=list)  # (name, error)
    aborted: bool = False
    abort_reason: Optional[str] = None

    @property
    def duration_s(self) -> float:
        """Get shutdown duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.utcnow() - self.started_at).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "reason": self.reason.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_s": self.duration_s,
            "success": self.success,
            "phases_completed": [p.value for p in self.phases_completed],
            "tasks_completed": self.tasks_completed,
            "tasks_failed": [
                {"name": name, "error": error}
                for name, error in self.tasks_failed
            ],
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }


# Callback types
ShutdownStartCallback = Callable[[ShutdownReason], Awaitable[None]]
ShutdownCompleteCallback = Callable[[ShutdownResult], Awaitable[None]]


class GracefulShutdown:
    """Manages graceful shutdown of the trading bot.

    Coordinates ordered shutdown of all components with timeouts.
    Handles signal registration and emergency abort.

    Example:
        shutdown = GracefulShutdown()

        # Register shutdown tasks
        shutdown.register(
            name="cancel_orders",
            phase=ShutdownPhase.CANCEL_PENDING,
            handler=order_manager.cancel_all,
            timeout_s=30,
        )

        # Install signal handlers
        shutdown.install_signal_handlers()

        # Manual shutdown
        await shutdown.initiate(ShutdownReason.USER_REQUEST)
    """

    def __init__(
        self,
        total_timeout_s: float = 120,
        on_shutdown_start: Optional[ShutdownStartCallback] = None,
        on_shutdown_complete: Optional[ShutdownCompleteCallback] = None,
    ):
        """Initialize graceful shutdown manager.

        Args:
            total_timeout_s: Maximum time for entire shutdown
            on_shutdown_start: Callback when shutdown begins
            on_shutdown_complete: Callback when shutdown ends
        """
        self.total_timeout_s = total_timeout_s
        self._on_shutdown_start = on_shutdown_start
        self._on_shutdown_complete = on_shutdown_complete

        self._tasks: list[ShutdownTask] = []
        self._shutdown_in_progress = False
        self._shutdown_complete = False
        self._result: Optional[ShutdownResult] = None

        # For signal handling
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutdown_in_progress

    @property
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown is complete."""
        return self._shutdown_complete

    def register(
        self,
        name: str,
        phase: ShutdownPhase,
        handler: Callable[[], Awaitable[None]],
        timeout_s: float = 30,
        critical: bool = False,
        order: int = 0,
    ) -> None:
        """Register a shutdown task.

        Args:
            name: Task name for logging
            phase: Shutdown phase
            handler: Async function to call
            timeout_s: Timeout for this task
            critical: Whether failure should abort shutdown
            order: Order within phase (lower first)
        """
        task = ShutdownTask(
            name=name,
            phase=phase,
            handler=handler,
            timeout_s=timeout_s,
            critical=critical,
            order=order,
        )
        self._tasks.append(task)

        logger.debug(
            "shutdown_task_registered",
            name=name,
            phase=phase.value,
        )

    def unregister(self, name: str) -> bool:
        """Unregister a shutdown task.

        Args:
            name: Task name

        Returns:
            True if task was found and removed
        """
        for i, task in enumerate(self._tasks):
            if task.name == name:
                self._tasks.pop(i)
                return True
        return False

    def install_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Install signal handlers for graceful shutdown.

        Args:
            loop: Event loop (uses running loop if not provided)
        """
        self._loop = loop or asyncio.get_event_loop()

        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._signal_handler(s)),
                )
                logger.debug("signal_handler_installed", signal=sig.name)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, self._sync_signal_handler)
                logger.debug("signal_handler_installed_sync", signal=sig.name)

    def _sync_signal_handler(self, signum: int, frame: Any) -> None:
        """Synchronous signal handler for Windows."""
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._signal_handler(signal.Signals(signum))
                )
            )

    async def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle shutdown signal.

        Args:
            sig: Signal received
        """
        reason = ShutdownReason.SIGINT if sig == signal.SIGINT else ShutdownReason.SIGTERM

        logger.warning(
            "shutdown_signal_received",
            signal=sig.name,
            reason=reason.value,
        )

        await self.initiate(reason)

    async def initiate(self, reason: ShutdownReason) -> ShutdownResult:
        """Initiate graceful shutdown.

        Args:
            reason: Reason for shutdown

        Returns:
            ShutdownResult with outcome
        """
        if self._shutdown_in_progress:
            logger.warning("shutdown_already_in_progress")
            if self._result:
                return self._result
            # Wait for completion
            await self._shutdown_event.wait()
            return self._result or ShutdownResult(
                reason=reason,
                started_at=datetime.utcnow(),
                success=False,
            )

        self._shutdown_in_progress = True
        self._result = ShutdownResult(
            reason=reason,
            started_at=datetime.utcnow(),
        )

        logger.critical(
            "SHUTDOWN_INITIATED",
            reason=reason.value,
            total_timeout_s=self.total_timeout_s,
            tasks=len(self._tasks),
        )

        # Notify callback
        if self._on_shutdown_start:
            try:
                await self._on_shutdown_start(reason)
            except Exception as e:
                logger.error("shutdown_start_callback_error", error=str(e))

        # Execute shutdown with timeout
        try:
            await asyncio.wait_for(
                self._execute_shutdown(),
                timeout=self.total_timeout_s,
            )
            self._result.success = True
        except asyncio.TimeoutError:
            logger.error(
                "shutdown_timeout",
                elapsed_s=self._result.duration_s,
            )
            self._result.aborted = True
            self._result.abort_reason = "Total timeout exceeded"
        except Exception as e:
            logger.error("shutdown_error", error=str(e))
            self._result.aborted = True
            self._result.abort_reason = str(e)

        self._result.completed_at = datetime.utcnow()
        self._shutdown_complete = True
        self._shutdown_event.set()

        logger.info(
            "shutdown_complete",
            success=self._result.success,
            duration_s=self._result.duration_s,
            tasks_completed=len(self._result.tasks_completed),
            tasks_failed=len(self._result.tasks_failed),
        )

        # Notify callback
        if self._on_shutdown_complete:
            try:
                await self._on_shutdown_complete(self._result)
            except Exception as e:
                logger.error("shutdown_complete_callback_error", error=str(e))

        return self._result

    async def _execute_shutdown(self) -> None:
        """Execute shutdown phases in order."""
        # Group tasks by phase
        phase_tasks: dict[ShutdownPhase, list[ShutdownTask]] = {}
        for task in self._tasks:
            if task.phase not in phase_tasks:
                phase_tasks[task.phase] = []
            phase_tasks[task.phase].append(task)

        # Sort tasks within each phase
        for phase in phase_tasks:
            phase_tasks[phase].sort(key=lambda t: t.order)

        # Execute phases in order
        for phase in ShutdownPhase:
            if phase not in phase_tasks:
                continue

            logger.info(
                "shutdown_phase_starting",
                phase=phase.value,
                tasks=len(phase_tasks[phase]),
            )

            for task in phase_tasks[phase]:
                success = await self._execute_task(task)

                if not success and task.critical:
                    self._result.aborted = True
                    self._result.abort_reason = f"Critical task failed: {task.name}"
                    logger.error(
                        "shutdown_aborted",
                        reason=self._result.abort_reason,
                    )
                    return

            self._result.phases_completed.append(phase)

            logger.info(
                "shutdown_phase_complete",
                phase=phase.value,
            )

    async def _execute_task(self, task: ShutdownTask) -> bool:
        """Execute a single shutdown task.

        Args:
            task: Task to execute

        Returns:
            True if successful
        """
        logger.debug(
            "shutdown_task_starting",
            name=task.name,
            timeout_s=task.timeout_s,
        )

        try:
            await asyncio.wait_for(
                task.handler(),
                timeout=task.timeout_s,
            )
            self._result.tasks_completed.append(task.name)

            logger.debug(
                "shutdown_task_complete",
                name=task.name,
            )
            return True

        except asyncio.TimeoutError:
            error = f"Timeout after {task.timeout_s}s"
            self._result.tasks_failed.append((task.name, error))

            logger.error(
                "shutdown_task_timeout",
                name=task.name,
                timeout_s=task.timeout_s,
            )
            return False

        except Exception as e:
            self._result.tasks_failed.append((task.name, str(e)))

            logger.error(
                "shutdown_task_error",
                name=task.name,
                error=str(e),
            )
            return False

    async def wait_for_shutdown(self) -> ShutdownResult:
        """Wait for shutdown to complete.

        Returns:
            ShutdownResult when complete
        """
        await self._shutdown_event.wait()
        return self._result or ShutdownResult(
            reason=ShutdownReason.USER_REQUEST,
            started_at=datetime.utcnow(),
            success=False,
        )

    def get_status(self) -> dict:
        """Get current shutdown status.

        Returns:
            Status dictionary
        """
        return {
            "in_progress": self._shutdown_in_progress,
            "complete": self._shutdown_complete,
            "registered_tasks": len(self._tasks),
            "result": self._result.to_dict() if self._result else None,
        }


class EmergencyShutdown:
    """Emergency shutdown for critical situations.

    Provides fast path to halt trading without graceful cleanup.
    Use when immediate stop is required (e.g., suspected breach).
    """

    def __init__(
        self,
        halt_trading_func: Callable[[], Awaitable[None]],
        cancel_all_func: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """Initialize emergency shutdown.

        Args:
            halt_trading_func: Function to immediately halt trading
            cancel_all_func: Function to cancel all orders
        """
        self._halt_trading = halt_trading_func
        self._cancel_all = cancel_all_func
        self._triggered = False
        self._triggered_at: Optional[datetime] = None
        self._reason: Optional[str] = None

    @property
    def is_triggered(self) -> bool:
        """Check if emergency shutdown was triggered."""
        return self._triggered

    async def trigger(self, reason: str) -> None:
        """Trigger emergency shutdown.

        Args:
            reason: Reason for emergency shutdown
        """
        if self._triggered:
            logger.warning("emergency_shutdown_already_triggered")
            return

        self._triggered = True
        self._triggered_at = datetime.utcnow()
        self._reason = reason

        logger.critical(
            "EMERGENCY_SHUTDOWN_TRIGGERED",
            reason=reason,
        )

        # Halt trading immediately
        try:
            await self._halt_trading()
        except Exception as e:
            logger.error("halt_trading_error", error=str(e))

        # Attempt to cancel orders
        if self._cancel_all:
            try:
                await asyncio.wait_for(
                    self._cancel_all(),
                    timeout=10,  # Short timeout for emergency
                )
            except asyncio.TimeoutError:
                logger.error("cancel_all_timeout")
            except Exception as e:
                logger.error("cancel_all_error", error=str(e))

    def get_status(self) -> dict:
        """Get emergency shutdown status.

        Returns:
            Status dictionary
        """
        return {
            "triggered": self._triggered,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "reason": self._reason,
        }


class ShutdownCoordinator:
    """Coordinates both graceful and emergency shutdown.

    Provides unified interface for shutdown management.

    Example:
        coordinator = ShutdownCoordinator()

        # Setup graceful shutdown
        coordinator.graceful.register(...)
        coordinator.graceful.install_signal_handlers()

        # Setup emergency shutdown
        coordinator.setup_emergency(halt_func, cancel_func)

        # In critical error handler
        await coordinator.emergency("Suspected breach")

        # Or normal shutdown
        await coordinator.shutdown()
    """

    def __init__(
        self,
        total_timeout_s: float = 120,
    ):
        """Initialize shutdown coordinator.

        Args:
            total_timeout_s: Timeout for graceful shutdown
        """
        self.graceful = GracefulShutdown(total_timeout_s=total_timeout_s)
        self._emergency: Optional[EmergencyShutdown] = None

    def setup_emergency(
        self,
        halt_trading_func: Callable[[], Awaitable[None]],
        cancel_all_func: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Setup emergency shutdown.

        Args:
            halt_trading_func: Function to halt trading
            cancel_all_func: Function to cancel orders
        """
        self._emergency = EmergencyShutdown(halt_trading_func, cancel_all_func)

    async def shutdown(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST) -> ShutdownResult:
        """Initiate graceful shutdown.

        Args:
            reason: Shutdown reason

        Returns:
            ShutdownResult
        """
        return await self.graceful.initiate(reason)

    async def emergency(self, reason: str) -> None:
        """Trigger emergency shutdown.

        Args:
            reason: Reason for emergency
        """
        if self._emergency:
            await self._emergency.trigger(reason)
        else:
            logger.error("emergency_shutdown_not_configured")
            # Fall back to graceful
            await self.graceful.initiate(ShutdownReason.FATAL_ERROR)

    @property
    def is_shutting_down(self) -> bool:
        """Check if any shutdown is in progress."""
        emergency = self._emergency.is_triggered if self._emergency else False
        return self.graceful.is_shutting_down or emergency

    def get_status(self) -> dict:
        """Get shutdown status.

        Returns:
            Combined status dictionary
        """
        return {
            "graceful": self.graceful.get_status(),
            "emergency": self._emergency.get_status() if self._emergency else None,
        }
