"""Execution queue for order management.

This module provides:
- Priority-based order queueing
- Concurrent order processing
- Order lifecycle management
- Queue statistics and monitoring
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import IntEnum
from typing import Optional, Callable, Awaitable
from collections import defaultdict

from ..utils.logging import get_logger
from .order_builder import CopyOrder, OrderSource

logger = get_logger(__name__)


class OrderPriority(IntEnum):
    """Order priority levels (lower = higher priority)."""

    CRITICAL = 0  # Exit orders when target exits
    HIGH = 1  # Copy trades following target
    NORMAL = 2  # Position sync orders
    LOW = 3  # Rebalancing orders


@dataclass
class QueuedOrder:
    """An order in the execution queue."""

    order: CopyOrder
    priority: OrderPriority
    queued_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    last_error: Optional[str] = None

    # Tracking
    processing: bool = False
    completed: bool = False
    cancelled: bool = False

    @property
    def age_ms(self) -> float:
        """Get age of queued order in milliseconds."""
        delta = datetime.utcnow() - self.queued_at
        return delta.total_seconds() * 1000

    def __lt__(self, other: "QueuedOrder") -> bool:
        """Compare for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.queued_at < other.queued_at


@dataclass
class QueueStats:
    """Statistics for the execution queue."""

    orders_queued: int = 0
    orders_processed: int = 0
    orders_succeeded: int = 0
    orders_failed: int = 0
    orders_cancelled: int = 0
    orders_expired: int = 0
    total_wait_time_ms: float = 0
    total_process_time_ms: float = 0

    @property
    def avg_wait_time_ms(self) -> float:
        """Average time in queue."""
        if self.orders_processed == 0:
            return 0
        return self.total_wait_time_ms / self.orders_processed

    @property
    def avg_process_time_ms(self) -> float:
        """Average processing time."""
        if self.orders_processed == 0:
            return 0
        return self.total_process_time_ms / self.orders_processed

    @property
    def success_rate(self) -> float:
        """Order success rate."""
        if self.orders_processed == 0:
            return 0
        return self.orders_succeeded / self.orders_processed

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "orders_queued": self.orders_queued,
            "orders_processed": self.orders_processed,
            "orders_succeeded": self.orders_succeeded,
            "orders_failed": self.orders_failed,
            "orders_cancelled": self.orders_cancelled,
            "orders_expired": self.orders_expired,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "avg_process_time_ms": round(self.avg_process_time_ms, 2),
            "success_rate": round(self.success_rate, 4),
        }


class ExecutionQueue:
    """Priority queue for order execution.

    Orders are processed based on priority:
    1. CRITICAL - Exit orders (must execute ASAP)
    2. HIGH - Copy trade orders
    3. NORMAL - Position sync orders
    4. LOW - Rebalancing orders

    Example:
        queue = ExecutionQueue(max_concurrent=3)
        await queue.start(executor_callback)

        # Add orders
        await queue.enqueue(order, priority=OrderPriority.HIGH)

        # Monitor
        print(queue.stats.to_dict())
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_size: int = 100,
        order_timeout_ms: int = 30000,
        max_retries: int = 3,
        stale_order_ms: int = 5000,
    ):
        """Initialize the execution queue.

        Args:
            max_concurrent: Maximum concurrent order executions
            max_queue_size: Maximum queue size before rejecting
            order_timeout_ms: Order execution timeout
            max_retries: Maximum retry attempts
            stale_order_ms: Age at which orders are considered stale
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.order_timeout_ms = order_timeout_ms
        self.max_retries = max_retries
        self.stale_order_ms = stale_order_ms

        # Queue state
        self._queue: asyncio.PriorityQueue[QueuedOrder] = asyncio.PriorityQueue()
        self._orders: dict[str, QueuedOrder] = {}
        self._processing: set[str] = set()

        # Execution control
        self._running = False
        self._workers: list[asyncio.Task] = []
        self._executor: Optional[Callable[[CopyOrder], Awaitable[bool]]] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Statistics
        self.stats = QueueStats()

        # Per-token rate limiting to avoid self-competition
        self._token_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def processing_count(self) -> int:
        return len(self._processing)

    async def start(
        self,
        executor: Callable[[CopyOrder], Awaitable[bool]],
    ) -> None:
        """Start the execution queue.

        Args:
            executor: Async function that executes orders, returns success bool
        """
        if self._running:
            return

        self._running = True
        self._executor = executor
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(
            "execution_queue_started",
            max_concurrent=self.max_concurrent,
            max_queue_size=self.max_queue_size,
        )

    async def stop(self) -> None:
        """Stop the execution queue."""
        if not self._running:
            return

        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info(
            "execution_queue_stopped",
            stats=self.stats.to_dict(),
        )

    async def enqueue(
        self,
        order: CopyOrder,
        priority: Optional[OrderPriority] = None,
    ) -> bool:
        """Add an order to the execution queue.

        Args:
            order: Order to execute
            priority: Order priority (auto-determined if not provided)

        Returns:
            True if order was queued, False if rejected
        """
        if not self._running:
            logger.warning("queue_not_running", order_id=order.order_id)
            return False

        if self.queue_size >= self.max_queue_size:
            logger.warning(
                "queue_full",
                order_id=order.order_id,
                queue_size=self.queue_size,
            )
            return False

        # Check for duplicate
        if order.order_id in self._orders:
            logger.warning("duplicate_order", order_id=order.order_id)
            return False

        # Determine priority if not provided
        if priority is None:
            priority = self._determine_priority(order)

        queued = QueuedOrder(order=order, priority=priority)
        self._orders[order.order_id] = queued

        await self._queue.put(queued)
        self.stats.orders_queued += 1

        logger.debug(
            "order_queued",
            order_id=order.order_id,
            priority=priority.name,
            queue_size=self.queue_size,
        )

        return True

    async def cancel(self, order_id: str) -> bool:
        """Cancel a queued order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if order was cancelled
        """
        if order_id not in self._orders:
            return False

        queued = self._orders[order_id]

        if queued.processing or queued.completed:
            return False

        queued.cancelled = True
        self.stats.orders_cancelled += 1

        logger.debug("order_cancelled", order_id=order_id)
        return True

    async def cancel_all_for_token(self, token_id: str) -> int:
        """Cancel all pending orders for a token.

        Args:
            token_id: Token ID to cancel orders for

        Returns:
            Number of orders cancelled
        """
        cancelled = 0

        for order_id, queued in list(self._orders.items()):
            if queued.order.token_id == token_id:
                if await self.cancel(order_id):
                    cancelled += 1

        return cancelled

    def get_order(self, order_id: str) -> Optional[QueuedOrder]:
        """Get a queued order by ID."""
        return self._orders.get(order_id)

    def get_pending_for_token(self, token_id: str) -> list[QueuedOrder]:
        """Get all pending orders for a token."""
        return [
            q for q in self._orders.values()
            if q.order.token_id == token_id
            and not q.completed
            and not q.cancelled
        ]

    def _determine_priority(self, order: CopyOrder) -> OrderPriority:
        """Determine order priority based on order properties.

        Args:
            order: The order to prioritize

        Returns:
            Appropriate priority level
        """
        # Exit orders (sells to close position) are critical
        if order.is_sell and order.source == OrderSource.COPY_TRADE:
            return OrderPriority.CRITICAL

        # Copy trades are high priority
        if order.source == OrderSource.COPY_TRADE:
            return OrderPriority.HIGH

        # Position syncs are normal
        if order.source == OrderSource.POSITION_SYNC:
            return OrderPriority.NORMAL

        # Rebalancing is low priority
        if order.source == OrderSource.REBALANCE:
            return OrderPriority.LOW

        return OrderPriority.NORMAL

    async def _worker(self, worker_id: int) -> None:
        """Worker task that processes orders from the queue.

        Args:
            worker_id: Worker identifier for logging
        """
        logger.debug("queue_worker_started", worker_id=worker_id)

        while self._running:
            try:
                # Get next order (with timeout to check running state)
                try:
                    queued = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the order
                await self._process_order(queued, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "queue_worker_error",
                    worker_id=worker_id,
                    error=str(e),
                )

        logger.debug("queue_worker_stopped", worker_id=worker_id)

    async def _process_order(
        self,
        queued: QueuedOrder,
        worker_id: int,
    ) -> None:
        """Process a single order.

        Args:
            queued: Queued order to process
            worker_id: Worker identifier
        """
        order = queued.order

        # Skip cancelled orders
        if queued.cancelled:
            self._cleanup_order(order.order_id)
            return

        # Check if order is stale
        if queued.age_ms > self.stale_order_ms:
            logger.warning(
                "order_stale",
                order_id=order.order_id,
                age_ms=queued.age_ms,
            )
            self.stats.orders_expired += 1
            self._cleanup_order(order.order_id)
            return

        # Mark as processing
        queued.processing = True
        self._processing.add(order.order_id)

        # Acquire per-token lock to avoid self-competition
        async with self._token_locks[order.token_id]:
            start_time = datetime.utcnow()
            wait_time_ms = queued.age_ms

            try:
                # Execute with semaphore
                async with self._semaphore:
                    success = await asyncio.wait_for(
                        self._executor(order),
                        timeout=self.order_timeout_ms / 1000,
                    )

                process_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                # Update stats
                self.stats.orders_processed += 1
                self.stats.total_wait_time_ms += wait_time_ms
                self.stats.total_process_time_ms += process_time_ms

                if success:
                    self.stats.orders_succeeded += 1
                    queued.completed = True
                    order.status = "executed"

                    logger.info(
                        "order_executed",
                        order_id=order.order_id,
                        wait_ms=round(wait_time_ms, 2),
                        process_ms=round(process_time_ms, 2),
                        worker=worker_id,
                    )
                else:
                    # Retry if attempts remain
                    queued.attempts += 1

                    if queued.attempts < self.max_retries:
                        queued.processing = False
                        self._processing.discard(order.order_id)
                        await self._queue.put(queued)

                        logger.warning(
                            "order_retry",
                            order_id=order.order_id,
                            attempt=queued.attempts,
                        )
                    else:
                        self.stats.orders_failed += 1
                        queued.completed = True
                        order.status = "failed"

                        logger.error(
                            "order_failed_max_retries",
                            order_id=order.order_id,
                            attempts=queued.attempts,
                        )

            except asyncio.TimeoutError:
                queued.attempts += 1
                queued.last_error = "timeout"
                self.stats.orders_failed += 1
                order.status = "timeout"

                logger.error(
                    "order_timeout",
                    order_id=order.order_id,
                    timeout_ms=self.order_timeout_ms,
                )

            except Exception as e:
                queued.attempts += 1
                queued.last_error = str(e)
                self.stats.orders_failed += 1
                order.status = "error"

                logger.error(
                    "order_execution_error",
                    order_id=order.order_id,
                    error=str(e),
                )

            finally:
                self._processing.discard(order.order_id)

                if queued.completed or queued.attempts >= self.max_retries:
                    self._cleanup_order(order.order_id)

    def _cleanup_order(self, order_id: str) -> None:
        """Remove order from tracking.

        Args:
            order_id: Order ID to clean up
        """
        self._orders.pop(order_id, None)
        self._processing.discard(order_id)


class BatchExecutionQueue(ExecutionQueue):
    """Execution queue with batch processing support.

    Collects orders for a short window and batches them for
    more efficient execution when multiple orders target the
    same market.
    """

    def __init__(
        self,
        batch_window_ms: int = 100,
        max_batch_size: int = 5,
        **kwargs,
    ):
        """Initialize batch execution queue.

        Args:
            batch_window_ms: Time to collect orders before batching
            max_batch_size: Maximum orders per batch
            **kwargs: Passed to ExecutionQueue
        """
        super().__init__(**kwargs)
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        self._batch_buffer: dict[str, list[QueuedOrder]] = defaultdict(list)
        self._batch_timers: dict[str, asyncio.Task] = {}

    async def enqueue(
        self,
        order: CopyOrder,
        priority: Optional[OrderPriority] = None,
    ) -> bool:
        """Add order to batch buffer.

        Orders are collected by token_id and processed together.
        """
        if not self._running:
            return False

        if priority is None:
            priority = self._determine_priority(order)

        # Critical orders bypass batching
        if priority == OrderPriority.CRITICAL:
            return await super().enqueue(order, priority)

        queued = QueuedOrder(order=order, priority=priority)
        self._orders[order.order_id] = queued

        # Add to batch buffer
        token_id = order.token_id
        self._batch_buffer[token_id].append(queued)

        # If batch is full, flush immediately
        if len(self._batch_buffer[token_id]) >= self.max_batch_size:
            await self._flush_batch(token_id)
        else:
            # Start/reset batch timer
            if token_id in self._batch_timers:
                self._batch_timers[token_id].cancel()

            self._batch_timers[token_id] = asyncio.create_task(
                self._batch_timer(token_id)
            )

        self.stats.orders_queued += 1
        return True

    async def _batch_timer(self, token_id: str) -> None:
        """Timer to flush batch after window expires."""
        await asyncio.sleep(self.batch_window_ms / 1000)
        await self._flush_batch(token_id)

    async def _flush_batch(self, token_id: str) -> None:
        """Flush all orders for a token to the queue."""
        if token_id in self._batch_timers:
            self._batch_timers[token_id].cancel()
            del self._batch_timers[token_id]

        orders = self._batch_buffer.pop(token_id, [])
        if not orders:
            return

        # Sort by priority and queue
        orders.sort()
        for queued in orders:
            if not queued.cancelled:
                await self._queue.put(queued)

        logger.debug(
            "batch_flushed",
            token_id=token_id[:16] + "...",
            order_count=len(orders),
        )

    async def stop(self) -> None:
        """Stop queue and flush all batches."""
        # Flush all pending batches
        for token_id in list(self._batch_buffer.keys()):
            await self._flush_batch(token_id)

        await super().stop()
