"""Order executor for submitting orders to the CLOB.

This module handles:
- Order signing with L2 authentication
- Order submission to CLOB API
- Order status monitoring
- Retry logic with exponential backoff
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..api.clob import CLOBClient, OrderSide
from ..api.auth import PolymarketAuth
from ..utils.logging import get_logger
from .order_builder import CopyOrder
from .slippage import SlippageCalculator, SlippageStatus

logger = get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Status of order execution."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    MATCHED = "matched"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""

    order_id: str
    clob_order_id: Optional[str]
    status: ExecutionStatus
    filled_size: Decimal = Decimal("0")
    filled_price: Optional[Decimal] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0
    attempts: int = 1

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status in (
            ExecutionStatus.SUBMITTED,
            ExecutionStatus.MATCHED,
            ExecutionStatus.FILLED,
            ExecutionStatus.PARTIALLY_FILLED,
        )

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (no more fills expected)."""
        return self.status in (
            ExecutionStatus.FILLED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.EXPIRED,
            ExecutionStatus.FAILED,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "clob_order_id": self.clob_order_id,
            "status": self.status.value,
            "filled_size": str(self.filled_size),
            "filled_price": str(self.filled_price) if self.filled_price else None,
            "error_message": self.error_message,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "attempts": self.attempts,
        }


@dataclass
class ExecutorConfig:
    """Configuration for the order executor."""

    # Retry settings
    max_retries: int = 3
    initial_retry_delay_ms: int = 100
    max_retry_delay_ms: int = 2000
    retry_multiplier: float = 2.0

    # Execution settings
    submit_timeout_ms: int = 5000
    status_poll_interval_ms: int = 500
    max_status_polls: int = 10

    # Pre-flight checks
    check_price_before_submit: bool = True
    max_price_drift_percent: Decimal = Decimal("0.02")  # 2%


@dataclass
class ExecutorStats:
    """Statistics for the executor."""

    orders_submitted: int = 0
    orders_filled: int = 0
    orders_partial: int = 0
    orders_failed: int = 0
    total_retries: int = 0
    total_execution_time_ms: float = 0

    @property
    def success_rate(self) -> float:
        total = self.orders_submitted
        if total == 0:
            return 0
        return (self.orders_filled + self.orders_partial) / total

    @property
    def avg_execution_time_ms(self) -> float:
        if self.orders_submitted == 0:
            return 0
        return self.total_execution_time_ms / self.orders_submitted


class OrderExecutor:
    """Executes orders on the Polymarket CLOB.

    Handles the complete lifecycle of order submission:
    1. Pre-flight price validation
    2. Order signing
    3. Submission with retries
    4. Status monitoring

    Example:
        executor = OrderExecutor(clob_client, auth, slippage_calc)
        result = await executor.execute(order)
        if result.is_success:
            print(f"Order filled: {result.filled_size}")
    """

    def __init__(
        self,
        clob_client: CLOBClient,
        auth: PolymarketAuth,
        slippage_calculator: Optional[SlippageCalculator] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize the executor.

        Args:
            clob_client: CLOB API client
            auth: Polymarket authentication
            slippage_calculator: Optional slippage calculator for pre-flight checks
            config: Executor configuration
        """
        self.clob = clob_client
        self.auth = auth
        self.slippage = slippage_calculator or SlippageCalculator()
        self.config = config or ExecutorConfig()
        self.stats = ExecutorStats()

    async def execute(self, order: CopyOrder) -> ExecutionResult:
        """Execute an order.

        Args:
            order: Order to execute

        Returns:
            ExecutionResult with execution details
        """
        start_time = datetime.utcnow()
        attempts = 0
        last_error = None

        while attempts < self.config.max_retries:
            attempts += 1

            try:
                result = await self._execute_once(order, attempts)

                # Update stats
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                result.execution_time_ms = execution_time
                result.attempts = attempts
                self._update_stats(result, attempts - 1)

                if result.is_success or not self._should_retry(result):
                    return result

                last_error = result.error_message

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "execution_attempt_failed",
                    order_id=order.order_id,
                    attempt=attempts,
                    error=last_error,
                )

            # Calculate retry delay
            if attempts < self.config.max_retries:
                delay = self._calculate_retry_delay(attempts)
                self.stats.total_retries += 1
                await asyncio.sleep(delay / 1000)

        # All retries exhausted
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.stats.orders_failed += 1

        return ExecutionResult(
            order_id=order.order_id,
            clob_order_id=None,
            status=ExecutionStatus.FAILED,
            error_message=f"Max retries exceeded: {last_error}",
            execution_time_ms=execution_time,
            attempts=attempts,
        )

    async def _execute_once(
        self,
        order: CopyOrder,
        attempt: int,
    ) -> ExecutionResult:
        """Single execution attempt.

        Args:
            order: Order to execute
            attempt: Current attempt number

        Returns:
            ExecutionResult from this attempt
        """
        # Pre-flight price check
        if self.config.check_price_before_submit:
            price_ok = await self._check_price_valid(order)
            if not price_ok:
                return ExecutionResult(
                    order_id=order.order_id,
                    clob_order_id=None,
                    status=ExecutionStatus.FAILED,
                    error_message="Price drifted too far",
                )

        # Get order parameters
        order_params = order.to_clob_params()

        # Sign the order
        signed_order = await self._sign_order(order_params)

        logger.debug(
            "submitting_order",
            order_id=order.order_id,
            token_id=order.token_id[:16] + "...",
            side=order.side.value,
            price=str(order.price),
            size=str(order.size),
            attempt=attempt,
        )

        # Submit to CLOB
        try:
            response = await asyncio.wait_for(
                self.clob.place_order(signed_order),
                timeout=self.config.submit_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            return ExecutionResult(
                order_id=order.order_id,
                clob_order_id=None,
                status=ExecutionStatus.FAILED,
                error_message="Order submission timeout",
            )

        if not response.get("success", False):
            error_msg = response.get("error", "Unknown error")
            return ExecutionResult(
                order_id=order.order_id,
                clob_order_id=None,
                status=ExecutionStatus.FAILED,
                error_message=error_msg,
            )

        clob_order_id = response.get("orderID") or response.get("order_id")
        order.clob_order_id = clob_order_id
        order.submitted = True

        self.stats.orders_submitted += 1

        # Poll for status
        status_result = await self._poll_order_status(order, clob_order_id)

        return ExecutionResult(
            order_id=order.order_id,
            clob_order_id=clob_order_id,
            status=status_result["status"],
            filled_size=status_result.get("filled_size", Decimal("0")),
            filled_price=status_result.get("avg_price"),
        )

    async def _sign_order(self, order_params: dict) -> dict:
        """Sign an order for submission.

        Args:
            order_params: Order parameters

        Returns:
            Signed order ready for submission
        """
        # The CLOB client handles signing internally
        # This is a hook for additional processing if needed
        return order_params

    async def _check_price_valid(self, order: CopyOrder) -> bool:
        """Check if current price is still valid for execution.

        Args:
            order: Order to check

        Returns:
            True if price is valid
        """
        try:
            order_book = await self.clob.get_order_book(order.token_id)

            if order.is_buy:
                current_price = order_book.best_ask
            else:
                current_price = order_book.best_bid

            if current_price is None:
                logger.warning(
                    "no_price_for_validation",
                    order_id=order.order_id,
                )
                return True  # Allow execution anyway

            # Check drift
            drift = abs(current_price - order.price) / order.price

            if drift > self.config.max_price_drift_percent:
                logger.warning(
                    "price_drift_exceeded",
                    order_id=order.order_id,
                    order_price=str(order.price),
                    current_price=str(current_price),
                    drift_percent=str(drift * 100),
                )
                return False

            return True

        except Exception as e:
            logger.warning(
                "price_check_failed",
                order_id=order.order_id,
                error=str(e),
            )
            return True  # Allow execution on error

    async def _poll_order_status(
        self,
        order: CopyOrder,
        clob_order_id: str,
    ) -> dict:
        """Poll for order status until terminal state.

        Args:
            order: The order
            clob_order_id: CLOB order ID

        Returns:
            Dict with status info
        """
        polls = 0

        while polls < self.config.max_status_polls:
            polls += 1

            try:
                status = await self.clob.get_order(clob_order_id)

                if status is None:
                    # Order not found yet, may still be processing
                    await asyncio.sleep(self.config.status_poll_interval_ms / 1000)
                    continue

                order_status = status.get("status", "").upper()
                filled_size = Decimal(str(status.get("filledSize", "0")))

                # Map CLOB status to our status
                if order_status == "FILLED":
                    order.fill_size = filled_size
                    order.status = "filled"
                    return {
                        "status": ExecutionStatus.FILLED,
                        "filled_size": filled_size,
                        "avg_price": Decimal(str(status.get("avgFillPrice", order.price))),
                    }

                elif order_status == "MATCHED":
                    return {
                        "status": ExecutionStatus.MATCHED,
                        "filled_size": filled_size,
                    }

                elif order_status == "CANCELLED":
                    if filled_size > 0:
                        order.fill_size = filled_size
                        return {
                            "status": ExecutionStatus.PARTIALLY_FILLED,
                            "filled_size": filled_size,
                        }
                    return {"status": ExecutionStatus.CANCELLED}

                elif order_status == "EXPIRED":
                    return {"status": ExecutionStatus.EXPIRED}

                elif order_status in ("OPEN", "LIVE", "PENDING"):
                    # Still active, keep polling
                    await asyncio.sleep(self.config.status_poll_interval_ms / 1000)
                    continue

                else:
                    # Unknown status, treat as submitted
                    return {"status": ExecutionStatus.SUBMITTED}

            except Exception as e:
                logger.warning(
                    "status_poll_error",
                    order_id=order.order_id,
                    error=str(e),
                )
                await asyncio.sleep(self.config.status_poll_interval_ms / 1000)

        # Max polls reached, return submitted status
        return {"status": ExecutionStatus.SUBMITTED}

    def _should_retry(self, result: ExecutionResult) -> bool:
        """Determine if execution should be retried.

        Args:
            result: Previous execution result

        Returns:
            True if should retry
        """
        # Don't retry successful orders
        if result.is_success:
            return False

        # Don't retry cancelled or expired
        if result.status in (ExecutionStatus.CANCELLED, ExecutionStatus.EXPIRED):
            return False

        # Retry failed orders
        return result.status == ExecutionStatus.FAILED

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before next retry.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in milliseconds
        """
        delay = self.config.initial_retry_delay_ms * (
            self.config.retry_multiplier ** (attempt - 1)
        )
        return min(delay, self.config.max_retry_delay_ms)

    def _update_stats(self, result: ExecutionResult, retries: int) -> None:
        """Update executor statistics.

        Args:
            result: Execution result
            retries: Number of retries used
        """
        self.stats.total_execution_time_ms += result.execution_time_ms
        self.stats.total_retries += retries

        if result.status == ExecutionStatus.FILLED:
            self.stats.orders_filled += 1
        elif result.status == ExecutionStatus.PARTIALLY_FILLED:
            self.stats.orders_partial += 1
        elif not result.is_success:
            self.stats.orders_failed += 1

    async def cancel_order(self, clob_order_id: str) -> bool:
        """Cancel an open order.

        Args:
            clob_order_id: CLOB order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            result = await self.clob.cancel_order(clob_order_id)
            return result.get("success", False)
        except Exception as e:
            logger.error(
                "cancel_order_failed",
                clob_order_id=clob_order_id,
                error=str(e),
            )
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        try:
            result = await self.clob.cancel_all_orders()
            return result.get("cancelled", 0)
        except Exception as e:
            logger.error("cancel_all_orders_failed", error=str(e))
            return 0


class ExecutorPool:
    """Pool of executors for parallel order execution.

    Manages multiple executor instances for different targets
    or market conditions.
    """

    def __init__(
        self,
        clob_client: CLOBClient,
        auth: PolymarketAuth,
        pool_size: int = 3,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize executor pool.

        Args:
            clob_client: Shared CLOB client
            auth: Authentication
            pool_size: Number of executors
            config: Executor configuration
        """
        self.executors = [
            OrderExecutor(clob_client, auth, config=config)
            for _ in range(pool_size)
        ]
        self._index = 0
        self._lock = asyncio.Lock()

    async def execute(self, order: CopyOrder) -> ExecutionResult:
        """Execute order using next available executor.

        Args:
            order: Order to execute

        Returns:
            ExecutionResult
        """
        async with self._lock:
            executor = self.executors[self._index]
            self._index = (self._index + 1) % len(self.executors)

        return await executor.execute(order)

    def get_aggregate_stats(self) -> dict:
        """Get aggregated stats from all executors."""
        total = ExecutorStats()

        for executor in self.executors:
            stats = executor.stats
            total.orders_submitted += stats.orders_submitted
            total.orders_filled += stats.orders_filled
            total.orders_partial += stats.orders_partial
            total.orders_failed += stats.orders_failed
            total.total_retries += stats.total_retries
            total.total_execution_time_ms += stats.total_execution_time_ms

        return {
            "orders_submitted": total.orders_submitted,
            "orders_filled": total.orders_filled,
            "orders_partial": total.orders_partial,
            "orders_failed": total.orders_failed,
            "total_retries": total.total_retries,
            "success_rate": round(total.success_rate, 4),
            "avg_execution_time_ms": round(total.avg_execution_time_ms, 2),
        }
