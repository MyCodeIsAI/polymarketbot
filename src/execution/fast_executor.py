"""Ultra-low-latency order executor optimized for copy trading.

This executor prioritizes speed over everything else:
- Uses time.perf_counter() for microsecond-precision timing
- Skips non-essential pre-flight checks
- Aggressive timeouts
- Minimal logging in hot path
- Direct submission without queuing

WARNING: This is for PRODUCTION copy trading with real money.
Every millisecond matters for execution quality.
"""

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ..api.clob import CLOBClient
from ..api.auth import PolymarketAuth
from ..core.performance import (
    PerformanceConfig,
    LatencyTracker,
    LatencyBudget,
    get_latency_tracker,
    create_latency_budget,
)
from ..utils.logging import get_logger
from .order_builder import CopyOrder
from .executor import ExecutionStatus, ExecutionResult

logger = get_logger(__name__)


@dataclass
class FastExecutionResult:
    """Minimal result structure for speed."""

    order_id: str
    clob_order_id: Optional[str]
    status: ExecutionStatus
    filled_size: Decimal = Decimal("0")
    filled_price: Optional[Decimal] = None
    error: Optional[str] = None

    # Timing breakdown (all in milliseconds)
    total_ms: float = 0
    sign_ms: float = 0
    submit_ms: float = 0
    confirm_ms: float = 0

    @property
    def is_success(self) -> bool:
        return self.status in (
            ExecutionStatus.SUBMITTED,
            ExecutionStatus.MATCHED,
            ExecutionStatus.FILLED,
            ExecutionStatus.PARTIALLY_FILLED,
        )

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "clob_order_id": self.clob_order_id,
            "status": self.status.value,
            "filled_size": str(self.filled_size),
            "filled_price": str(self.filled_price) if self.filled_price else None,
            "error": self.error,
            "timing": {
                "total_ms": round(self.total_ms, 3),
                "sign_ms": round(self.sign_ms, 3),
                "submit_ms": round(self.submit_ms, 3),
                "confirm_ms": round(self.confirm_ms, 3),
            },
        }


class FastExecutor:
    """Ultra-low-latency order executor.

    Optimizations:
    1. No pre-flight order book fetch (caller already has price)
    2. Minimal object allocation in hot path
    3. Aggressive timeouts (fail fast)
    4. Single retry with minimal delay
    5. Parallel status check (don't block on confirmation)

    Usage:
        executor = FastExecutor(clob_client, auth)
        result = await executor.execute(order)
    """

    def __init__(
        self,
        clob_client: CLOBClient,
        auth: PolymarketAuth,
        config: Optional[PerformanceConfig] = None,
    ):
        self.clob = clob_client
        self.auth = auth
        self.config = config or PerformanceConfig.aggressive()
        self.tracker = get_latency_tracker()

        # Stats
        self._submitted = 0
        self._filled = 0
        self._failed = 0
        self._total_latency_ms = 0

    async def execute(
        self,
        order: CopyOrder,
        budget: Optional[LatencyBudget] = None,
    ) -> FastExecutionResult:
        """Execute order with minimum latency.

        Args:
            order: Order to execute
            budget: Optional latency budget tracker

        Returns:
            FastExecutionResult with timing breakdown
        """
        # Use perf_counter for high precision
        t_start = time.perf_counter()
        budget = budget or create_latency_budget(self.config.latency_target_e2e_ms)

        result = FastExecutionResult(
            order_id=order.order_id,
            clob_order_id=None,
            status=ExecutionStatus.PENDING,
        )

        try:
            # Step 1: Build order params (minimal processing)
            order_params = order.to_clob_params()

            # Step 2: Sign order
            t_sign = time.perf_counter()
            # Auth signing is handled by CLOB client internally
            result.sign_ms = (time.perf_counter() - t_sign) * 1000

            # Step 3: Submit with aggressive timeout
            t_submit = time.perf_counter()
            submit_timeout = budget.get_timeout_for_stage(
                "submit",
                self.config.order_submit_timeout_ms,
            ) / 1000  # Convert to seconds

            try:
                response = await asyncio.wait_for(
                    self.clob.place_order(order_params),
                    timeout=submit_timeout,
                )
            except asyncio.TimeoutError:
                result.status = ExecutionStatus.FAILED
                result.error = f"Submit timeout ({submit_timeout*1000:.0f}ms)"
                result.submit_ms = (time.perf_counter() - t_submit) * 1000
                self._failed += 1
                return self._finalize(result, t_start)

            result.submit_ms = (time.perf_counter() - t_submit) * 1000

            # Check immediate response
            if not response.get("success", False):
                result.status = ExecutionStatus.FAILED
                result.error = response.get("error", "Unknown submission error")
                self._failed += 1
                return self._finalize(result, t_start)

            # Got order ID - submission successful
            clob_order_id = response.get("orderID") or response.get("order_id")
            result.clob_order_id = clob_order_id
            result.status = ExecutionStatus.SUBMITTED
            self._submitted += 1

            # Step 4: Quick status check (non-blocking confirmation)
            t_confirm = time.perf_counter()

            # Only poll if we have budget remaining
            if budget.remaining_ms > 50:
                fill_result = await self._quick_status_check(
                    clob_order_id,
                    max_polls=3,  # Quick check only
                    poll_interval_ms=self.config.order_status_poll_ms,
                )

                if fill_result:
                    result.status = fill_result["status"]
                    result.filled_size = fill_result.get("filled_size", Decimal("0"))
                    result.filled_price = fill_result.get("avg_price")

                    if result.status == ExecutionStatus.FILLED:
                        self._filled += 1

            result.confirm_ms = (time.perf_counter() - t_confirm) * 1000

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            self._failed += 1
            logger.error("fast_execute_error", order_id=order.order_id, error=str(e))

        return self._finalize(result, t_start)

    async def _quick_status_check(
        self,
        clob_order_id: str,
        max_polls: int,
        poll_interval_ms: int,
    ) -> Optional[dict]:
        """Quick non-blocking status check.

        Returns immediately if order is filled, or after max_polls.
        """
        for _ in range(max_polls):
            try:
                status = await asyncio.wait_for(
                    self.clob.get_order(clob_order_id),
                    timeout=0.2,  # 200ms max per poll
                )

                if status is None:
                    await asyncio.sleep(poll_interval_ms / 1000)
                    continue

                order_status = status.get("status", "").upper()
                filled_size = Decimal(str(status.get("filledSize", "0")))

                if order_status == "FILLED":
                    return {
                        "status": ExecutionStatus.FILLED,
                        "filled_size": filled_size,
                        "avg_price": Decimal(str(status.get("avgFillPrice", "0"))),
                    }
                elif order_status == "MATCHED":
                    return {
                        "status": ExecutionStatus.MATCHED,
                        "filled_size": filled_size,
                    }
                elif order_status in ("CANCELLED", "EXPIRED"):
                    return {
                        "status": ExecutionStatus.CANCELLED if order_status == "CANCELLED" else ExecutionStatus.EXPIRED,
                        "filled_size": filled_size,
                    }

                # Still pending, quick sleep
                await asyncio.sleep(poll_interval_ms / 1000)

            except asyncio.TimeoutError:
                continue
            except Exception:
                continue

        return None

    def _finalize(
        self,
        result: FastExecutionResult,
        start_time: float,
    ) -> FastExecutionResult:
        """Finalize result with total timing."""
        result.total_ms = (time.perf_counter() - start_time) * 1000
        self._total_latency_ms += result.total_ms

        # Record to tracker (async-safe fire and forget)
        asyncio.create_task(
            self.tracker.record("execution", result.total_ms)
        )

        return result

    async def execute_batch(
        self,
        orders: list[CopyOrder],
        max_concurrent: int = 3,
    ) -> list[FastExecutionResult]:
        """Execute multiple orders with controlled concurrency.

        Args:
            orders: Orders to execute
            max_concurrent: Max parallel executions

        Returns:
            List of results in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(order: CopyOrder) -> FastExecutionResult:
            async with semaphore:
                return await self.execute(order)

        tasks = [execute_with_semaphore(order) for order in orders]
        return await asyncio.gather(*tasks)

    @property
    def stats(self) -> dict:
        """Get executor statistics."""
        return {
            "submitted": self._submitted,
            "filled": self._filled,
            "failed": self._failed,
            "fill_rate": self._filled / max(self._submitted, 1),
            "avg_latency_ms": round(
                self._total_latency_ms / max(self._submitted + self._failed, 1),
                2,
            ),
        }


class CopyTradeExecutor:
    """Specialized executor for copy trading scenarios.

    Combines FastExecutor with copy-trade specific logic:
    - Position ratio scaling
    - Slippage protection
    - Target wallet tracking
    """

    def __init__(
        self,
        clob_client: CLOBClient,
        auth: PolymarketAuth,
        position_ratio: Decimal = Decimal("0.01"),
        max_position_usd: Decimal = Decimal("500"),
        slippage_tolerance: Decimal = Decimal("0.05"),
        config: Optional[PerformanceConfig] = None,
    ):
        self.executor = FastExecutor(clob_client, auth, config)
        self.position_ratio = position_ratio
        self.max_position_usd = max_position_usd
        self.slippage_tolerance = slippage_tolerance
        self.tracker = get_latency_tracker()

    async def copy_trade(
        self,
        target_trade: dict,
        current_price: Optional[Decimal] = None,
    ) -> FastExecutionResult:
        """Copy a trade from target wallet.

        Args:
            target_trade: Trade to copy (from WebSocket)
            current_price: Current market price (if known)

        Returns:
            Execution result
        """
        t_start = time.perf_counter()

        # Extract trade details
        token_id = target_trade.get("token_id") or target_trade.get("tokenId")
        side = target_trade.get("side", "").upper()
        target_size = Decimal(str(target_trade.get("size", 0)))
        target_price = Decimal(str(target_trade.get("price", 0)))

        # Scale position
        our_size = min(
            target_size * self.position_ratio,
            self.max_position_usd / target_price if target_price > 0 else Decimal("0"),
        )

        # Check slippage if we have current price
        if current_price:
            slippage = abs(current_price - target_price) / target_price
            if slippage > self.slippage_tolerance:
                return FastExecutionResult(
                    order_id=f"copy_{target_trade.get('id', 'unknown')}",
                    clob_order_id=None,
                    status=ExecutionStatus.FAILED,
                    error=f"Slippage {slippage*100:.1f}% > {self.slippage_tolerance*100:.1f}% tolerance",
                    total_ms=(time.perf_counter() - t_start) * 1000,
                )

        # Build copy order
        from .order_builder import CopyOrder, OrderSide, OrderType

        order = CopyOrder(
            order_id=f"copy_{target_trade.get('id', 'unknown')}",
            token_id=token_id,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            price=current_price or target_price,
            size=our_size,
            order_type=OrderType.GTC,
        )

        # Record detection latency
        detection_ms = (time.perf_counter() - t_start) * 1000
        await self.tracker.record("detection", detection_ms)

        # Execute
        result = await self.executor.execute(order)

        # Record total copy latency
        total_ms = (time.perf_counter() - t_start) * 1000
        await self.tracker.record("e2e", total_ms)

        return result

    @property
    def stats(self) -> dict:
        """Get combined statistics."""
        base_stats = self.executor.stats
        latency_stats = self.tracker.get_all_stats()
        return {
            **base_stats,
            "latency": latency_stats,
        }
