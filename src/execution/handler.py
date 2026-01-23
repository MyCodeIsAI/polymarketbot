"""Copy trade handler - connects monitoring to execution.

This is the central component that:
1. Listens to position change events
2. Calculates appropriate trade sizes
3. Checks slippage constraints
4. Builds and submits orders
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..api.clob import CLOBClient, OrderSide
from ..config.models import TargetAccount, SlippageAction
from ..monitoring.events import (
    EventBus,
    BaseEvent,
    PositionOpenedEvent,
    PositionIncreasedEvent,
    PositionDecreasedEvent,
    PositionClosedEvent,
)
from ..utils.logging import get_logger
from .order_builder import OrderBuilder, CopyOrder, OrderSource
from .queue import ExecutionQueue, OrderPriority
from .sizer import PositionSizer, SizingResult, SizeRejectionReason
from .slippage import SlippageCalculator, SlippageCheckResult, SlippageStatus
from .executor import OrderExecutor, ExecutionResult

logger = get_logger(__name__)


@dataclass
class CopyTradeDecision:
    """Decision about whether to copy a trade."""

    should_copy: bool
    reason: str
    order: Optional[CopyOrder] = None
    sizing_result: Optional[SizingResult] = None
    slippage_result: Optional[SlippageCheckResult] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "should_copy": self.should_copy,
            "reason": self.reason,
            "order_id": self.order.order_id if self.order else None,
            "sizing": self.sizing_result.to_dict() if self.sizing_result else None,
            "slippage": self.slippage_result.to_dict() if self.slippage_result else None,
        }


@dataclass
class HandlerStats:
    """Statistics for the copy trade handler."""

    events_received: int = 0
    positions_opened: int = 0
    positions_increased: int = 0
    positions_decreased: int = 0
    positions_closed: int = 0
    orders_created: int = 0
    orders_rejected_size: int = 0
    orders_rejected_slippage: int = 0
    orders_queued: int = 0
    orders_failed_to_queue: int = 0

    def to_dict(self) -> dict:
        return {
            "events_received": self.events_received,
            "positions_opened": self.positions_opened,
            "positions_increased": self.positions_increased,
            "positions_decreased": self.positions_decreased,
            "positions_closed": self.positions_closed,
            "orders_created": self.orders_created,
            "orders_rejected_size": self.orders_rejected_size,
            "orders_rejected_slippage": self.orders_rejected_slippage,
            "orders_queued": self.orders_queued,
        }


@dataclass
class TargetContext:
    """Context for a specific target account."""

    target: TargetAccount
    sizer: PositionSizer
    slippage_calculator: SlippageCalculator
    our_positions: dict[str, Decimal] = field(default_factory=dict)


class CopyTradeHandler:
    """Handles copy trading logic.

    Listens to position events from the monitoring system and:
    1. Evaluates whether to copy the trade
    2. Calculates appropriate position size
    3. Checks slippage constraints
    4. Creates and submits orders

    Example:
        handler = CopyTradeHandler(
            event_bus=event_bus,
            clob_client=clob_client,
            execution_queue=queue,
            targets=config.targets,
            our_balance=Decimal("1000"),
        )
        await handler.start()
    """

    def __init__(
        self,
        event_bus: EventBus,
        clob_client: CLOBClient,
        execution_queue: ExecutionQueue,
        targets: list[TargetAccount],
        our_balance: Decimal,
        order_builder: Optional[OrderBuilder] = None,
    ):
        """Initialize the copy trade handler.

        Args:
            event_bus: Event bus for receiving position events
            clob_client: CLOB API client for order book data
            execution_queue: Queue for order execution
            targets: Target accounts to copy
            our_balance: Our available balance
            order_builder: Order builder instance
        """
        self.event_bus = event_bus
        self.clob = clob_client
        self.queue = execution_queue
        self.our_balance = our_balance
        self.order_builder = order_builder or OrderBuilder()

        # Create context for each target
        self._targets: dict[str, TargetContext] = {}
        for target in targets:
            self._targets[target.name] = TargetContext(
                target=target,
                sizer=PositionSizer(target),
                slippage_calculator=SlippageCalculator(
                    max_slippage=target.max_slippage,
                    slippage_action=target.slippage_action,
                ),
            )

        # Our position tracking (token_id -> size)
        self._our_positions: dict[str, Decimal] = {}

        self.stats = HandlerStats()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start listening for position events."""
        if self._running:
            return

        self._running = True

        # Subscribe to all position events
        self.event_bus.subscribe("position_opened", self._handle_position_opened)
        self.event_bus.subscribe("position_increased", self._handle_position_increased)
        self.event_bus.subscribe("position_decreased", self._handle_position_decreased)
        self.event_bus.subscribe("position_closed", self._handle_position_closed)

        logger.info(
            "copy_trade_handler_started",
            targets=list(self._targets.keys()),
            balance=str(self.our_balance),
        )

    async def stop(self) -> None:
        """Stop the handler."""
        if not self._running:
            return

        self._running = False
        logger.info("copy_trade_handler_stopped", stats=self.stats.to_dict())

    def update_balance(self, balance: Decimal) -> None:
        """Update our available balance.

        Args:
            balance: New balance
        """
        self.our_balance = balance
        logger.debug("balance_updated", balance=str(balance))

    def update_our_position(self, token_id: str, size: Decimal) -> None:
        """Update our position for a token.

        Args:
            token_id: Token ID
            size: Our position size
        """
        if size > 0:
            self._our_positions[token_id] = size
        else:
            self._our_positions.pop(token_id, None)

    async def _handle_position_opened(self, event: BaseEvent) -> None:
        """Handle target opening a new position.

        Args:
            event: Position opened event
        """
        if not isinstance(event, PositionOpenedEvent):
            return

        self.stats.events_received += 1
        self.stats.positions_opened += 1

        ctx = self._targets.get(event.target_name)
        if not ctx or not ctx.target.enabled:
            return

        logger.info(
            "target_position_opened",
            target=event.target_name,
            token_id=event.token_id[:16] + "...",
            size=str(event.size),
            price=str(event.entry_price),
            usd_value=str(event.usd_value),
        )

        # Evaluate and create copy trade
        decision = await self._evaluate_entry(
            ctx=ctx,
            token_id=event.token_id,
            condition_id=event.condition_id,
            outcome=event.outcome,
            target_size=event.size,
            target_price=event.entry_price,
        )

        await self._process_decision(decision, OrderPriority.HIGH)

    async def _handle_position_increased(self, event: BaseEvent) -> None:
        """Handle target increasing a position.

        Args:
            event: Position increased event
        """
        if not isinstance(event, PositionIncreasedEvent):
            return

        self.stats.events_received += 1
        self.stats.positions_increased += 1

        ctx = self._targets.get(event.target_name)
        if not ctx or not ctx.target.enabled:
            return

        logger.info(
            "target_position_increased",
            target=event.target_name,
            token_id=event.token_id[:16] + "...",
            delta=str(event.size_delta),
            new_size=str(event.new_size),
            price=str(event.price),
        )

        # Evaluate and create copy trade for the increase
        decision = await self._evaluate_entry(
            ctx=ctx,
            token_id=event.token_id,
            condition_id=event.condition_id,
            outcome=event.outcome,
            target_size=event.size_delta,  # Only copy the increase
            target_price=event.price,
        )

        await self._process_decision(decision, OrderPriority.HIGH)

    async def _handle_position_decreased(self, event: BaseEvent) -> None:
        """Handle target reducing a position.

        Args:
            event: Position decreased event
        """
        if not isinstance(event, PositionDecreasedEvent):
            return

        self.stats.events_received += 1
        self.stats.positions_decreased += 1

        ctx = self._targets.get(event.target_name)
        if not ctx or not ctx.target.enabled:
            return

        logger.info(
            "target_position_decreased",
            target=event.target_name,
            token_id=event.token_id[:16] + "...",
            delta=str(event.size_delta),
            remaining=str(event.new_size),
            price=str(event.price),
        )

        # Evaluate and create exit trade
        decision = await self._evaluate_exit(
            ctx=ctx,
            token_id=event.token_id,
            condition_id=event.condition_id,
            outcome=event.outcome,
            target_exit_size=event.size_delta,
            target_remaining=event.new_size,
            exit_price=event.price,
        )

        await self._process_decision(decision, OrderPriority.HIGH)

    async def _handle_position_closed(self, event: BaseEvent) -> None:
        """Handle target closing a position.

        Args:
            event: Position closed event
        """
        if not isinstance(event, PositionClosedEvent):
            return

        self.stats.events_received += 1
        self.stats.positions_closed += 1

        ctx = self._targets.get(event.target_name)
        if not ctx or not ctx.target.enabled:
            return

        logger.info(
            "target_position_closed",
            target=event.target_name,
            token_id=event.token_id[:16] + "...",
            closed_size=str(event.closed_size),
            exit_price=str(event.exit_price),
        )

        # Full exit - close our position
        decision = await self._evaluate_full_exit(
            ctx=ctx,
            token_id=event.token_id,
            condition_id=event.condition_id,
            outcome=event.outcome,
            exit_price=event.exit_price,
        )

        # Exit orders are critical priority
        await self._process_decision(decision, OrderPriority.CRITICAL)

    async def _evaluate_entry(
        self,
        ctx: TargetContext,
        token_id: str,
        condition_id: str,
        outcome: str,
        target_size: Decimal,
        target_price: Decimal,
    ) -> CopyTradeDecision:
        """Evaluate whether to copy an entry trade.

        Args:
            ctx: Target context
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            target_size: Target's trade size
            target_price: Target's execution price

        Returns:
            CopyTradeDecision
        """
        # Get current order book
        try:
            order_book = await self.clob.get_order_book(token_id)
            current_price = order_book.best_ask if order_book else None
        except Exception as e:
            logger.warning("failed_to_get_orderbook", error=str(e))
            order_book = None
            current_price = None

        # Check slippage
        slippage_result = ctx.slippage_calculator.check_slippage(
            target_price=target_price,
            order_book=order_book,
            side=OrderSide.BUY,
            size=target_size,
        )

        if slippage_result.status == SlippageStatus.NO_LIQUIDITY:
            return CopyTradeDecision(
                should_copy=False,
                reason="no_liquidity",
                slippage_result=slippage_result,
            )

        if slippage_result.status == SlippageStatus.EXCEEDED:
            if slippage_result.recommended_action == SlippageAction.SKIP:
                self.stats.orders_rejected_slippage += 1
                return CopyTradeDecision(
                    should_copy=False,
                    reason="slippage_exceeded",
                    slippage_result=slippage_result,
                )

        # Calculate size
        our_existing = self._our_positions.get(token_id, Decimal("0"))
        sizing_result = ctx.sizer.calculate_size(
            target_size=target_size,
            current_price=slippage_result.execution_price or current_price or target_price,
            available_balance=self.our_balance,
            existing_position_size=our_existing,
            is_exit=False,
        )

        if not sizing_result.approved:
            self.stats.orders_rejected_size += 1
            return CopyTradeDecision(
                should_copy=False,
                reason=f"size_rejected: {sizing_result.rejection_reason.value}",
                sizing_result=sizing_result,
                slippage_result=slippage_result,
            )

        # Build order
        execution_price = slippage_result.execution_price or current_price or target_price

        order = self.order_builder.build_market_buy(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            size=sizing_result.final_size,
            price=execution_price,
            target_name=ctx.target.name,
            target_price=target_price,
        )

        self.stats.orders_created += 1

        return CopyTradeDecision(
            should_copy=True,
            reason="approved",
            order=order,
            sizing_result=sizing_result,
            slippage_result=slippage_result,
        )

    async def _evaluate_exit(
        self,
        ctx: TargetContext,
        token_id: str,
        condition_id: str,
        outcome: str,
        target_exit_size: Decimal,
        target_remaining: Decimal,
        exit_price: Decimal,
    ) -> CopyTradeDecision:
        """Evaluate whether to copy an exit trade.

        Args:
            ctx: Target context
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            target_exit_size: How much target is selling
            target_remaining: Target's remaining position
            exit_price: Target's exit price

        Returns:
            CopyTradeDecision
        """
        our_position = self._our_positions.get(token_id, Decimal("0"))

        if our_position <= 0:
            return CopyTradeDecision(
                should_copy=False,
                reason="no_position_to_exit",
            )

        # Get current order book
        try:
            order_book = await self.clob.get_order_book(token_id)
            current_price = order_book.best_bid if order_book else None
        except Exception as e:
            logger.warning("failed_to_get_orderbook", error=str(e))
            order_book = None
            current_price = None

        # Check slippage for sell
        slippage_result = ctx.slippage_calculator.check_slippage(
            target_price=exit_price,
            order_book=order_book,
            side=OrderSide.SELL,
        )

        # For exits, we may be more lenient on slippage
        if slippage_result.status == SlippageStatus.EXCEEDED:
            if slippage_result.recommended_action == SlippageAction.SKIP:
                self.stats.orders_rejected_slippage += 1
                return CopyTradeDecision(
                    should_copy=False,
                    reason="exit_slippage_exceeded",
                    slippage_result=slippage_result,
                )

        # Calculate proportional exit size
        total_target_position = target_exit_size + target_remaining
        sizing_result = ctx.sizer.calculate_exit_size(
            target_exit_size=target_exit_size,
            our_position_size=our_position,
            target_total_position=total_target_position,
            current_price=slippage_result.execution_price or current_price or exit_price,
        )

        if not sizing_result.approved:
            self.stats.orders_rejected_size += 1
            return CopyTradeDecision(
                should_copy=False,
                reason=f"exit_size_rejected: {sizing_result.rejection_reason.value}",
                sizing_result=sizing_result,
                slippage_result=slippage_result,
            )

        # Build sell order
        execution_price = slippage_result.execution_price or current_price or exit_price

        order = self.order_builder.build_market_sell(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            size=sizing_result.final_size,
            price=execution_price,
            target_name=ctx.target.name,
            target_price=exit_price,
        )

        self.stats.orders_created += 1

        return CopyTradeDecision(
            should_copy=True,
            reason="approved",
            order=order,
            sizing_result=sizing_result,
            slippage_result=slippage_result,
        )

    async def _evaluate_full_exit(
        self,
        ctx: TargetContext,
        token_id: str,
        condition_id: str,
        outcome: str,
        exit_price: Decimal,
    ) -> CopyTradeDecision:
        """Evaluate full position exit.

        Args:
            ctx: Target context
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            exit_price: Target's exit price

        Returns:
            CopyTradeDecision
        """
        our_position = self._our_positions.get(token_id, Decimal("0"))

        if our_position <= 0:
            return CopyTradeDecision(
                should_copy=False,
                reason="no_position_to_exit",
            )

        # Get current order book
        try:
            order_book = await self.clob.get_order_book(token_id)
            current_price = order_book.best_bid if order_book else exit_price
        except Exception as e:
            logger.warning("failed_to_get_orderbook", error=str(e))
            current_price = exit_price

        # For full exits, we generally want to execute
        # Just check if there's any liquidity
        if current_price is None:
            return CopyTradeDecision(
                should_copy=False,
                reason="no_bid_liquidity",
            )

        # Build sell order for full position
        order = self.order_builder.build_market_sell(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            size=our_position,
            price=current_price,
            target_name=ctx.target.name,
            target_price=exit_price,
        )

        self.stats.orders_created += 1

        return CopyTradeDecision(
            should_copy=True,
            reason="full_exit",
            order=order,
        )

    async def _process_decision(
        self,
        decision: CopyTradeDecision,
        priority: OrderPriority,
    ) -> None:
        """Process a copy trade decision.

        Args:
            decision: The decision
            priority: Order priority
        """
        if not decision.should_copy:
            logger.info(
                "copy_trade_rejected",
                reason=decision.reason,
                details=decision.to_dict(),
            )
            return

        if decision.order is None:
            return

        # Queue the order for execution
        queued = await self.queue.enqueue(decision.order, priority)

        if queued:
            self.stats.orders_queued += 1
            logger.info(
                "copy_trade_queued",
                order_id=decision.order.order_id,
                side=decision.order.side.value,
                size=str(decision.order.size),
                price=str(decision.order.price),
                priority=priority.name,
            )
        else:
            self.stats.orders_failed_to_queue += 1
            logger.warning(
                "copy_trade_queue_failed",
                order_id=decision.order.order_id,
            )


class CopyTradeOrchestrator:
    """Orchestrates the complete copy trading system.

    Combines:
    - Copy trade handler (event processing)
    - Execution queue (order management)
    - Order executor (order submission)
    """

    def __init__(
        self,
        event_bus: EventBus,
        clob_client: CLOBClient,
        executor: OrderExecutor,
        targets: list[TargetAccount],
        initial_balance: Decimal,
        max_concurrent_orders: int = 3,
    ):
        """Initialize the orchestrator.

        Args:
            event_bus: Event bus
            clob_client: CLOB client
            executor: Order executor
            targets: Target accounts
            initial_balance: Starting balance
            max_concurrent_orders: Max parallel order execution
        """
        self.executor = executor

        # Create execution queue
        self.queue = ExecutionQueue(
            max_concurrent=max_concurrent_orders,
            max_queue_size=100,
        )

        # Create copy trade handler
        self.handler = CopyTradeHandler(
            event_bus=event_bus,
            clob_client=clob_client,
            execution_queue=self.queue,
            targets=targets,
            our_balance=initial_balance,
        )

        self._running = False

    async def start(self) -> None:
        """Start the copy trading system."""
        if self._running:
            return

        self._running = True

        # Start execution queue with executor callback
        await self.queue.start(self._execute_order)

        # Start handler
        await self.handler.start()

        logger.info("copy_trade_orchestrator_started")

    async def stop(self) -> None:
        """Stop the copy trading system."""
        if not self._running:
            return

        self._running = False

        await self.handler.stop()
        await self.queue.stop()

        logger.info("copy_trade_orchestrator_stopped")

    async def _execute_order(self, order: CopyOrder) -> bool:
        """Execute an order.

        Args:
            order: Order to execute

        Returns:
            True if execution succeeded
        """
        result = await self.executor.execute(order)

        # Update our position tracking based on result
        if result.is_success and result.filled_size > 0:
            current = self.handler._our_positions.get(order.token_id, Decimal("0"))

            if order.is_buy:
                new_size = current + result.filled_size
            else:
                new_size = max(Decimal("0"), current - result.filled_size)

            self.handler.update_our_position(order.token_id, new_size)

            # Adjust balance estimate
            if order.is_buy:
                cost = result.filled_size * (result.filled_price or order.price)
                self.handler.update_balance(self.handler.our_balance - cost)
            else:
                proceeds = result.filled_size * (result.filled_price or order.price)
                self.handler.update_balance(self.handler.our_balance + proceeds)

        return result.is_success

    def get_stats(self) -> dict:
        """Get combined statistics."""
        return {
            "handler": self.handler.stats.to_dict(),
            "queue": self.queue.stats.to_dict(),
            "executor": self.executor.stats.__dict__,
        }
