"""Execution module for PolymarketBot.

This module handles all order execution logic:
- Position sizing
- Slippage calculation and protection
- Order building
- Execution queue management
- Order execution with retries
- Copy trade handling
"""

from .sizer import (
    PositionSizer,
    SizingResult,
    SizeRejectionReason,
    MIN_ORDER_SIZE_USD,
    MIN_ORDER_SIZE_SHARES,
    calculate_proportional_size,
)
from .slippage import (
    SlippageCalculator,
    SlippageCheckResult,
    SlippageStatus,
    LiquidityChecker,
    calculate_slippage,
    is_slippage_acceptable,
)
from .order_builder import (
    OrderBuilder,
    CopyOrder,
    OrderSource,
)
from .queue import (
    ExecutionQueue,
    BatchExecutionQueue,
    QueuedOrder,
    OrderPriority,
    QueueStats,
)
from .executor import (
    OrderExecutor,
    ExecutorPool,
    ExecutionResult,
    ExecutionStatus,
    ExecutorConfig,
    ExecutorStats,
)
from .handler import (
    CopyTradeHandler,
    CopyTradeOrchestrator,
    CopyTradeDecision,
    HandlerStats,
)

__all__ = [
    # Sizer
    "PositionSizer",
    "SizingResult",
    "SizeRejectionReason",
    "MIN_ORDER_SIZE_USD",
    "MIN_ORDER_SIZE_SHARES",
    "calculate_proportional_size",
    # Slippage
    "SlippageCalculator",
    "SlippageCheckResult",
    "SlippageStatus",
    "LiquidityChecker",
    "calculate_slippage",
    "is_slippage_acceptable",
    # Order builder
    "OrderBuilder",
    "CopyOrder",
    "OrderSource",
    # Queue
    "ExecutionQueue",
    "BatchExecutionQueue",
    "QueuedOrder",
    "OrderPriority",
    "QueueStats",
    # Executor
    "OrderExecutor",
    "ExecutorPool",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutorConfig",
    "ExecutorStats",
    # Handler
    "CopyTradeHandler",
    "CopyTradeOrchestrator",
    "CopyTradeDecision",
    "HandlerStats",
]
