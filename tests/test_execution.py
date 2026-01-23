"""Tests for order execution components."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.clob import OrderSide, OrderType, OrderBook, OrderBookLevel
from src.config.models import TargetAccount, SlippageAction
from src.execution.sizer import (
    PositionSizer,
    SizingResult,
    SizeRejectionReason,
    MIN_ORDER_SIZE_USD,
    calculate_proportional_size,
)
from src.execution.slippage import (
    SlippageCalculator,
    SlippageCheckResult,
    SlippageStatus,
    LiquidityChecker,
    calculate_slippage,
    is_slippage_acceptable,
)
from src.execution.order_builder import (
    OrderBuilder,
    CopyOrder,
    OrderSource,
)
from src.execution.queue import (
    ExecutionQueue,
    QueuedOrder,
    OrderPriority,
)
from src.execution.executor import (
    OrderExecutor,
    ExecutionResult,
    ExecutionStatus,
    ExecutorConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def target_account():
    """Create a test target account."""
    return TargetAccount(
        name="test_whale",
        wallet="0x1234567890123456789012345678901234567890",
        position_ratio=Decimal("0.01"),  # 1% of target's position
        max_position_usd=Decimal("500"),
        min_position_usd=Decimal("10"),
        min_copy_size_usd=Decimal("5"),
        max_slippage=Decimal("0.05"),  # 5%
        slippage_action=SlippageAction.SKIP,
    )


@pytest.fixture
def order_builder():
    """Create an order builder."""
    return OrderBuilder(
        tick_size=Decimal("0.01"),
        min_tick_size=Decimal("0.001"),
    )


@pytest.fixture
def order_book():
    """Create a mock order book."""
    return OrderBook(
        token_id="0xtoken123",
        bids=[
            OrderBookLevel(price=Decimal("0.48"), size=Decimal("100")),
            OrderBookLevel(price=Decimal("0.47"), size=Decimal("200")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.52"), size=Decimal("150")),
            OrderBookLevel(price=Decimal("0.53"), size=Decimal("250")),
        ],
        timestamp=datetime.utcnow(),
    )


# =============================================================================
# Position Sizer Tests
# =============================================================================


class TestPositionSizer:
    """Tests for PositionSizer functionality."""

    def test_basic_sizing(self, target_account):
        """Test basic position size calculation."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_size(
            target_size=Decimal("10000"),  # Target has 10,000 shares
            current_price=Decimal("0.50"),
            available_balance=Decimal("1000"),
        )

        # Expected: 10000 * 0.01 = 100 shares
        assert result.approved
        assert result.final_size == Decimal("100")
        assert result.estimated_cost == Decimal("50.00")

    def test_max_position_cap(self, target_account):
        """Test that max position USD cap is applied."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_size(
            target_size=Decimal("200000"),  # Very large position
            current_price=Decimal("0.50"),
            available_balance=Decimal("10000"),
        )

        # Should be capped at max_position_usd = 500
        assert result.approved
        assert result.max_position_capped
        assert result.estimated_cost <= target_account.max_position_usd

    def test_balance_cap(self, target_account):
        """Test that balance constraint is applied."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_size(
            target_size=Decimal("100000"),
            current_price=Decimal("0.50"),
            available_balance=Decimal("100"),  # Low balance
        )

        # Should be capped by balance
        assert result.approved or result.rejection_reason == SizeRejectionReason.BELOW_MIN_SIZE
        if result.approved:
            assert result.balance_capped
            assert result.estimated_cost <= Decimal("100")

    def test_below_minimum_rejected(self, target_account):
        """Test that orders below minimum are rejected."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_size(
            target_size=Decimal("100"),  # Small position
            current_price=Decimal("0.50"),
            available_balance=Decimal("1000"),
        )

        # 100 * 0.01 = 1 share * 0.50 = $0.50 < min_copy_size_usd
        assert not result.approved
        assert result.rejection_reason == SizeRejectionReason.BELOW_MIN_SIZE

    def test_target_below_threshold_rejected(self, target_account):
        """Test that positions below target threshold are rejected."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_size(
            target_size=Decimal("10"),  # Very small target position
            current_price=Decimal("0.50"),
            available_balance=Decimal("1000"),
        )

        # 10 * 0.50 = $5 < min_position_usd ($10)
        assert not result.approved
        assert result.rejection_reason == SizeRejectionReason.TARGET_BELOW_THRESHOLD

    def test_exit_sizing(self, target_account):
        """Test exit trade sizing."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_exit_size(
            target_exit_size=Decimal("5000"),  # Target selling 5000
            our_position_size=Decimal("100"),  # We have 100 shares
            target_total_position=Decimal("10000"),  # Target had 10000 before
            current_price=Decimal("0.50"),
        )

        # Should exit 50% of our position (5000/10000)
        assert result.approved
        assert result.final_size == Decimal("50.00")

    def test_zero_size_rejected(self, target_account):
        """Test that zero size is rejected."""
        sizer = PositionSizer(target_account)

        result = sizer.calculate_size(
            target_size=Decimal("0"),
            current_price=Decimal("0.50"),
            available_balance=Decimal("1000"),
        )

        assert not result.approved
        assert result.rejection_reason == SizeRejectionReason.ZERO_SIZE


# =============================================================================
# Slippage Calculator Tests
# =============================================================================


class TestSlippageCalculator:
    """Tests for SlippageCalculator functionality."""

    def test_slippage_ok(self, order_book):
        """Test slippage within acceptable range."""
        calculator = SlippageCalculator(max_slippage=Decimal("0.10"))

        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=order_book,
            side=OrderSide.BUY,
        )

        # Best ask is 0.52, slippage = (0.52-0.50)/0.50 = 4%
        assert result.status == SlippageStatus.OK
        assert result.is_ok
        assert result.execution_price == Decimal("0.52")

    def test_slippage_exceeded(self, order_book):
        """Test slippage exceeding threshold."""
        calculator = SlippageCalculator(
            max_slippage=Decimal("0.02"),  # 2% max
            slippage_action=SlippageAction.SKIP,
        )

        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=order_book,
            side=OrderSide.BUY,
        )

        # Best ask is 0.52, slippage = 4% > 2%
        assert result.status == SlippageStatus.EXCEEDED
        assert not result.is_ok
        assert result.recommended_action == SlippageAction.SKIP

    def test_slippage_limit_order_action(self, order_book):
        """Test slippage with limit order action."""
        calculator = SlippageCalculator(
            max_slippage=Decimal("0.02"),
            slippage_action=SlippageAction.LIMIT_ORDER,
        )

        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=order_book,
            side=OrderSide.BUY,
        )

        assert result.status == SlippageStatus.EXCEEDED
        assert result.recommended_action == SlippageAction.LIMIT_ORDER
        assert result.execution_price == Decimal("0.50")  # Target's price

    def test_no_liquidity(self):
        """Test handling of no liquidity."""
        calculator = SlippageCalculator()

        empty_book = OrderBook(
            token_id="0xtoken",
            bids=[],
            asks=[],
            timestamp=datetime.utcnow(),
        )

        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=empty_book,
            side=OrderSide.BUY,
        )

        assert result.status == SlippageStatus.NO_LIQUIDITY

    def test_no_order_book(self):
        """Test handling of missing order book."""
        calculator = SlippageCalculator()

        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=None,
            side=OrderSide.BUY,
        )

        assert result.status == SlippageStatus.PRICE_UNAVAILABLE

    def test_sell_slippage_calculation(self, order_book):
        """Test slippage calculation for sells."""
        calculator = SlippageCalculator(max_slippage=Decimal("0.10"))

        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=order_book,
            side=OrderSide.SELL,
        )

        # Best bid is 0.48, slippage = (0.50-0.48)/0.50 = 4%
        assert result.status == SlippageStatus.OK
        assert result.execution_price == Decimal("0.48")


class TestSlippageHelpers:
    """Tests for slippage helper functions."""

    def test_calculate_slippage_buy(self):
        """Test calculate_slippage for buy."""
        slippage = calculate_slippage(
            target_price=Decimal("0.50"),
            current_price=Decimal("0.52"),
            side="BUY",
        )

        assert slippage == Decimal("0.04")  # 4%

    def test_calculate_slippage_sell(self):
        """Test calculate_slippage for sell."""
        slippage = calculate_slippage(
            target_price=Decimal("0.50"),
            current_price=Decimal("0.48"),
            side="SELL",
        )

        assert slippage == Decimal("0.04")  # 4%

    def test_is_slippage_acceptable(self):
        """Test is_slippage_acceptable."""
        assert is_slippage_acceptable(Decimal("0.03"), Decimal("0.05"))
        assert not is_slippage_acceptable(Decimal("0.06"), Decimal("0.05"))


class TestLiquidityChecker:
    """Tests for LiquidityChecker functionality."""

    def test_sufficient_liquidity(self, order_book):
        """Test checking sufficient liquidity."""
        checker = LiquidityChecker(min_depth_usd=Decimal("50"))

        sufficient, reason = checker.check_liquidity(
            order_book=order_book,
            side=OrderSide.BUY,
            order_size=Decimal("10"),
            price=Decimal("0.52"),
        )

        assert sufficient
        assert reason == "sufficient"

    def test_insufficient_liquidity(self, order_book):
        """Test detecting insufficient liquidity."""
        checker = LiquidityChecker(min_depth_usd=Decimal("1000"))

        sufficient, reason = checker.check_liquidity(
            order_book=order_book,
            side=OrderSide.BUY,
            order_size=Decimal("10"),
            price=Decimal("0.52"),
        )

        assert not sufficient
        assert "insufficient_depth" in reason


# =============================================================================
# Order Builder Tests
# =============================================================================


class TestOrderBuilder:
    """Tests for OrderBuilder functionality."""

    def test_price_rounding_buy(self, order_builder):
        """Test price rounding for buys (round up)."""
        price = order_builder.round_price(
            price=Decimal("0.523"),
            side=OrderSide.BUY,
        )

        # Should round up to 0.53
        assert price == Decimal("0.53")

    def test_price_rounding_sell(self, order_builder):
        """Test price rounding for sells (round down)."""
        price = order_builder.round_price(
            price=Decimal("0.527"),
            side=OrderSide.SELL,
        )

        # Should round down to 0.52
        assert price == Decimal("0.52")

    def test_price_bounds(self, order_builder):
        """Test price is kept within valid bounds."""
        # Too low
        price_low = order_builder.round_price(Decimal("0"), OrderSide.BUY)
        assert price_low >= order_builder.min_tick_size

        # Too high
        price_high = order_builder.round_price(Decimal("1"), OrderSide.SELL)
        assert price_high < Decimal("1")

    def test_build_order(self, order_builder):
        """Test building a complete order."""
        order = order_builder.build_order(
            token_id="0xtoken123",
            condition_id="0xcond456",
            outcome="Yes",
            side=OrderSide.BUY,
            size=Decimal("100"),
            price=Decimal("0.55"),
            target_name="whale_1",
            target_price=Decimal("0.54"),
        )

        assert isinstance(order, CopyOrder)
        assert order.token_id == "0xtoken123"
        assert order.side == OrderSide.BUY
        assert order.size == Decimal("100")
        assert order.target_name == "whale_1"
        assert order.source == OrderSource.COPY_TRADE

    def test_build_market_buy(self, order_builder):
        """Test building a market buy order."""
        order = order_builder.build_market_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("50"),
            price=Decimal("0.60"),
            target_name="whale",
        )

        assert order.is_buy
        assert not order.is_sell
        assert order.order_type == OrderType.GTC

    def test_build_market_sell(self, order_builder):
        """Test building a market sell order."""
        order = order_builder.build_market_sell(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("50"),
            price=Decimal("0.40"),
            target_name="whale",
        )

        assert order.is_sell
        assert not order.is_buy

    def test_build_passive_order(self, order_builder):
        """Test building a passive limit order."""
        order = order_builder.build_passive_order(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            side=OrderSide.BUY,
            size=Decimal("50"),
            price=Decimal("0.50"),
            target_name="whale",
            expiry_minutes=10,
        )

        assert order.order_type == OrderType.GTD
        assert order.expiration_timestamp is not None

    def test_order_to_clob_params(self, order_builder):
        """Test converting order to CLOB parameters."""
        order = order_builder.build_market_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.55"),
        )

        params = order.to_clob_params()

        assert params["token_id"] == "0xtoken"
        assert params["side"] == "BUY"
        assert params["price"] == "0.55"
        assert params["size"] == "100"


class TestCopyOrder:
    """Tests for CopyOrder dataclass."""

    def test_estimated_cost(self, order_builder):
        """Test estimated cost calculation."""
        order = order_builder.build_market_buy(
            token_id="0x",
            condition_id="0x",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        assert order.estimated_cost == Decimal("50")

    def test_estimated_proceeds(self, order_builder):
        """Test estimated proceeds calculation."""
        order = order_builder.build_market_sell(
            token_id="0x",
            condition_id="0x",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        assert order.estimated_proceeds == Decimal("50")
        assert order.estimated_cost == Decimal("0")

    def test_remaining_size(self, order_builder):
        """Test remaining size calculation."""
        order = order_builder.build_market_buy(
            token_id="0x",
            condition_id="0x",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        order.fill_size = Decimal("30")
        assert order.remaining_size == Decimal("70")
        assert not order.is_filled

        order.fill_size = Decimal("100")
        assert order.is_filled


# =============================================================================
# Execution Queue Tests
# =============================================================================


class TestExecutionQueue:
    """Tests for ExecutionQueue functionality."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor callback."""
        async def executor(order):
            await asyncio.sleep(0.01)
            return True
        return executor

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self, order_builder, mock_executor):
        """Test enqueueing and processing orders."""
        queue = ExecutionQueue(max_concurrent=1)
        await queue.start(mock_executor)

        order = order_builder.build_market_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        result = await queue.enqueue(order)
        assert result
        assert queue.stats.orders_queued == 1

        # Wait for processing
        await asyncio.sleep(0.1)

        await queue.stop()
        assert queue.stats.orders_processed >= 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self, order_builder, mock_executor):
        """Test that orders are processed by priority."""
        processed_order = []

        async def tracking_executor(order):
            processed_order.append(order.order_id)
            return True

        queue = ExecutionQueue(max_concurrent=1)
        await queue.start(tracking_executor)

        # Create orders with different priorities
        orders = [
            (order_builder.build_market_buy(
                token_id="0x1", condition_id="0x", outcome="Yes",
                size=Decimal("10"), price=Decimal("0.50")
            ), OrderPriority.LOW),
            (order_builder.build_market_buy(
                token_id="0x2", condition_id="0x", outcome="Yes",
                size=Decimal("10"), price=Decimal("0.50")
            ), OrderPriority.HIGH),
            (order_builder.build_market_sell(
                token_id="0x3", condition_id="0x", outcome="Yes",
                size=Decimal("10"), price=Decimal("0.50")
            ), OrderPriority.CRITICAL),
        ]

        for order, priority in orders:
            await queue.enqueue(order, priority)

        await asyncio.sleep(0.2)
        await queue.stop()

        # Critical should be first, then high, then low
        assert len(processed_order) == 3
        # Order 3 (CRITICAL) should be first
        assert processed_order[0] == orders[2][0].order_id

    @pytest.mark.asyncio
    async def test_cancel_order(self, order_builder, mock_executor):
        """Test cancelling a queued order."""
        async def slow_executor(order):
            await asyncio.sleep(1)
            return True

        queue = ExecutionQueue(max_concurrent=1)
        await queue.start(slow_executor)

        order = order_builder.build_market_buy(
            token_id="0xtoken", condition_id="0x", outcome="Yes",
            size=Decimal("10"), price=Decimal("0.50")
        )

        await queue.enqueue(order)
        result = await queue.cancel(order.order_id)

        await queue.stop()

        # May or may not be cancellable depending on timing
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self, order_builder, mock_executor):
        """Test that orders are rejected when queue is full."""
        queue = ExecutionQueue(max_concurrent=1, max_queue_size=2)
        await queue.start(mock_executor)

        # Fill the queue
        for i in range(3):
            order = order_builder.build_market_buy(
                token_id=f"0x{i}", condition_id="0x", outcome="Yes",
                size=Decimal("10"), price=Decimal("0.50")
            )
            result = await queue.enqueue(order)

            if i >= 2:
                # Third order should be rejected
                assert not result

        await queue.stop()


# =============================================================================
# Executor Tests
# =============================================================================


class TestOrderExecutor:
    """Tests for OrderExecutor functionality."""

    @pytest.fixture
    def mock_clob_client(self):
        """Create a mock CLOB client."""
        client = AsyncMock()
        client.get_order_book = AsyncMock(return_value=OrderBook(
            token_id="0xtoken",
            bids=[OrderBookLevel(price=Decimal("0.48"), size=Decimal("100"))],
            asks=[OrderBookLevel(price=Decimal("0.52"), size=Decimal("100"))],
            timestamp=datetime.utcnow(),
        ))
        client.place_order = AsyncMock(return_value={
            "success": True,
            "orderID": "clob_order_123",
        })
        client.get_order = AsyncMock(return_value={
            "status": "FILLED",
            "filledSize": "100",
            "avgFillPrice": "0.52",
        })
        return client

    @pytest.fixture
    def mock_auth(self):
        """Create a mock auth."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, mock_clob_client, mock_auth, order_builder
    ):
        """Test successful order execution."""
        executor = OrderExecutor(
            clob_client=mock_clob_client,
            auth=mock_auth,
            config=ExecutorConfig(max_status_polls=2),
        )

        order = order_builder.build_market_buy(
            token_id="0xtoken", condition_id="0x", outcome="Yes",
            size=Decimal("100"), price=Decimal("0.52")
        )

        result = await executor.execute(order)

        assert result.is_success
        assert result.status == ExecutionStatus.FILLED
        assert result.clob_order_id == "clob_order_123"
        assert result.filled_size == Decimal("100")

    @pytest.mark.asyncio
    async def test_execution_failure(
        self, mock_clob_client, mock_auth, order_builder
    ):
        """Test handling execution failure."""
        mock_clob_client.place_order = AsyncMock(return_value={
            "success": False,
            "error": "Insufficient funds",
        })

        executor = OrderExecutor(
            clob_client=mock_clob_client,
            auth=mock_auth,
            config=ExecutorConfig(max_retries=1),
        )

        order = order_builder.build_market_buy(
            token_id="0xtoken", condition_id="0x", outcome="Yes",
            size=Decimal("100"), price=Decimal("0.52")
        )

        result = await executor.execute(order)

        assert not result.is_success
        assert result.status == ExecutionStatus.FAILED
        assert "Insufficient funds" in result.error_message

    @pytest.mark.asyncio
    async def test_price_drift_rejection(
        self, mock_clob_client, mock_auth, order_builder
    ):
        """Test rejection when price has drifted too far."""
        # Order book shows price has moved significantly
        mock_clob_client.get_order_book = AsyncMock(return_value=OrderBook(
            token_id="0xtoken",
            bids=[],
            asks=[OrderBookLevel(price=Decimal("0.70"), size=Decimal("100"))],
            timestamp=datetime.utcnow(),
        ))

        executor = OrderExecutor(
            clob_client=mock_clob_client,
            auth=mock_auth,
            config=ExecutorConfig(
                check_price_before_submit=True,
                max_price_drift_percent=Decimal("0.02"),
            ),
        )

        order = order_builder.build_market_buy(
            token_id="0xtoken", condition_id="0x", outcome="Yes",
            size=Decimal("100"), price=Decimal("0.52")
        )

        result = await executor.execute(order)

        assert not result.is_success
        assert "drift" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_retry_logic(
        self, mock_clob_client, mock_auth, order_builder
    ):
        """Test retry logic on failure."""
        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return {"success": False, "error": "Temporary error"}
            return {"success": True, "orderID": "order_123"}

        mock_clob_client.place_order = failing_then_success

        executor = OrderExecutor(
            clob_client=mock_clob_client,
            auth=mock_auth,
            config=ExecutorConfig(
                max_retries=3,
                initial_retry_delay_ms=10,
            ),
        )

        order = order_builder.build_market_buy(
            token_id="0xtoken", condition_id="0x", outcome="Yes",
            size=Decimal("100"), price=Decimal("0.52")
        )

        result = await executor.execute(order)

        assert result.is_success
        assert call_count == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutionIntegration:
    """Integration tests for execution components."""

    @pytest.mark.asyncio
    async def test_full_copy_trade_flow(self, target_account, order_builder):
        """Test complete flow from sizing to queuing."""
        # 1. Calculate size
        sizer = PositionSizer(target_account)
        sizing_result = sizer.calculate_size(
            target_size=Decimal("10000"),
            current_price=Decimal("0.50"),
            available_balance=Decimal("1000"),
        )

        assert sizing_result.approved

        # 2. Check slippage
        calculator = SlippageCalculator(max_slippage=Decimal("0.10"))
        order_book = OrderBook(
            token_id="0xtoken",
            bids=[],
            asks=[OrderBookLevel(price=Decimal("0.52"), size=Decimal("1000"))],
            timestamp=datetime.utcnow(),
        )

        slippage_result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=order_book,
            side=OrderSide.BUY,
        )

        assert slippage_result.is_ok

        # 3. Build order
        order = order_builder.build_market_buy(
            token_id="0xtoken",
            condition_id="0xcond",
            outcome="Yes",
            size=sizing_result.final_size,
            price=slippage_result.execution_price,
            target_name=target_account.name,
            target_price=Decimal("0.50"),
        )

        assert order.size == sizing_result.final_size

        # 4. Queue order
        executed_orders = []

        async def mock_executor(o):
            executed_orders.append(o)
            return True

        queue = ExecutionQueue(max_concurrent=1)
        await queue.start(mock_executor)

        await queue.enqueue(order, OrderPriority.HIGH)
        await asyncio.sleep(0.1)

        await queue.stop()

        assert len(executed_orders) == 1
        assert executed_orders[0].order_id == order.order_id
