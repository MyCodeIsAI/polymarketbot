"""Tests for safety systems module."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.safety.circuit_breaker import (
    BreakerState,
    TripReason,
    TripEvent,
    StateSnapshot,
    BreakerCondition,
    CircuitBreaker,
    CompositeCircuitBreaker,
)
from src.safety.limits import (
    MaxDailyLoss,
    MaxDrawdown,
    MaxConsecutiveFailures,
    MaxSlippageEvents,
    APIErrorRate,
    BalanceTooLow,
    WebSocketDisconnected,
    RateLimitExceeded,
)
from src.safety.health_check import (
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
from src.safety.alerts import (
    AlertSeverity,
    AlertCategory,
    Alert,
    LoggingChannel,
    FileChannel,
    AlertThrottler,
    AlertManager,
)
from src.safety.shutdown import (
    ShutdownReason,
    ShutdownPhase,
    ShutdownResult,
    GracefulShutdown,
    EmergencyShutdown,
    ShutdownCoordinator,
)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

class TestTripEvent:
    """Tests for TripEvent dataclass."""

    def test_create_trip_event(self):
        """Test creating a trip event."""
        event = TripEvent(
            reason=TripReason.MAX_DAILY_LOSS,
            details="Loss exceeded $500",
        )

        assert event.reason == TripReason.MAX_DAILY_LOSS
        assert event.details == "Loss exceeded $500"
        assert event.auto_reset is False
        assert isinstance(event.timestamp, datetime)

    def test_to_dict(self):
        """Test converting trip event to dict."""
        event = TripEvent(
            reason=TripReason.MANUAL,
            details="Test",
            condition_name="test_condition",
            auto_reset=True,
            reset_after_s=60,
        )

        data = event.to_dict()
        assert data["reason"] == "manual"
        assert data["details"] == "Test"
        assert data["condition_name"] == "test_condition"
        assert data["auto_reset"] is True


class TestStateSnapshot:
    """Tests for StateSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            total_balance=Decimal("1000"),
            available_balance=Decimal("500"),
            open_positions=3,
            daily_pnl=Decimal("-50"),
        )

        assert snapshot.total_balance == Decimal("1000")
        assert snapshot.open_positions == 3

    def test_to_dict(self):
        """Test converting snapshot to dict."""
        snapshot = StateSnapshot(
            total_balance=Decimal("1000"),
            daily_pnl=Decimal("-50"),
        )

        data = snapshot.to_dict()
        assert data["balance"]["total"] == "1000"
        assert data["pnl"]["daily"] == "-50"


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker for testing."""
        return CircuitBreaker(check_interval_s=0.1)

    def test_initial_state(self, breaker):
        """Test initial state is closed."""
        assert breaker.state == BreakerState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False
        assert breaker.current_trip is None

    @pytest.mark.asyncio
    async def test_manual_trip(self, breaker):
        """Test manually tripping the breaker."""
        await breaker.trip(TripReason.MANUAL, "Test trip")

        assert breaker.is_open is True
        assert breaker.current_trip.reason == TripReason.MANUAL
        assert breaker.current_trip.details == "Test trip"

    @pytest.mark.asyncio
    async def test_trip_when_already_open(self, breaker):
        """Test tripping when already open does nothing."""
        await breaker.trip(TripReason.MANUAL, "First trip")
        await breaker.trip(TripReason.MAX_DAILY_LOSS, "Second trip")

        # Should still have first trip
        assert breaker.current_trip.reason == TripReason.MANUAL

    @pytest.mark.asyncio
    async def test_reset(self, breaker):
        """Test resetting the breaker."""
        await breaker.trip(TripReason.MANUAL)
        result = await breaker.reset()

        assert result is True
        assert breaker.is_closed is True

    @pytest.mark.asyncio
    async def test_force_reset(self, breaker):
        """Test force resetting the breaker."""
        await breaker.trip(TripReason.MANUAL)
        await breaker.force_reset()

        assert breaker.is_closed is True

    @pytest.mark.asyncio
    async def test_trip_callback(self, breaker):
        """Test trip callback is called."""
        callback = AsyncMock()
        breaker._on_trip = callback

        await breaker.trip(TripReason.MANUAL, "Test")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0].reason == TripReason.MANUAL

    @pytest.mark.asyncio
    async def test_snapshot_on_trip(self, breaker):
        """Test snapshot is taken on trip."""
        async def mock_snapshot():
            return StateSnapshot(
                total_balance=Decimal("1000"),
                daily_pnl=Decimal("-100"),
            )

        breaker._snapshot_provider = mock_snapshot

        await breaker.trip(TripReason.MANUAL)

        assert len(breaker.get_snapshots()) == 1
        assert breaker.get_snapshots()[0].total_balance == Decimal("1000")

    def test_add_condition(self, breaker):
        """Test adding conditions."""
        condition = MaxDailyLoss(
            threshold=Decimal("500"),
            get_daily_loss=lambda: Decimal("100"),
        )

        breaker.add_condition(condition)
        assert len(breaker._conditions) == 1

    def test_remove_condition(self, breaker):
        """Test removing conditions."""
        condition = MaxDailyLoss(
            threshold=Decimal("500"),
            get_daily_loss=lambda: Decimal("100"),
        )
        breaker.add_condition(condition)

        result = breaker.remove_condition("max_daily_loss")
        assert result is True
        assert len(breaker._conditions) == 0

    def test_get_status(self, breaker):
        """Test getting breaker status."""
        status = breaker.get_status()

        assert status["state"] == "closed"
        assert status["current_trip"] is None
        assert status["trip_count"] == 0


class TestCompositeCircuitBreaker:
    """Tests for CompositeCircuitBreaker class."""

    def test_add_and_get_breaker(self):
        """Test adding and getting breakers."""
        composite = CompositeCircuitBreaker()
        breaker = CircuitBreaker()

        composite.add_breaker("test", breaker)
        assert composite.get_breaker("test") is breaker
        assert composite.get_breaker("nonexistent") is None

    @pytest.mark.asyncio
    async def test_any_open(self):
        """Test any_open property."""
        composite = CompositeCircuitBreaker()
        breaker1 = CircuitBreaker()
        breaker2 = CircuitBreaker()

        composite.add_breaker("b1", breaker1)
        composite.add_breaker("b2", breaker2)

        assert composite.any_open is False
        assert composite.all_closed is True

        await breaker1.trip(TripReason.MANUAL)

        assert composite.any_open is True
        assert composite.all_closed is False


# =============================================================================
# Limits Tests
# =============================================================================

class TestMaxDailyLoss:
    """Tests for MaxDailyLoss condition."""

    @pytest.mark.asyncio
    async def test_not_triggered_below_threshold(self):
        """Test condition not triggered when below threshold."""
        condition = MaxDailyLoss(
            threshold=Decimal("500"),
            get_daily_loss=lambda: Decimal("100"),
        )

        assert await condition.is_triggered() is False

    @pytest.mark.asyncio
    async def test_triggered_at_threshold(self):
        """Test condition triggered when at threshold."""
        condition = MaxDailyLoss(
            threshold=Decimal("500"),
            get_daily_loss=lambda: Decimal("500"),
        )

        assert await condition.is_triggered() is True

    def test_get_details(self):
        """Test getting condition details."""
        condition = MaxDailyLoss(
            threshold=Decimal("500"),
            get_daily_loss=lambda: Decimal("600"),
        )

        details = condition.get_details()
        assert "600" in details
        assert "500" in details


class TestMaxConsecutiveFailures:
    """Tests for MaxConsecutiveFailures condition."""

    @pytest.mark.asyncio
    async def test_not_triggered_below_count(self):
        """Test condition not triggered when below count."""
        condition = MaxConsecutiveFailures(count=5)

        for _ in range(4):
            condition.record_failure()

        assert await condition.is_triggered() is False

    @pytest.mark.asyncio
    async def test_triggered_at_count(self):
        """Test condition triggered when at count."""
        condition = MaxConsecutiveFailures(count=5)

        for _ in range(5):
            condition.record_failure()

        assert await condition.is_triggered() is True

    @pytest.mark.asyncio
    async def test_success_resets_count(self):
        """Test success resets failure count."""
        condition = MaxConsecutiveFailures(count=5)

        for _ in range(4):
            condition.record_failure()

        condition.record_success()

        for _ in range(4):
            condition.record_failure()

        assert await condition.is_triggered() is False


class TestMaxSlippageEvents:
    """Tests for MaxSlippageEvents condition."""

    @pytest.mark.asyncio
    async def test_not_triggered_below_count(self):
        """Test condition not triggered when below count."""
        condition = MaxSlippageEvents(
            count=3,
            window_s=60,
            slippage_threshold=Decimal("0.02"),
        )

        condition.record_slippage(Decimal("0.03"))
        condition.record_slippage(Decimal("0.03"))

        assert await condition.is_triggered() is False

    @pytest.mark.asyncio
    async def test_triggered_at_count(self):
        """Test condition triggered when at count."""
        condition = MaxSlippageEvents(
            count=3,
            window_s=60,
            slippage_threshold=Decimal("0.02"),
        )

        for _ in range(3):
            condition.record_slippage(Decimal("0.03"))

        assert await condition.is_triggered() is True

    @pytest.mark.asyncio
    async def test_below_threshold_not_counted(self):
        """Test slippage below threshold not counted."""
        condition = MaxSlippageEvents(
            count=3,
            window_s=60,
            slippage_threshold=Decimal("0.02"),
        )

        for _ in range(5):
            condition.record_slippage(Decimal("0.01"))

        assert await condition.is_triggered() is False


class TestBalanceTooLow:
    """Tests for BalanceTooLow condition."""

    @pytest.mark.asyncio
    async def test_not_triggered_above_threshold(self):
        """Test condition not triggered when above threshold."""
        condition = BalanceTooLow(
            min_balance=Decimal("100"),
            get_balance=lambda: Decimal("200"),
        )

        assert await condition.is_triggered() is False

    @pytest.mark.asyncio
    async def test_triggered_at_threshold(self):
        """Test condition triggered when at threshold."""
        condition = BalanceTooLow(
            min_balance=Decimal("100"),
            get_balance=lambda: Decimal("50"),
        )

        assert await condition.is_triggered() is True


class TestWebSocketDisconnected:
    """Tests for WebSocketDisconnected condition."""

    @pytest.mark.asyncio
    async def test_not_triggered_when_connected(self):
        """Test condition not triggered when connected."""
        condition = WebSocketDisconnected(
            is_connected_func=lambda: True,
            grace_period_s=5,
        )

        assert await condition.is_triggered() is False

    @pytest.mark.asyncio
    async def test_triggered_after_grace_period(self):
        """Test condition triggered after grace period."""
        condition = WebSocketDisconnected(
            is_connected_func=lambda: False,
            grace_period_s=0,  # No grace period for test
        )

        # Force disconnect time in the past
        condition._disconnect_time = datetime.utcnow() - timedelta(seconds=1)

        assert await condition.is_triggered() is True


# =============================================================================
# Health Check Tests
# =============================================================================

class TestAPIHealthChecker:
    """Tests for APIHealthChecker class."""

    @pytest.mark.asyncio
    async def test_healthy_response(self):
        """Test healthy response when API is reachable."""
        async def ping():
            return True

        checker = APIHealthChecker(ping_func=ping)
        health = await checker.check()

        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms is not None

    @pytest.mark.asyncio
    async def test_unhealthy_response(self):
        """Test unhealthy response when API fails."""
        async def ping():
            return False

        checker = APIHealthChecker(ping_func=ping)
        health = await checker.check()

        assert health.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test timeout handling."""
        async def slow_ping():
            await asyncio.sleep(10)
            return True

        checker = APIHealthChecker(ping_func=slow_ping, timeout_ms=100)
        health = await checker.check()

        assert health.status == HealthStatus.UNHEALTHY
        assert "Timeout" in health.message


class TestWebSocketHealthChecker:
    """Tests for WebSocketHealthChecker class."""

    @pytest.mark.asyncio
    async def test_healthy_when_connected(self):
        """Test healthy when connected."""
        checker = WebSocketHealthChecker(
            is_connected_func=lambda: True,
        )
        health = await checker.check()

        assert health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_when_disconnected(self):
        """Test unhealthy when disconnected."""
        checker = WebSocketHealthChecker(
            is_connected_func=lambda: False,
        )
        health = await checker.check()

        assert health.status == HealthStatus.UNHEALTHY


class TestRateLimitHealthChecker:
    """Tests for RateLimitHealthChecker class."""

    @pytest.mark.asyncio
    async def test_healthy_below_threshold(self):
        """Test healthy when below warning threshold."""
        checker = RateLimitHealthChecker(
            get_usage_func=lambda: (50, 100),
        )
        health = await checker.check()

        assert health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_above_warning(self):
        """Test degraded when above warning threshold."""
        checker = RateLimitHealthChecker(
            get_usage_func=lambda: (85, 100),
            warning_threshold=0.8,
        )
        health = await checker.check()

        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_above_critical(self):
        """Test unhealthy when above critical threshold."""
        checker = RateLimitHealthChecker(
            get_usage_func=lambda: (96, 100),
            critical_threshold=0.95,
        )
        health = await checker.check()

        assert health.status == HealthStatus.UNHEALTHY


class TestHealthCheckManager:
    """Tests for HealthCheckManager class."""

    @pytest.mark.asyncio
    async def test_aggregate_healthy(self):
        """Test aggregation when all healthy."""
        manager = HealthCheckManager(check_interval_s=1)

        async def healthy_check():
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
            )

        manager.add_custom_check(healthy_check)

        health = await manager.get_health()
        assert health.overall_status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_aggregate_unhealthy(self):
        """Test aggregation when any unhealthy."""
        manager = HealthCheckManager(check_interval_s=1)

        async def healthy_check():
            return ComponentHealth(name="healthy", status=HealthStatus.HEALTHY)

        async def unhealthy_check():
            return ComponentHealth(name="unhealthy", status=HealthStatus.UNHEALTHY)

        manager.add_custom_check(healthy_check)
        manager.add_custom_check(unhealthy_check)

        health = await manager.get_health()
        assert health.overall_status == HealthStatus.UNHEALTHY
        assert len(health.unhealthy_components) == 1


# =============================================================================
# Alert Tests
# =============================================================================

class TestAlert:
    """Tests for Alert dataclass."""

    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            category=AlertCategory.CIRCUIT_BREAKER,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            message="Test message",
        )

        assert alert.category == AlertCategory.CIRCUIT_BREAKER
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.is_critical is True

    def test_to_dict(self):
        """Test converting alert to dict."""
        alert = Alert(
            category=AlertCategory.TRADE,
            severity=AlertSeverity.INFO,
            title="Trade",
            message="Buy executed",
        )

        data = alert.to_dict()
        assert data["category"] == "trade"
        assert data["severity"] == "info"

    def test_to_json(self):
        """Test converting alert to JSON."""
        alert = Alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Warning",
            message="Test",
        )

        json_str = alert.to_json()
        assert "system" in json_str
        assert "warning" in json_str


class TestAlertThrottler:
    """Tests for AlertThrottler class."""

    def test_allows_initial_alerts(self):
        """Test throttler allows initial alerts."""
        throttler = AlertThrottler(window_s=60, max_per_window=3)

        alert = Alert(
            category=AlertCategory.TRADE,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        assert throttler.should_send(alert) is True

    def test_throttles_after_limit(self):
        """Test throttler blocks after limit."""
        throttler = AlertThrottler(window_s=60, max_per_window=2)

        alert = Alert(
            category=AlertCategory.TRADE,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        assert throttler.should_send(alert) is True
        assert throttler.should_send(alert) is True
        assert throttler.should_send(alert) is False


class TestLoggingChannel:
    """Tests for LoggingChannel class."""

    def test_should_send_above_severity(self):
        """Test channel sends alerts above min severity."""
        channel = LoggingChannel(min_severity=AlertSeverity.WARNING)

        error_alert = Alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.ERROR,
            title="Error",
            message="Test",
        )

        info_alert = Alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            title="Info",
            message="Test",
        )

        assert channel.should_send(error_alert) is True
        assert channel.should_send(info_alert) is False


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending an alert."""
        manager = AlertManager(enable_throttling=False)
        manager.add_channel(LoggingChannel())

        alert = await manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test",
            "Test message",
        )

        assert alert is not None
        assert manager._alerts_sent == 1

    @pytest.mark.asyncio
    async def test_throttling(self):
        """Test alert throttling."""
        manager = AlertManager(throttle_window_s=60, throttle_max=2)

        # First two should succeed
        alert1 = await manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test 1",
            "Message",
        )
        alert2 = await manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test 2",
            "Message",
        )

        # Third should be throttled
        alert3 = await manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test 3",
            "Message",
        )

        assert alert1 is not None
        assert alert2 is not None
        assert alert3 is None
        assert manager._alerts_throttled == 1

    @pytest.mark.asyncio
    async def test_critical_bypasses_throttle(self):
        """Test critical alerts bypass throttle."""
        manager = AlertManager(throttle_window_s=60, throttle_max=1)

        # First alert
        await manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test",
            "Message",
        )

        # Second should be throttled
        alert = await manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test",
            "Message",
        )
        assert alert is None

        # Critical should bypass
        critical = await manager.critical(
            "system",
            "Critical message",
        )
        assert critical is not None

    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        """Test convenience methods."""
        manager = AlertManager(enable_throttling=False)

        info = await manager.info("system", "Info message")
        warning = await manager.warning("system", "Warning message")
        error = await manager.error("system", "Error message")

        assert info.severity == AlertSeverity.INFO
        assert warning.severity == AlertSeverity.WARNING
        assert error.severity == AlertSeverity.ERROR


# =============================================================================
# Shutdown Tests
# =============================================================================

class TestGracefulShutdown:
    """Tests for GracefulShutdown class."""

    @pytest.fixture
    def shutdown(self):
        """Create a shutdown manager for testing."""
        return GracefulShutdown(total_timeout_s=30)

    def test_initial_state(self, shutdown):
        """Test initial state."""
        assert shutdown.is_shutting_down is False
        assert shutdown.is_shutdown_complete is False

    @pytest.mark.asyncio
    async def test_register_and_execute_task(self, shutdown):
        """Test registering and executing shutdown tasks."""
        executed = []

        async def task1():
            executed.append("task1")

        async def task2():
            executed.append("task2")

        shutdown.register("task1", ShutdownPhase.STOP_NEW_ORDERS, task1)
        shutdown.register("task2", ShutdownPhase.CLEANUP, task2)

        result = await shutdown.initiate(ShutdownReason.USER_REQUEST)

        assert result.success is True
        assert "task1" in executed
        assert "task2" in executed
        assert "task1" in result.tasks_completed
        assert "task2" in result.tasks_completed

    @pytest.mark.asyncio
    async def test_task_timeout(self, shutdown):
        """Test task timeout handling."""
        async def slow_task():
            await asyncio.sleep(10)

        shutdown.register(
            "slow_task",
            ShutdownPhase.CLEANUP,
            slow_task,
            timeout_s=0.1,
        )

        result = await shutdown.initiate(ShutdownReason.USER_REQUEST)

        assert ("slow_task", "Timeout after 0.1s") in result.tasks_failed

    @pytest.mark.asyncio
    async def test_critical_task_failure_aborts(self, shutdown):
        """Test critical task failure aborts shutdown."""
        async def failing_task():
            raise Exception("Test failure")

        async def later_task():
            pass  # Should not run

        shutdown.register(
            "failing",
            ShutdownPhase.STOP_NEW_ORDERS,
            failing_task,
            critical=True,
        )
        shutdown.register(
            "later",
            ShutdownPhase.CLEANUP,
            later_task,
        )

        result = await shutdown.initiate(ShutdownReason.USER_REQUEST)

        assert result.aborted is True
        assert "later" not in result.tasks_completed

    @pytest.mark.asyncio
    async def test_callbacks(self, shutdown):
        """Test shutdown callbacks."""
        start_called = []
        complete_called = []

        async def on_start(reason):
            start_called.append(reason)

        async def on_complete(result):
            complete_called.append(result)

        shutdown._on_shutdown_start = on_start
        shutdown._on_shutdown_complete = on_complete

        await shutdown.initiate(ShutdownReason.MANUAL)

        assert len(start_called) == 1
        assert start_called[0] == ShutdownReason.MANUAL
        assert len(complete_called) == 1

    def test_get_status(self, shutdown):
        """Test getting shutdown status."""
        status = shutdown.get_status()

        assert status["in_progress"] is False
        assert status["complete"] is False


class TestEmergencyShutdown:
    """Tests for EmergencyShutdown class."""

    @pytest.mark.asyncio
    async def test_trigger(self):
        """Test triggering emergency shutdown."""
        halt_called = []
        cancel_called = []

        async def halt():
            halt_called.append(True)

        async def cancel():
            cancel_called.append(True)

        emergency = EmergencyShutdown(
            halt_trading_func=halt,
            cancel_all_func=cancel,
        )

        await emergency.trigger("Test emergency")

        assert emergency.is_triggered is True
        assert len(halt_called) == 1
        assert len(cancel_called) == 1

    @pytest.mark.asyncio
    async def test_double_trigger_ignored(self):
        """Test double trigger is ignored."""
        call_count = []

        async def halt():
            call_count.append(True)

        emergency = EmergencyShutdown(halt_trading_func=halt)

        await emergency.trigger("First")
        await emergency.trigger("Second")

        assert len(call_count) == 1


class TestShutdownCoordinator:
    """Tests for ShutdownCoordinator class."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown through coordinator."""
        coordinator = ShutdownCoordinator()

        executed = []

        async def task():
            executed.append(True)

        coordinator.graceful.register("task", ShutdownPhase.CLEANUP, task)

        result = await coordinator.shutdown()

        assert result.success is True
        assert len(executed) == 1

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self):
        """Test emergency shutdown through coordinator."""
        coordinator = ShutdownCoordinator()

        halted = []

        async def halt():
            halted.append(True)

        coordinator.setup_emergency(halt)

        await coordinator.emergency("Test")

        assert len(halted) == 1
        assert coordinator.is_shutting_down is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with conditions."""

    @pytest.mark.asyncio
    async def test_breaker_trips_on_condition(self):
        """Test breaker trips when condition is triggered."""
        breaker = CircuitBreaker(check_interval_s=0.05)

        # Add condition that will trigger
        condition = MaxDailyLoss(
            threshold=Decimal("100"),
            get_daily_loss=lambda: Decimal("150"),
        )
        breaker.add_condition(condition)

        await breaker.start()
        await asyncio.sleep(0.15)  # Allow time for check
        await breaker.stop()

        assert breaker.is_open is True
        assert breaker.current_trip.reason == TripReason.MAX_DAILY_LOSS

    @pytest.mark.asyncio
    async def test_breaker_with_multiple_conditions(self):
        """Test breaker with multiple conditions."""
        breaker = CircuitBreaker(check_interval_s=0.05)

        # Add conditions - first will trigger
        condition1 = BalanceTooLow(
            min_balance=Decimal("100"),
            get_balance=lambda: Decimal("50"),
        )
        condition2 = MaxDailyLoss(
            threshold=Decimal("500"),
            get_daily_loss=lambda: Decimal("100"),
        )

        breaker.add_condition(condition1)
        breaker.add_condition(condition2)

        await breaker.start()
        await asyncio.sleep(0.15)
        await breaker.stop()

        assert breaker.is_open is True
        assert breaker.current_trip.reason == TripReason.LOW_BALANCE


class TestAlertIntegration:
    """Integration tests for alert system."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_alert(self):
        """Test circuit breaker triggers alert."""
        alerts_received = []

        manager = AlertManager(enable_throttling=False)

        class TestChannel:
            async def send(self, alert):
                alerts_received.append(alert)
                return True

            def should_send(self, alert):
                return True

        manager.add_channel(TestChannel())

        await manager.circuit_breaker_tripped(
            reason="MAX_DAILY_LOSS",
            details="Loss exceeded $500",
            snapshot={"balance": "500"},
        )

        assert len(alerts_received) == 1
        assert alerts_received[0].category == AlertCategory.CIRCUIT_BREAKER
        assert alerts_received[0].severity == AlertSeverity.CRITICAL


class TestHealthAndShutdownIntegration:
    """Integration tests for health check and shutdown."""

    @pytest.mark.asyncio
    async def test_health_triggers_shutdown(self):
        """Test unhealthy status can trigger shutdown."""
        shutdown_triggered = []

        coordinator = ShutdownCoordinator()

        async def on_health_change(health):
            if not health.is_healthy:
                shutdown_triggered.append(True)
                await coordinator.shutdown(ShutdownReason.HEALTH_FAILURE)

        manager = HealthCheckManager(
            check_interval_s=0.1,
            on_status_change=on_health_change,
        )

        # Add unhealthy checker
        async def unhealthy():
            return ComponentHealth(
                name="test",
                status=HealthStatus.UNHEALTHY,
            )

        manager.add_custom_check(unhealthy)

        await manager.start()
        await asyncio.sleep(0.25)
        await manager.stop()

        assert len(shutdown_triggered) >= 1
