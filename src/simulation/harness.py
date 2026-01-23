"""Simulation harness for copy trading testing.

Provides a complete test environment that can:
- Connect to live Polymarket for dry-run testing
- Replay historical trades
- Generate synthetic trades
- Measure detection latency and execution quality
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable

from pydantic import BaseModel

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SimulationMode(str, Enum):
    """Simulation modes."""

    DRY_RUN = "dry_run"  # Real detection, no execution
    HISTORICAL = "historical"  # Replay historical trades
    SYNTHETIC = "synthetic"  # Generated mock trades
    HYBRID = "hybrid"  # Mix of real and synthetic


class SimulationConfig(BaseModel):
    """Configuration for simulation."""

    mode: SimulationMode = SimulationMode.DRY_RUN
    target_wallets: List[str] = []

    # Timing
    speed_multiplier: float = 1.0  # For historical replay
    start_delay_s: float = 0.0

    # Trade generation (synthetic mode)
    trades_per_hour: int = 10
    min_trade_size_usd: float = 10.0
    max_trade_size_usd: float = 1000.0
    buy_probability: float = 0.6

    # Execution simulation
    simulated_latency_ms: int = 50
    simulated_slippage_pct: float = 0.5
    failure_probability: float = 0.02

    # Metrics collection
    collect_metrics: bool = True
    metrics_interval_s: float = 10.0

    # Output
    log_all_trades: bool = True
    save_results_path: Optional[str] = None


@dataclass
class SimulatedTrade:
    """A simulated or detected trade."""

    trade_id: str
    wallet: str
    market_id: str
    market_name: str
    outcome: str
    side: str  # BUY or SELL
    size: Decimal
    price: Decimal

    # Timing
    detected_at: datetime
    original_timestamp: Optional[datetime] = None

    # Execution (if simulated)
    executed_at: Optional[datetime] = None
    execution_price: Optional[Decimal] = None
    execution_size: Optional[Decimal] = None
    slippage: Optional[Decimal] = None

    # Status
    status: str = "detected"  # detected, executing, executed, failed, skipped


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Trade counts
    trades_detected: int = 0
    trades_executed: int = 0
    trades_failed: int = 0
    trades_skipped: int = 0

    # Latency (ms)
    detection_latencies: List[float] = field(default_factory=list)
    execution_latencies: List[float] = field(default_factory=list)

    # Slippage
    slippages: List[float] = field(default_factory=list)

    # P&L
    simulated_pnl: Decimal = Decimal("0")

    def avg_detection_latency(self) -> float:
        """Average detection latency in ms."""
        if not self.detection_latencies:
            return 0.0
        return sum(self.detection_latencies) / len(self.detection_latencies)

    def avg_execution_latency(self) -> float:
        """Average execution latency in ms."""
        if not self.execution_latencies:
            return 0.0
        return sum(self.execution_latencies) / len(self.execution_latencies)

    def avg_slippage(self) -> float:
        """Average slippage percentage."""
        if not self.slippages:
            return 0.0
        return sum(self.slippages) / len(self.slippages)

    def success_rate(self) -> float:
        """Trade success rate."""
        total = self.trades_executed + self.trades_failed
        if total == 0:
            return 0.0
        return self.trades_executed / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_s": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "trades_detected": self.trades_detected,
            "trades_executed": self.trades_executed,
            "trades_failed": self.trades_failed,
            "trades_skipped": self.trades_skipped,
            "avg_detection_latency_ms": self.avg_detection_latency(),
            "avg_execution_latency_ms": self.avg_execution_latency(),
            "avg_slippage_pct": self.avg_slippage() * 100,
            "success_rate_pct": self.success_rate() * 100,
            "simulated_pnl": float(self.simulated_pnl),
        }


class SimulationHarness:
    """Main simulation harness for copy trading testing."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.metrics = SimulationMetrics()
        self.trades: List[SimulatedTrade] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Callbacks
        self._on_trade_detected: Optional[Callable[[SimulatedTrade], Awaitable[None]]] = None
        self._on_trade_executed: Optional[Callable[[SimulatedTrade], Awaitable[None]]] = None

        # Components (injected based on mode)
        self._trade_source = None
        self._executor = None

    def on_trade_detected(self, callback: Callable[[SimulatedTrade], Awaitable[None]]):
        """Register callback for trade detection."""
        self._on_trade_detected = callback

    def on_trade_executed(self, callback: Callable[[SimulatedTrade], Awaitable[None]]):
        """Register callback for trade execution."""
        self._on_trade_executed = callback

    async def start(self) -> None:
        """Start the simulation."""
        if self._running:
            return

        logger.info(
            "simulation_starting",
            mode=self.config.mode,
            target_wallets=self.config.target_wallets,
        )

        self._running = True
        self.metrics = SimulationMetrics()

        # Start delay if configured
        if self.config.start_delay_s > 0:
            await asyncio.sleep(self.config.start_delay_s)

        # Start based on mode
        if self.config.mode == SimulationMode.DRY_RUN:
            await self._start_dry_run()
        elif self.config.mode == SimulationMode.HISTORICAL:
            await self._start_historical()
        elif self.config.mode == SimulationMode.SYNTHETIC:
            await self._start_synthetic()
        elif self.config.mode == SimulationMode.HYBRID:
            await self._start_hybrid()

        # Start metrics collection
        if self.config.collect_metrics:
            self._tasks.append(asyncio.create_task(self._metrics_loop()))

    async def stop(self) -> SimulationMetrics:
        """Stop the simulation and return metrics."""
        if not self._running:
            return self.metrics

        logger.info("simulation_stopping")

        self._running = False
        self.metrics.end_time = datetime.utcnow()

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Save results if configured
        if self.config.save_results_path:
            await self._save_results()

        logger.info(
            "simulation_complete",
            **self.metrics.to_dict(),
        )

        return self.metrics

    async def _start_dry_run(self) -> None:
        """Start dry-run mode with real detection."""
        from .mock_websocket import MockWebSocketFeed

        # In dry-run, we use real WebSocket but simulate execution
        logger.info("dry_run_mode", wallets=self.config.target_wallets)

        # This would connect to real Polymarket WebSocket
        # For now, create a mock that can be replaced with real connection
        self._trade_source = MockWebSocketFeed(
            wallets=self.config.target_wallets,
            real_connection=True,  # Would use actual WebSocket
        )

        self._tasks.append(asyncio.create_task(self._trade_detection_loop()))

    async def _start_historical(self) -> None:
        """Start historical replay mode."""
        from .historical_replay import HistoricalReplay

        logger.info("historical_mode", wallets=self.config.target_wallets)

        self._trade_source = HistoricalReplay(
            wallets=self.config.target_wallets,
            speed_multiplier=self.config.speed_multiplier,
        )

        self._tasks.append(asyncio.create_task(self._trade_detection_loop()))

    async def _start_synthetic(self) -> None:
        """Start synthetic trade generation mode."""
        from .trade_generator import TradeGenerator

        logger.info("synthetic_mode")

        self._trade_source = TradeGenerator(
            wallets=self.config.target_wallets or ["0xSIMULATED"],
            trades_per_hour=self.config.trades_per_hour,
            min_size=self.config.min_trade_size_usd,
            max_size=self.config.max_trade_size_usd,
            buy_probability=self.config.buy_probability,
        )

        self._tasks.append(asyncio.create_task(self._trade_detection_loop()))

    async def _start_hybrid(self) -> None:
        """Start hybrid mode with real detection and synthetic fills."""
        logger.info("hybrid_mode")

        # Both real detection and synthetic generation
        await self._start_dry_run()

    async def _trade_detection_loop(self) -> None:
        """Main loop for processing trades from source."""
        try:
            async for trade_data in self._trade_source.trades():
                if not self._running:
                    break

                trade = self._create_trade(trade_data)
                await self._process_trade(trade)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("trade_detection_error", error=str(e))

    def _create_trade(self, data: Dict[str, Any]) -> SimulatedTrade:
        """Create SimulatedTrade from raw data."""
        now = datetime.utcnow()

        return SimulatedTrade(
            trade_id=data.get("trade_id", f"sim_{now.timestamp()}"),
            wallet=data.get("wallet", "unknown"),
            market_id=data.get("market_id", "unknown"),
            market_name=data.get("market_name", "Unknown Market"),
            outcome=data.get("outcome", "Yes"),
            side=data.get("side", "BUY"),
            size=Decimal(str(data.get("size", 0))),
            price=Decimal(str(data.get("price", 0.5))),
            detected_at=now,
            original_timestamp=data.get("timestamp"),
        )

    async def _process_trade(self, trade: SimulatedTrade) -> None:
        """Process a detected trade."""
        self.trades.append(trade)
        self.metrics.trades_detected += 1

        if self.config.log_all_trades:
            logger.info(
                "trade_detected",
                trade_id=trade.trade_id,
                wallet=trade.wallet[:10] + "...",
                market=trade.market_name[:30],
                side=trade.side,
                size=float(trade.size),
                price=float(trade.price),
            )

        # Calculate detection latency if we have original timestamp
        if trade.original_timestamp:
            latency_ms = (trade.detected_at - trade.original_timestamp).total_seconds() * 1000
            self.metrics.detection_latencies.append(latency_ms)

        # Fire callback
        if self._on_trade_detected:
            await self._on_trade_detected(trade)

        # Simulate execution
        await self._simulate_execution(trade)

    async def _simulate_execution(self, trade: SimulatedTrade) -> None:
        """Simulate trade execution."""
        import random

        # Simulate latency
        latency_s = self.config.simulated_latency_ms / 1000
        await asyncio.sleep(latency_s)

        # Check for simulated failure
        if random.random() < self.config.failure_probability:
            trade.status = "failed"
            self.metrics.trades_failed += 1
            logger.warning("simulated_trade_failed", trade_id=trade.trade_id)
            return

        # Calculate slippage
        slippage_pct = random.uniform(0, self.config.simulated_slippage_pct * 2) / 100
        if trade.side == "BUY":
            execution_price = trade.price * (1 + Decimal(str(slippage_pct)))
        else:
            execution_price = trade.price * (1 - Decimal(str(slippage_pct)))

        trade.executed_at = datetime.utcnow()
        trade.execution_price = execution_price
        trade.execution_size = trade.size
        trade.slippage = Decimal(str(slippage_pct))
        trade.status = "executed"

        self.metrics.trades_executed += 1
        self.metrics.execution_latencies.append(
            (trade.executed_at - trade.detected_at).total_seconds() * 1000
        )
        self.metrics.slippages.append(float(slippage_pct))

        if self.config.log_all_trades:
            logger.info(
                "trade_executed",
                trade_id=trade.trade_id,
                execution_price=float(execution_price),
                slippage_pct=slippage_pct * 100,
            )

        # Fire callback
        if self._on_trade_executed:
            await self._on_trade_executed(trade)

    async def _metrics_loop(self) -> None:
        """Periodic metrics reporting."""
        try:
            while self._running:
                await asyncio.sleep(self.config.metrics_interval_s)

                logger.info(
                    "simulation_metrics",
                    trades_detected=self.metrics.trades_detected,
                    trades_executed=self.metrics.trades_executed,
                    avg_latency_ms=self.metrics.avg_detection_latency(),
                    avg_slippage_pct=self.metrics.avg_slippage() * 100,
                )
        except asyncio.CancelledError:
            pass

    async def _save_results(self) -> None:
        """Save simulation results to file."""
        import json

        results = {
            "config": self.config.model_dump(),
            "metrics": self.metrics.to_dict(),
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "wallet": t.wallet,
                    "market": t.market_name,
                    "side": t.side,
                    "size": float(t.size),
                    "price": float(t.price),
                    "execution_price": float(t.execution_price) if t.execution_price else None,
                    "slippage": float(t.slippage) if t.slippage else None,
                    "status": t.status,
                    "detected_at": t.detected_at.isoformat(),
                    "executed_at": t.executed_at.isoformat() if t.executed_at else None,
                }
                for t in self.trades
            ],
        }

        with open(self.config.save_results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("results_saved", path=self.config.save_results_path)


async def run_simulation(
    config: SimulationConfig,
    duration_s: Optional[float] = None,
) -> SimulationMetrics:
    """Run a simulation with the given config.

    Args:
        config: Simulation configuration
        duration_s: Optional duration limit

    Returns:
        Simulation metrics
    """
    harness = SimulationHarness(config)
    await harness.start()

    if duration_s:
        await asyncio.sleep(duration_s)

    return await harness.stop()
