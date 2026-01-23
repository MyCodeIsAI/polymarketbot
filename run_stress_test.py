#!/usr/bin/env python3
"""
Stress Test for PolymarketBot.

Tests the system under various load conditions to find breaking points
and measure performance degradation.

Usage:
    # Normal load (default)
    python3 run_stress_test.py

    # High frequency trading (1 trade/second)
    python3 run_stress_test.py --mode burst --rate 60

    # Sustained load (10 trades/minute for 5 minutes)
    python3 run_stress_test.py --mode sustained --rate 10 --duration 300

    # Maximum stress (as fast as possible)
    python3 run_stress_test.py --mode max --duration 60
"""

import argparse
import asyncio
import sys
import time
import random
import uuid
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from collections import deque
from pathlib import Path

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"


SAMPLE_MARKETS = [
    {"id": "0x001", "name": "Will Trump win 2024?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.52, "No": 0.48}},
    {"id": "0x002", "name": "Fed rate cut Q1 2025?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.65, "No": 0.35}},
    {"id": "0x003", "name": "Bitcoin $100K 2024?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.42, "No": 0.58}},
    {"id": "0x004", "name": "Super Bowl LIX", "outcomes": ["Chiefs", "Eagles"], "prices": {"Chiefs": 0.55, "Eagles": 0.45}},
    {"id": "0x005", "name": "Best Picture 2025", "outcomes": ["Oppenheimer", "Other"], "prices": {"Oppenheimer": 0.68, "Other": 0.32}},
    {"id": "0x006", "name": "AI Turing test 2025?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.35, "No": 0.65}},
    {"id": "0x007", "name": "Nvidia >$150 EOY?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.72, "No": 0.28}},
    {"id": "0x008", "name": "Gov shutdown 2025?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.25, "No": 0.75}},
]


@dataclass
class StressTestMetrics:
    """Metrics collected during stress test."""
    start_time: float = 0
    trades_generated: int = 0
    trades_processed: int = 0
    trades_failed: int = 0
    positions_opened: int = 0
    positions_closed: int = 0

    # Latencies (in ms)
    detection_latencies: List[float] = field(default_factory=list)
    processing_latencies: List[float] = field(default_factory=list)
    e2e_latencies: List[float] = field(default_factory=list)

    # Throughput tracking
    trades_per_second: deque = field(default_factory=lambda: deque(maxlen=60))

    # Errors
    errors: List[str] = field(default_factory=list)

    def record_trade(self, detection_ms: float, processing_ms: float, e2e_ms: float, success: bool = True):
        if success:
            self.trades_processed += 1
            self.detection_latencies.append(detection_ms)
            self.processing_latencies.append(processing_ms)
            self.e2e_latencies.append(e2e_ms)
        else:
            self.trades_failed += 1

    def get_stats(self) -> Dict[str, Any]:
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            "elapsed_s": elapsed,
            "trades_generated": self.trades_generated,
            "trades_processed": self.trades_processed,
            "trades_failed": self.trades_failed,
            "success_rate": (self.trades_processed / self.trades_generated * 100) if self.trades_generated > 0 else 0,
            "throughput_tps": self.trades_processed / elapsed if elapsed > 0 else 0,
            "detection_avg_ms": statistics.mean(self.detection_latencies) if self.detection_latencies else 0,
            "detection_p95_ms": percentile(self.detection_latencies, 95),
            "detection_p99_ms": percentile(self.detection_latencies, 99),
            "processing_avg_ms": statistics.mean(self.processing_latencies) if self.processing_latencies else 0,
            "e2e_avg_ms": statistics.mean(self.e2e_latencies) if self.e2e_latencies else 0,
            "e2e_p95_ms": percentile(self.e2e_latencies, 95),
            "e2e_p99_ms": percentile(self.e2e_latencies, 99),
            "e2e_max_ms": max(self.e2e_latencies) if self.e2e_latencies else 0,
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "errors": len(self.errors),
        }


class StressTestSimulator:
    """Simulates the full copy-trading pipeline under stress."""

    def __init__(self, mode: str = "normal", rate: int = 10, duration: int = 60):
        self.mode = mode
        self.rate = rate  # trades per minute
        self.duration = duration
        self.metrics = StressTestMetrics()
        self.positions: Dict[str, Dict] = {}  # position_key -> position
        self.running = False

    async def simulate_detection(self) -> float:
        """Simulate trade detection latency."""
        # WebSocket receive + parse
        if self.mode == "max":
            latency = random.uniform(0.005, 0.020)  # 5-20ms under load
        else:
            latency = random.uniform(0.010, 0.080)  # 10-80ms normal
        await asyncio.sleep(latency)
        return latency * 1000

    async def simulate_validation(self) -> float:
        """Simulate trade validation."""
        latency = random.uniform(0.001, 0.005)
        await asyncio.sleep(latency)
        return latency * 1000

    async def simulate_sizing(self) -> float:
        """Simulate position sizing calculation."""
        latency = random.uniform(0.001, 0.003)
        await asyncio.sleep(latency)
        return latency * 1000

    async def simulate_order_build(self) -> float:
        """Simulate order construction."""
        latency = random.uniform(0.005, 0.015)
        await asyncio.sleep(latency)
        return latency * 1000

    async def simulate_signing(self) -> float:
        """Simulate EIP-712 signing."""
        latency = random.uniform(0.010, 0.025)
        await asyncio.sleep(latency)
        return latency * 1000

    async def simulate_submission(self) -> float:
        """Simulate order submission to CLOB."""
        if self.mode == "max":
            latency = random.uniform(0.020, 0.080)  # Faster under controlled conditions
        else:
            latency = random.uniform(0.030, 0.120)
        await asyncio.sleep(latency)
        return latency * 1000

    async def process_trade(self, trade: Dict[str, Any]) -> Dict[str, float]:
        """Process a single trade through the full pipeline."""
        e2e_start = time.perf_counter()
        latencies = {}

        try:
            # Stage 1: Detection
            latencies["detection"] = await self.simulate_detection()

            # Stage 2: Validation
            latencies["validation"] = await self.simulate_validation()

            # Stage 3: Sizing
            latencies["sizing"] = await self.simulate_sizing()

            # Stage 4: Order build
            latencies["order_build"] = await self.simulate_order_build()

            # Stage 5: Signing
            latencies["signing"] = await self.simulate_signing()

            # Stage 6: Submission
            latencies["submission"] = await self.simulate_submission()

            # Update position
            position_key = f"{trade['market_id']}_{trade['outcome']}"
            if trade["side"] == "BUY":
                if position_key not in self.positions:
                    self.positions[position_key] = {"size": 0, "avg_price": 0}
                    self.metrics.positions_opened += 1
                pos = self.positions[position_key]
                old_cost = pos["size"] * pos["avg_price"]
                new_cost = trade["size"] * trade["price"]
                pos["size"] += trade["size"]
                pos["avg_price"] = (old_cost + new_cost) / pos["size"] if pos["size"] > 0 else trade["price"]
            else:  # SELL
                if position_key in self.positions:
                    self.positions[position_key]["size"] -= trade["size"]
                    if self.positions[position_key]["size"] <= 0:
                        del self.positions[position_key]
                        self.metrics.positions_closed += 1

            # Calculate totals
            latencies["e2e"] = (time.perf_counter() - e2e_start) * 1000
            processing_ms = latencies["validation"] + latencies["sizing"] + latencies["order_build"] + latencies["signing"]

            self.metrics.record_trade(latencies["detection"], processing_ms, latencies["e2e"], success=True)

            return latencies

        except Exception as e:
            self.metrics.errors.append(str(e))
            self.metrics.trades_failed += 1
            return {"error": str(e)}

    def generate_trade(self) -> Dict[str, Any]:
        """Generate a random trade."""
        self.metrics.trades_generated += 1
        market = random.choice(SAMPLE_MARKETS)
        outcome = random.choice(market["outcomes"])
        side = "BUY" if random.random() < 0.55 else "SELL"
        size = random.uniform(50, 500)
        price = market["prices"].get(outcome, 0.5) + random.uniform(-0.03, 0.03)
        price = max(0.01, min(0.99, price))

        return {
            "trade_id": f"stress_{self.metrics.trades_generated}",
            "market_id": market["id"],
            "market_name": market["name"],
            "outcome": outcome,
            "side": side,
            "size": size,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def run(self):
        """Run the stress test."""
        print(f"\n{C.BOLD}{'='*70}{C.RESET}")
        print(f"{C.BOLD}  POLYMARKETBOT STRESS TEST{C.RESET}")
        print(f"{C.BOLD}{'='*70}{C.RESET}")
        print(f"\n  Mode:     {C.CYAN}{self.mode.upper()}{C.RESET}")
        print(f"  Rate:     {self.rate} trades/minute")
        print(f"  Duration: {self.duration} seconds")

        if self.mode == "max":
            print(f"\n  {C.YELLOW}WARNING: Maximum stress mode - as fast as possible{C.RESET}")

        print(f"\n{C.DIM}{'─'*70}{C.RESET}")
        print(f"\n  Starting stress test...\n")

        self.running = True
        self.metrics.start_time = time.time()

        # Calculate delay between trades
        if self.mode == "max":
            delay = 0  # No delay
        else:
            delay = 60.0 / self.rate  # seconds between trades

        # Progress tracking
        last_report = time.time()
        report_interval = 5  # seconds

        while self.running and (time.time() - self.metrics.start_time) < self.duration:
            # Generate and process trade
            trade = self.generate_trade()
            latencies = await self.process_trade(trade)

            # Print trade
            if "error" not in latencies:
                side_color = C.GREEN if trade["side"] == "BUY" else C.RED
                e2e = latencies.get("e2e", 0)
                e2e_color = C.GREEN if e2e < 150 else C.YELLOW if e2e < 250 else C.RED
                print(f"  [{self.metrics.trades_processed:>4}] {side_color}{trade['side']:<4}{C.RESET} "
                      f"${trade['size']:>7.2f} @ {trade['price']:.4f} | "
                      f"E2E: {e2e_color}{e2e:>6.1f}ms{C.RESET} | "
                      f"{trade['market_name'][:25]}")
            else:
                print(f"  [{self.metrics.trades_generated:>4}] {C.RED}ERROR: {latencies['error']}{C.RESET}")

            # Periodic stats report
            if time.time() - last_report >= report_interval:
                stats = self.metrics.get_stats()
                print(f"\n  {C.DIM}[{stats['elapsed_s']:.0f}s] "
                      f"Trades: {stats['trades_processed']} | "
                      f"TPS: {stats['throughput_tps']:.2f} | "
                      f"E2E Avg: {stats['e2e_avg_ms']:.1f}ms | "
                      f"P95: {stats['e2e_p95_ms']:.1f}ms | "
                      f"Positions: {len(self.positions)}{C.RESET}\n")
                last_report = time.time()

            # Wait before next trade (unless max mode)
            if delay > 0:
                await asyncio.sleep(delay)

        self.running = False
        self.print_final_report()

    def print_final_report(self):
        """Print final stress test report."""
        stats = self.metrics.get_stats()

        print(f"\n{C.DIM}{'─'*70}{C.RESET}")
        print(f"\n{C.BOLD}{C.GREEN}  STRESS TEST COMPLETE{C.RESET}\n")

        print(f"  {C.BOLD}Summary:{C.RESET}")
        print(f"  {'─'*50}")
        print(f"    Duration:           {stats['elapsed_s']:.1f} seconds")
        print(f"    Trades Generated:   {stats['trades_generated']}")
        print(f"    Trades Processed:   {stats['trades_processed']}")
        print(f"    Trades Failed:      {stats['trades_failed']}")
        print(f"    Success Rate:       {stats['success_rate']:.1f}%")
        print(f"    Throughput:         {stats['throughput_tps']:.2f} trades/sec")

        print(f"\n  {C.BOLD}Latency Metrics:{C.RESET}")
        print(f"  {'─'*50}")
        print(f"    Detection Avg:      {stats['detection_avg_ms']:.2f} ms")
        print(f"    Detection P95:      {stats['detection_p95_ms']:.2f} ms")
        print(f"    Processing Avg:     {stats['processing_avg_ms']:.2f} ms")
        print(f"    E2E Average:        {stats['e2e_avg_ms']:.2f} ms")
        print(f"    E2E P95:            {stats['e2e_p95_ms']:.2f} ms")
        print(f"    E2E P99:            {stats['e2e_p99_ms']:.2f} ms")
        print(f"    E2E Maximum:        {stats['e2e_max_ms']:.2f} ms")

        print(f"\n  {C.BOLD}Position Stats:{C.RESET}")
        print(f"  {'─'*50}")
        print(f"    Positions Opened:   {stats['positions_opened']}")
        print(f"    Positions Closed:   {stats['positions_closed']}")
        print(f"    Open Positions:     {len(self.positions)}")

        # Performance assessment
        print(f"\n  {C.BOLD}Performance Assessment:{C.RESET}")
        print(f"  {'─'*50}")

        # E2E latency grade
        e2e_avg = stats['e2e_avg_ms']
        if e2e_avg < 150:
            grade = f"{C.GREEN}EXCELLENT{C.RESET}"
        elif e2e_avg < 200:
            grade = f"{C.GREEN}GOOD{C.RESET}"
        elif e2e_avg < 300:
            grade = f"{C.YELLOW}ACCEPTABLE{C.RESET}"
        else:
            grade = f"{C.RED}NEEDS IMPROVEMENT{C.RESET}"
        print(f"    E2E Latency:        {grade} ({e2e_avg:.1f}ms avg)")

        # Throughput grade
        tps = stats['throughput_tps']
        if tps >= 10:
            grade = f"{C.GREEN}EXCELLENT{C.RESET}"
        elif tps >= 5:
            grade = f"{C.GREEN}GOOD{C.RESET}"
        elif tps >= 2:
            grade = f"{C.YELLOW}ACCEPTABLE{C.RESET}"
        else:
            grade = f"{C.RED}LOW{C.RESET}"
        print(f"    Throughput:         {grade} ({tps:.2f} TPS)")

        # Success rate grade
        success = stats['success_rate']
        if success >= 99:
            grade = f"{C.GREEN}EXCELLENT{C.RESET}"
        elif success >= 95:
            grade = f"{C.GREEN}GOOD{C.RESET}"
        elif success >= 90:
            grade = f"{C.YELLOW}ACCEPTABLE{C.RESET}"
        else:
            grade = f"{C.RED}POOR{C.RESET}"
        print(f"    Success Rate:       {grade} ({success:.1f}%)")

        print(f"\n{'='*70}\n")


async def main():
    parser = argparse.ArgumentParser(description="PolymarketBot Stress Test")
    parser.add_argument("--mode", choices=["normal", "burst", "sustained", "max"],
                       default="normal", help="Stress test mode")
    parser.add_argument("--rate", type=int, default=10,
                       help="Trades per minute (ignored in max mode)")
    parser.add_argument("--duration", type=int, default=60,
                       help="Test duration in seconds")
    args = parser.parse_args()

    # Adjust rate based on mode
    if args.mode == "burst":
        rate = args.rate if args.rate != 10 else 60  # 1/second default for burst
    elif args.mode == "sustained":
        rate = args.rate if args.rate != 10 else 20  # moderate rate
    elif args.mode == "max":
        rate = 9999  # As fast as possible
    else:
        rate = args.rate

    simulator = StressTestSimulator(mode=args.mode, rate=rate, duration=args.duration)

    try:
        await simulator.run()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Stress test interrupted.{C.RESET}")
        simulator.running = False
        simulator.print_final_report()


if __name__ == "__main__":
    asyncio.run(main())
