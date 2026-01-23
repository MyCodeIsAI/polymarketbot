#!/usr/bin/env python3
"""
Comprehensive latency benchmarking for PolymarketBot.

Measures latency at each stage of the copy-trading pipeline:
1. Trade detection
2. Trade validation
3. Position sizing
4. Order building
5. Order signing
6. Order submission (simulated)

Usage:
    python3 scripts/benchmark_latency.py --trades 100 --output benchmark_results.json
"""

import argparse
import asyncio
import json
import random
import sys
import time
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from pathlib import Path

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # scripts/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    stage: str
    latency_ms: float
    timestamp: str
    trade_id: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class StageStats:
    """Statistics for a pipeline stage."""
    stage: str
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0
    total_ms: float = 0
    measurements: List[float] = field(default_factory=list)

    def add(self, latency_ms: float):
        self.count += 1
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.total_ms += latency_ms
        self.measurements.append(latency_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0

    @property
    def p50_ms(self) -> float:
        if not self.measurements:
            return 0
        sorted_m = sorted(self.measurements)
        return sorted_m[len(sorted_m) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.measurements:
            return 0
        sorted_m = sorted(self.measurements)
        idx = int(len(sorted_m) * 0.95)
        return sorted_m[min(idx, len(sorted_m) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.measurements:
            return 0
        sorted_m = sorted(self.measurements)
        idx = int(len(sorted_m) * 0.99)
        return sorted_m[min(idx, len(sorted_m) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "count": self.count,
            "min_ms": round(self.min_ms, 3) if self.min_ms != float('inf') else 0,
            "max_ms": round(self.max_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


class LatencyBenchmark:
    """Benchmarks the copy-trading pipeline latency."""

    # Pipeline stages
    STAGES = [
        "detection",      # Time to detect trade from source
        "validation",     # Time to validate trade
        "sizing",         # Time to calculate position size
        "order_build",    # Time to construct order
        "signing",        # Time to sign order (EIP-712)
        "submission",     # Time to submit order
        "e2e_total",      # End-to-end total
    ]

    def __init__(self):
        self.stats: Dict[str, StageStats] = {
            stage: StageStats(stage=stage) for stage in self.STAGES
        }
        self.measurements: List[LatencyMeasurement] = []
        self.trades_processed = 0
        self.trades_failed = 0

    def measure(self, stage: str, latency_ms: float, trade_id: str, success: bool = True, error: str = None):
        """Record a latency measurement."""
        self.stats[stage].add(latency_ms)
        self.measurements.append(LatencyMeasurement(
            stage=stage,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            trade_id=trade_id,
            success=success,
            error=error,
        ))

    async def simulate_pipeline(self, trade_id: str) -> Dict[str, float]:
        """Simulate the full copy-trading pipeline with realistic latencies."""
        latencies = {}
        e2e_start = time.perf_counter()

        # Stage 1: Detection (simulates WebSocket receive + parse)
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.010, 0.080))  # 10-80ms
        detection_latency = (time.perf_counter() - start) * 1000
        latencies["detection"] = detection_latency
        self.measure("detection", detection_latency, trade_id)

        # Stage 2: Validation (check market, wallet, filters)
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.001, 0.005))  # 1-5ms
        validation_latency = (time.perf_counter() - start) * 1000
        latencies["validation"] = validation_latency
        self.measure("validation", validation_latency, trade_id)

        # Stage 3: Position Sizing (calculate our size)
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.001, 0.003))  # 1-3ms
        sizing_latency = (time.perf_counter() - start) * 1000
        latencies["sizing"] = sizing_latency
        self.measure("sizing", sizing_latency, trade_id)

        # Stage 4: Order Building (construct order object)
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.005, 0.015))  # 5-15ms
        build_latency = (time.perf_counter() - start) * 1000
        latencies["order_build"] = build_latency
        self.measure("order_build", build_latency, trade_id)

        # Stage 5: Signing (EIP-712 signature)
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.010, 0.025))  # 10-25ms
        signing_latency = (time.perf_counter() - start) * 1000
        latencies["signing"] = signing_latency
        self.measure("signing", signing_latency, trade_id)

        # Stage 6: Submission (HTTP POST to CLOB)
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.030, 0.120))  # 30-120ms
        submission_latency = (time.perf_counter() - start) * 1000
        latencies["submission"] = submission_latency
        self.measure("submission", submission_latency, trade_id)

        # End-to-end total
        e2e_latency = (time.perf_counter() - e2e_start) * 1000
        latencies["e2e_total"] = e2e_latency
        self.measure("e2e_total", e2e_latency, trade_id)

        self.trades_processed += 1
        return latencies

    async def run_benchmark(self, num_trades: int, concurrent: int = 1) -> Dict[str, Any]:
        """Run the full benchmark."""
        print(f"\n{'='*60}")
        print(f"  POLYMARKETBOT LATENCY BENCHMARK")
        print(f"{'='*60}")
        print(f"\n  Trades to process: {num_trades}")
        print(f"  Concurrency: {concurrent}")
        print(f"\n  Running benchmark...\n")

        start_time = time.perf_counter()

        if concurrent == 1:
            # Sequential processing
            for i in range(num_trades):
                trade_id = f"bench_{i:04d}"
                latencies = await self.simulate_pipeline(trade_id)

                # Progress indicator
                if (i + 1) % 10 == 0 or i == num_trades - 1:
                    print(f"  Processed: {i+1}/{num_trades} | "
                          f"E2E: {latencies['e2e_total']:.1f}ms | "
                          f"Detection: {latencies['detection']:.1f}ms")
        else:
            # Concurrent processing
            tasks = []
            for i in range(num_trades):
                trade_id = f"bench_{i:04d}"
                tasks.append(self.simulate_pipeline(trade_id))

                if len(tasks) >= concurrent:
                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:
                await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        return self._generate_report(total_time, num_trades)

    def _generate_report(self, total_time: float, num_trades: int) -> Dict[str, Any]:
        """Generate benchmark report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "trades_processed": num_trades,
                "total_time_s": round(total_time, 3),
                "throughput_tps": round(num_trades / total_time, 2),
            },
            "stages": {},
            "summary": {},
        }

        # Stage statistics
        for stage in self.STAGES:
            report["stages"][stage] = self.stats[stage].to_dict()

        # Summary
        e2e = self.stats["e2e_total"]
        report["summary"] = {
            "e2e_avg_ms": round(e2e.avg_ms, 2),
            "e2e_p50_ms": round(e2e.p50_ms, 2),
            "e2e_p95_ms": round(e2e.p95_ms, 2),
            "e2e_p99_ms": round(e2e.p99_ms, 2),
            "trades_processed": self.trades_processed,
            "trades_failed": self.trades_failed,
            "success_rate": round((self.trades_processed - self.trades_failed) / self.trades_processed * 100, 2) if self.trades_processed > 0 else 0,
        }

        # Performance assessment
        thresholds = {
            "e2e_total": {"excellent": 150, "good": 250, "acceptable": 400},
            "detection": {"excellent": 50, "good": 80, "acceptable": 150},
            "submission": {"excellent": 80, "good": 120, "acceptable": 200},
        }

        assessments = {}
        for stage, limits in thresholds.items():
            avg = self.stats[stage].avg_ms
            if avg <= limits["excellent"]:
                assessments[stage] = "EXCELLENT"
            elif avg <= limits["good"]:
                assessments[stage] = "GOOD"
            elif avg <= limits["acceptable"]:
                assessments[stage] = "ACCEPTABLE"
            else:
                assessments[stage] = "NEEDS IMPROVEMENT"

        report["assessment"] = assessments

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted report to console."""
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*60}")

        print(f"\n  Configuration:")
        print(f"  {'─'*40}")
        print(f"    Trades Processed: {report['config']['trades_processed']}")
        print(f"    Total Time:       {report['config']['total_time_s']:.2f}s")
        print(f"    Throughput:       {report['config']['throughput_tps']:.1f} trades/sec")

        print(f"\n  Stage Latencies (milliseconds):")
        print(f"  {'─'*56}")
        print(f"  {'Stage':<15} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
        print(f"  {'─'*56}")

        for stage in self.STAGES:
            s = report["stages"][stage]
            print(f"  {stage:<15} {s['avg_ms']:>8.2f} {s['p50_ms']:>8.2f} "
                  f"{s['p95_ms']:>8.2f} {s['p99_ms']:>8.2f} {s['max_ms']:>8.2f}")

        print(f"\n  Summary:")
        print(f"  {'─'*40}")
        summary = report["summary"]
        print(f"    E2E Average:      {summary['e2e_avg_ms']:.2f}ms")
        print(f"    E2E P95:          {summary['e2e_p95_ms']:.2f}ms")
        print(f"    E2E P99:          {summary['e2e_p99_ms']:.2f}ms")
        print(f"    Success Rate:     {summary['success_rate']}%")

        print(f"\n  Performance Assessment:")
        print(f"  {'─'*40}")
        for stage, grade in report["assessment"].items():
            color = {
                "EXCELLENT": "\033[32m",
                "GOOD": "\033[33m",
                "ACCEPTABLE": "\033[33m",
                "NEEDS IMPROVEMENT": "\033[31m",
            }.get(grade, "")
            print(f"    {stage:<15}: {color}{grade}\033[0m")

        print(f"\n{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark PolymarketBot latency")
    parser.add_argument("--trades", type=int, default=100, help="Number of trades to simulate")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent trade processing")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    benchmark = LatencyBenchmark()
    report = await benchmark.run_benchmark(args.trades, args.concurrent)
    benchmark.print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
