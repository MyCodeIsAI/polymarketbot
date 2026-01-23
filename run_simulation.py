#!/usr/bin/env python3
"""
Standalone simulation runner for real-time monitoring.

Simulates copy-trading activity from the automatedaitradingbot wallet.
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.harness import SimulationHarness, SimulationConfig, SimulationMode
from src.simulation.trade_generator import TradePattern


# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def format_usd(value: float) -> str:
    """Format value as USD."""
    return f"${value:,.2f}"


def format_wallet(wallet: str) -> str:
    """Format wallet address."""
    return f"{wallet[:6]}...{wallet[-4:]}"


async def run_simulation():
    """Run the copy trading simulation."""

    # Configuration - simulating automatedaitradingbot
    target_wallet = "0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11"

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  PolymarketBot Copy-Trading Simulation{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    print(f"  {Colors.CYAN}Target Wallet:{Colors.RESET} {format_wallet(target_wallet)}")
    print(f"  {Colors.CYAN}Username:{Colors.RESET}      @automatedaitradingbot")
    print(f"  {Colors.CYAN}Mode:{Colors.RESET}          Synthetic (realistic pattern)")
    print(f"  {Colors.CYAN}Duration:{Colors.RESET}      60 seconds")
    print(f"  {Colors.CYAN}Trade Rate:{Colors.RESET}    ~20 trades/hour")
    print()

    config = SimulationConfig(
        mode=SimulationMode.SYNTHETIC,
        target_wallets=[target_wallet],
        trades_per_hour=20,  # Simulate active trader
        min_trade_size_usd=50.0,
        max_trade_size_usd=2000.0,
        buy_probability=0.55,  # Slightly bullish
        simulated_latency_ms=75,
        simulated_slippage_pct=0.8,
        failure_probability=0.03,
        log_all_trades=False,  # We'll log manually
        collect_metrics=True,
    )

    harness = SimulationHarness(config)

    # Stats tracking
    stats = {
        "detected": 0,
        "executed": 0,
        "failed": 0,
        "total_volume": 0.0,
        "total_slippage": 0.0,
    }

    print(f"{Colors.BOLD}Starting simulation...{Colors.RESET}")
    print(f"{Colors.DIM}{'─'*60}{Colors.RESET}\n")
    print(f"  {'Time':<12} {'Side':<6} {'Size':<12} {'Price':<10} {'Market':<30} {'Status'}")
    print(f"{Colors.DIM}  {'─'*12} {'─'*6} {'─'*12} {'─'*10} {'─'*30} {'─'*10}{Colors.RESET}")

    async def on_trade_detected(trade):
        """Called when a trade is detected."""
        stats["detected"] += 1

        # Format trade info
        time_str = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        side_color = Colors.GREEN if trade.side == "BUY" else Colors.RED
        side_str = f"{side_color}{trade.side:<6}{Colors.RESET}"
        size_str = format_usd(float(trade.size))
        price_str = f"{float(trade.price):.4f}"
        market_str = trade.market_name[:28] + ".." if len(trade.market_name) > 30 else trade.market_name

        print(f"  {Colors.DIM}{time_str}{Colors.RESET} {side_str} {size_str:<12} {price_str:<10} {market_str:<30} {Colors.YELLOW}detecting...{Colors.RESET}", end="\r")

    async def on_trade_executed(trade):
        """Called when a trade is executed."""
        time_str = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        side_color = Colors.GREEN if trade.side == "BUY" else Colors.RED
        side_str = f"{side_color}{trade.side:<6}{Colors.RESET}"
        size_str = format_usd(float(trade.size))
        price_str = f"{float(trade.price):.4f}"
        market_str = trade.market_name[:28] + ".." if len(trade.market_name) > 30 else trade.market_name

        if trade.status == "executed":
            stats["executed"] += 1
            stats["total_volume"] += float(trade.size) * float(trade.price)
            if trade.slippage:
                stats["total_slippage"] += float(trade.slippage)

            slippage_pct = float(trade.slippage) * 100 if trade.slippage else 0
            slippage_color = Colors.GREEN if slippage_pct < 0.5 else Colors.YELLOW if slippage_pct < 1.0 else Colors.RED
            status_str = f"{Colors.GREEN}executed{Colors.RESET} {slippage_color}({slippage_pct:.2f}% slip){Colors.RESET}"
        else:
            stats["failed"] += 1
            status_str = f"{Colors.RED}failed{Colors.RESET}"

        # Clear line and print final status
        print(f"  {Colors.DIM}{time_str}{Colors.RESET} {side_str} {size_str:<12} {price_str:<10} {market_str:<30} {status_str}          ")

    harness.on_trade_detected(on_trade_detected)
    harness.on_trade_executed(on_trade_executed)

    # Run simulation
    await harness.start()

    # Run for 60 seconds with periodic stats updates
    start_time = datetime.utcnow()
    duration_s = 60

    try:
        while (datetime.utcnow() - start_time).total_seconds() < duration_s:
            await asyncio.sleep(1)

            # Print periodic stats every 15 seconds
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if int(elapsed) % 15 == 0 and int(elapsed) > 0:
                remaining = duration_s - elapsed
                print(f"\n  {Colors.DIM}[{elapsed:.0f}s elapsed, {remaining:.0f}s remaining | "
                      f"Detected: {stats['detected']} | Executed: {stats['executed']} | "
                      f"Volume: {format_usd(stats['total_volume'])}]{Colors.RESET}\n")

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Simulation interrupted.{Colors.RESET}")

    # Stop and get final metrics
    metrics = await harness.stop()

    # Print final results
    print(f"\n{Colors.DIM}{'─'*60}{Colors.RESET}")
    print(f"\n{Colors.BOLD}{Colors.GREEN}Simulation Complete{Colors.RESET}\n")

    results = metrics.to_dict()

    print(f"  {Colors.BOLD}Results Summary:{Colors.RESET}")
    print(f"  {'─'*40}")
    print(f"  {'Trades Detected:':<25} {results.get('trades_detected', 0)}")
    print(f"  {'Trades Executed:':<25} {results.get('trades_executed', 0)}")
    print(f"  {'Trades Failed:':<25} {results.get('trades_failed', 0)}")
    print(f"  {'Success Rate:':<25} {results.get('success_rate_pct', 0):.1f}%")
    print()
    print(f"  {Colors.BOLD}Latency Metrics:{Colors.RESET}")
    print(f"  {'─'*40}")
    print(f"  {'Avg Detection Latency:':<25} {results.get('avg_detection_latency_ms', 0):.1f} ms")
    print(f"  {'Avg Execution Latency:':<25} {results.get('avg_execution_latency_ms', 0):.1f} ms")
    print(f"  {'Avg Slippage:':<25} {results.get('avg_slippage_pct', 0):.3f}%")
    print()
    print(f"  {Colors.BOLD}Volume:{Colors.RESET}")
    print(f"  {'─'*40}")
    print(f"  {'Total Simulated Volume:':<25} {format_usd(stats['total_volume'])}")
    print()

    # Performance assessment
    avg_latency = results.get('avg_detection_latency_ms', 0) + results.get('avg_execution_latency_ms', 0)
    if avg_latency < 100:
        latency_grade = f"{Colors.GREEN}Excellent{Colors.RESET}"
    elif avg_latency < 200:
        latency_grade = f"{Colors.YELLOW}Good{Colors.RESET}"
    else:
        latency_grade = f"{Colors.RED}Needs Improvement{Colors.RESET}"

    avg_slip = results.get('avg_slippage_pct', 0)
    if avg_slip < 0.5:
        slip_grade = f"{Colors.GREEN}Excellent{Colors.RESET}"
    elif avg_slip < 1.0:
        slip_grade = f"{Colors.YELLOW}Acceptable{Colors.RESET}"
    else:
        slip_grade = f"{Colors.RED}High{Colors.RESET}"

    print(f"  {Colors.BOLD}Performance Assessment:{Colors.RESET}")
    print(f"  {'─'*40}")
    print(f"  {'Latency Grade:':<25} {latency_grade}")
    print(f"  {'Slippage Grade:':<25} {slip_grade}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(run_simulation())
