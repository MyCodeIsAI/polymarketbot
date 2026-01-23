"""Main CLI application for PolymarketBot.

Provides commands for:
- Bot lifecycle (start, stop, status)
- Account management
- Position and trade viewing
- P&L reporting
- Configuration and diagnostics
"""

import asyncio
import os
import signal
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm

from .formatters import (
    console,
    format_status,
    format_accounts,
    format_positions,
    format_trades,
    format_pnl,
    format_error,
    format_success,
    format_warning,
    get_spinner,
    format_table,
    format_usd,
    format_wallet,
)

# Create main app and subcommand groups
app = typer.Typer(
    name="polybot",
    help="Polymarket Copy-Trading Bot - High-performance position mirroring",
    add_completion=False,
    no_args_is_help=True,
)

accounts_app = typer.Typer(help="Manage tracked accounts")
app.add_typer(accounts_app, name="accounts")


# =============================================================================
# Helper Functions
# =============================================================================

def get_config_dir() -> Path:
    """Get the config directory path."""
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    script_config = Path(__file__).parent.parent.parent / "config"
    if script_config.exists():
        return script_config

    return cwd_config


def load_app_config(config_dir: Optional[Path] = None):
    """Load application configuration."""
    from ..config import load_config

    config_path = config_dir or get_config_dir()
    return load_config(config_dir=config_path)


def get_database():
    """Get database connection."""
    from ..database import configure_database, DatabaseConfig

    # Use SQLite for now
    db_path = Path.cwd() / "polymarketbot.db"
    config = DatabaseConfig(url=f"sqlite:///{db_path}")
    return configure_database(config)


def setup_logging_for_cli(verbose: bool = False):
    """Setup logging for CLI commands."""
    from ..utils.logging import setup_logging

    level = "DEBUG" if verbose else "WARNING"
    setup_logging(level=level, json_format=False)


# =============================================================================
# Lifecycle Commands
# =============================================================================

@app.command()
def start(
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't execute trades"),
):
    """Start the copy-trading bot.

    This will begin monitoring target accounts and executing copy trades.
    """
    setup_logging_for_cli(verbose)

    console.print("\n[bold]Starting PolymarketBot...[/bold]\n")

    # Load config
    try:
        config = load_app_config(config_dir)
    except Exception as e:
        format_error("Failed to load configuration", str(e))
        raise typer.Exit(1)

    if not config.enabled_targets:
        format_error("No enabled target accounts configured")
        raise typer.Exit(1)

    console.print(f"Loaded {len(config.enabled_targets)} target account(s)")

    if dry_run:
        format_warning("Dry-run mode: trades will NOT be executed")

    # Initialize database
    try:
        db = get_database()
        console.print("[green]Database initialized[/green]")
    except Exception as e:
        format_error("Failed to initialize database", str(e))
        raise typer.Exit(1)

    # TODO: Initialize and start bot components
    # For now, just show that we would start
    console.print("\n[yellow]Bot start functionality not yet implemented.[/yellow]")
    console.print("Components to initialize:")
    console.print("  - API clients")
    console.print("  - WebSocket connections")
    console.print("  - Activity monitors")
    console.print("  - Execution engine")
    console.print("  - Circuit breakers")
    console.print("  - Health checks")

    # In full implementation, this would:
    # 1. Initialize all components
    # 2. Start monitoring loop
    # 3. Run until SIGINT/SIGTERM


@app.command()
def stop():
    """Stop the running bot gracefully.

    Sends shutdown signal to running bot process.
    """
    console.print("\n[bold]Stopping PolymarketBot...[/bold]\n")

    # Check if bot is running (via PID file or similar)
    pid_file = Path.cwd() / "polybot.pid"

    if not pid_file.exists():
        format_warning("Bot does not appear to be running (no PID file)")
        raise typer.Exit(0)

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent shutdown signal to process {pid}[/green]")

    except ProcessLookupError:
        format_warning(f"Process not found. Removing stale PID file.")
        pid_file.unlink()
    except Exception as e:
        format_error("Failed to stop bot", str(e))
        raise typer.Exit(1)


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
):
    """Show current bot status.

    Displays running state, positions, and health information.
    """
    setup_logging_for_cli(False)

    # Check if bot is running
    pid_file = Path.cwd() / "polybot.pid"
    is_running = pid_file.exists()

    if not is_running:
        panel = format_status(is_running=False, state={})
        console.print(panel)
        return

    # Load status from database/state
    try:
        db = get_database()

        from ..database import (
            PositionRepository,
            TradeLogRepository,
            BalanceSnapshotRepository,
            DailyStatsRepository,
        )

        pos_repo = PositionRepository(db)
        trade_repo = TradeLogRepository(db)
        balance_repo = BalanceSnapshotRepository(db)
        stats_repo = DailyStatsRepository(db)

        # Get current state
        positions = pos_repo.get_open_positions()
        latest_balance = balance_repo.get_latest()
        today_stats = stats_repo.get_or_create_today()

        # Calculate position value
        total_value = sum(
            float(p.our_size) * float(p.current_price or p.average_price)
            for p in positions
        )

        state = {
            "uptime": "unknown",  # Would come from running bot
            "balance": {
                "total": latest_balance.total_balance if latest_balance else None,
                "available": latest_balance.available_balance if latest_balance else None,
                "reserved": latest_balance.reserved_balance if latest_balance else None,
            },
            "positions": {
                "open": len(positions),
                "value": Decimal(str(total_value)),
                "unrealized_pnl": latest_balance.unrealized_pnl if latest_balance else None,
            },
            "today": {
                "trades": today_stats.trades_executed,
                "pnl": today_stats.realized_pnl,
            },
            "health": {
                "status": "unknown",
            },
            "circuit_breaker": {
                "is_open": False,
            },
        }

        panel = format_status(is_running=True, state=state)
        console.print(panel)

        if verbose:
            # Show position details
            if positions:
                pos_data = [p.to_dict() for p in positions]
                console.print(format_positions(pos_data))

    except Exception as e:
        format_error("Failed to get status", str(e))
        raise typer.Exit(1)


# =============================================================================
# Account Commands
# =============================================================================

@accounts_app.command("list")
def accounts_list(
    show_disabled: bool = typer.Option(False, "--all", "-a", help="Show disabled accounts"),
):
    """List all tracked accounts."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import AccountRepository

        repo = AccountRepository(db)

        if show_disabled:
            accounts = repo.get_all()
        else:
            accounts = repo.get_enabled()

        if not accounts:
            console.print("\n[yellow]No accounts configured.[/yellow]")
            console.print("Use [bold]polybot accounts add[/bold] to add a target account.\n")
            return

        account_data = [a.to_dict() for a in accounts]
        table = format_accounts(account_data)
        console.print(table)

    except Exception as e:
        format_error("Failed to list accounts", str(e))
        raise typer.Exit(1)


@accounts_app.command("add")
def accounts_add(
    name: str = typer.Option(..., "--name", "-n", prompt=True, help="Account name"),
    wallet: str = typer.Option(..., "--wallet", "-w", prompt=True, help="Target wallet address"),
    ratio: float = typer.Option(0.01, "--ratio", "-r", help="Position ratio (0.01 = 1%)"),
    max_usd: float = typer.Option(500, "--max-usd", "-m", help="Max position in USD"),
    slippage: float = typer.Option(0.05, "--slippage", "-s", help="Max slippage (0.05 = 5%)"),
):
    """Add a new target account to track."""
    setup_logging_for_cli(False)

    # Validate wallet address
    if not wallet.startswith("0x") or len(wallet) != 42:
        format_error("Invalid wallet address format")
        raise typer.Exit(1)

    try:
        db = get_database()
        from ..database import AccountRepository

        repo = AccountRepository(db)

        # Check if name already exists
        existing = repo.get_by_name(name)
        if existing:
            format_error(f"Account '{name}' already exists")
            raise typer.Exit(1)

        # Create account
        account = repo.create(
            name=name,
            target_wallet=wallet.lower(),
            position_ratio=Decimal(str(ratio)),
            max_position_usd=Decimal(str(max_usd)),
            slippage_tolerance=Decimal(str(slippage)),
            enabled=True,
        )

        format_success(f"Added account '{name}'")
        console.print(f"  Wallet: {format_wallet(wallet)}")
        console.print(f"  Ratio: {ratio*100:.1f}%")
        console.print(f"  Max USD: ${max_usd:.2f}")
        console.print(f"  Slippage: {slippage*100:.1f}%")

    except Exception as e:
        format_error("Failed to add account", str(e))
        raise typer.Exit(1)


@accounts_app.command("remove")
def accounts_remove(
    name: str = typer.Argument(..., help="Account name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a tracked account."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import AccountRepository, PositionRepository

        repo = AccountRepository(db)

        account = repo.get_by_name(name)
        if not account:
            format_error(f"Account '{name}' not found")
            raise typer.Exit(1)

        # Check for open positions
        pos_repo = PositionRepository(db)
        positions = pos_repo.get_open_positions(account.id)

        if positions and not force:
            format_warning(f"Account has {len(positions)} open position(s)")
            if not Confirm.ask("Are you sure you want to remove this account?"):
                raise typer.Exit(0)

        repo.delete(account.id)
        format_success(f"Removed account '{name}'")

    except typer.Exit:
        raise
    except Exception as e:
        format_error("Failed to remove account", str(e))
        raise typer.Exit(1)


@accounts_app.command("pause")
def accounts_pause(
    name: str = typer.Argument(..., help="Account name to pause"),
):
    """Pause tracking for an account."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import AccountRepository

        repo = AccountRepository(db)

        if not repo.set_enabled(name, False):
            format_error(f"Account '{name}' not found")
            raise typer.Exit(1)

        format_success(f"Paused tracking for '{name}'")

    except typer.Exit:
        raise
    except Exception as e:
        format_error("Failed to pause account", str(e))
        raise typer.Exit(1)


@accounts_app.command("resume")
def accounts_resume(
    name: str = typer.Argument(..., help="Account name to resume"),
):
    """Resume tracking for an account."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import AccountRepository

        repo = AccountRepository(db)

        if not repo.set_enabled(name, True):
            format_error(f"Account '{name}' not found")
            raise typer.Exit(1)

        format_success(f"Resumed tracking for '{name}'")

    except typer.Exit:
        raise
    except Exception as e:
        format_error("Failed to resume account", str(e))
        raise typer.Exit(1)


# =============================================================================
# Position and Trade Commands
# =============================================================================

@app.command()
def positions(
    account: Optional[str] = typer.Option(None, "--account", "-a", help="Filter by account"),
    show_closed: bool = typer.Option(False, "--closed", "-c", help="Show closed positions"),
):
    """Show current positions."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import PositionRepository, AccountRepository

        pos_repo = PositionRepository(db)
        acc_repo = AccountRepository(db)

        # Get account filter
        account_id = None
        if account:
            acc = acc_repo.get_by_name(account)
            if not acc:
                format_error(f"Account '{account}' not found")
                raise typer.Exit(1)
            account_id = acc.id

        positions_list = pos_repo.get_open_positions(account_id)

        if not positions_list:
            console.print("\n[yellow]No open positions.[/yellow]\n")
            return

        pos_data = [p.to_dict() for p in positions_list]
        table = format_positions(pos_data, show_closed=show_closed)
        console.print(table)

        # Summary
        total_value = sum(
            float(p.our_size) * float(p.current_price or p.average_price)
            for p in positions_list
        )
        console.print(f"\n  Total Value: {format_usd(Decimal(str(total_value)))}\n")

    except typer.Exit:
        raise
    except Exception as e:
        format_error("Failed to get positions", str(e))
        raise typer.Exit(1)


@app.command()
def trades(
    account: Optional[str] = typer.Option(None, "--account", "-a", help="Filter by account"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of trades to show"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """Show recent trade history."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import TradeLogRepository, AccountRepository, TradeStatus

        trade_repo = TradeLogRepository(db)
        acc_repo = AccountRepository(db)

        # Get account filter
        account_id = None
        if account:
            acc = acc_repo.get_by_name(account)
            if not acc:
                format_error(f"Account '{account}' not found")
                raise typer.Exit(1)
            account_id = acc.id

        # Get status filter
        status_filter = None
        if status:
            try:
                status_filter = TradeStatus(status)
            except ValueError:
                format_error(f"Invalid status: {status}")
                console.print(f"Valid statuses: {', '.join(s.value for s in TradeStatus)}")
                raise typer.Exit(1)

        trades_list = trade_repo.get_recent_trades(
            account_id=account_id,
            limit=limit,
            status=status_filter,
        )

        if not trades_list:
            console.print("\n[yellow]No trades found.[/yellow]\n")
            return

        trade_data = [t.to_dict() for t in trades_list]
        table = format_trades(trade_data, limit=limit)
        console.print(table)

    except typer.Exit:
        raise
    except Exception as e:
        format_error("Failed to get trades", str(e))
        raise typer.Exit(1)


@app.command()
def pnl(
    days: int = typer.Option(30, "--days", "-d", help="Number of days for P&L calculation"),
):
    """Show P&L summary."""
    setup_logging_for_cli(False)

    try:
        db = get_database()
        from ..database import AuditService

        audit = AuditService(db)
        summary = audit.get_performance_summary(days=days)

        # Also get current unrealized P&L from positions
        from ..database import PositionRepository, BalanceSnapshotRepository

        pos_repo = PositionRepository(db)
        balance_repo = BalanceSnapshotRepository(db)

        positions = pos_repo.get_open_positions()
        latest_balance = balance_repo.get_latest()

        unrealized = Decimal("0")
        for p in positions:
            if p.unrealized_pnl:
                unrealized += p.unrealized_pnl

        pnl_data = {
            "realized_pnl": summary.get("total_realized_pnl", "0"),
            "unrealized_pnl": str(unrealized),
            "total_invested": summary.get("total_volume_usd", "0"),
            "roi_percent": 0,  # Would calculate from invested vs current value
            "total_trades": summary.get("total_trades_executed", 0),
            "winning_trades": 0,  # Would need to track
            "losing_trades": 0,
            "win_rate_percent": (
                summary["total_trades_executed"] / summary["total_trades_detected"] * 100
                if summary.get("total_trades_detected", 0) > 0 else 0
            ),
        }

        panel = format_pnl(pnl_data)
        console.print(panel)

    except Exception as e:
        format_error("Failed to calculate P&L", str(e))
        raise typer.Exit(1)


# =============================================================================
# Utility Commands
# =============================================================================

@app.command("test-connection")
def test_connection(
    timeout: int = typer.Option(10, help="Request timeout in seconds"),
):
    """Test connectivity to Polymarket APIs.

    Verifies that your network/VPN allows access to Polymarket servers.
    """
    setup_logging_for_cli(False)

    console.print("\n[bold]Testing Polymarket API Connectivity...[/bold]\n")

    async def run_tests():
        from ..api.connectivity import test_all_endpoints
        return await test_all_endpoints(timeout_s=timeout)

    with get_spinner() as progress:
        task = progress.add_task("Testing endpoints...", total=None)
        results = asyncio.run(run_tests())

    # Display results
    from rich.table import Table

    table = Table(title="Connectivity Results")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Latency", style="yellow")
    table.add_column("Details", style="dim")

    for result in results:
        if result.is_ok:
            status = "[green]OK[/green]"
            latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "-"
            details = f"HTTP {result.status_code}" if result.status_code else ""
        else:
            status = f"[red]{result.status.value.upper()}[/red]"
            latency = "-"
            details = result.error_message or ""

        table.add_row(result.endpoint_name, status, latency, details)

    console.print(table)

    ok_count = sum(1 for r in results if r.is_ok)
    total = len(results)

    if ok_count == total:
        console.print(f"\n[green]All {total} endpoints reachable.[/green]\n")
    else:
        console.print(f"\n[yellow]{ok_count}/{total} endpoints reachable.[/yellow]")
        if any(r.status.value == "blocked" for r in results):
            console.print(
                "\n[red]Geo-blocking detected![/red] Ensure your VPN is connected.\n"
            )
        raise typer.Exit(1)


@app.command("validate")
def validate_config(
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c", help="Config directory"),
):
    """Validate configuration files.

    Checks that configuration is valid and required secrets are set.
    """
    setup_logging_for_cli(False)

    config_path = config_dir or get_config_dir()
    console.print(f"\n[bold]Validating configuration in {config_path}...[/bold]\n")

    try:
        from ..config import load_config
        from ..config.loader import validate_required_secrets
        from ..config.validation import validate_config as do_validate, format_validation_issues

        config = load_config(config_dir=config_path)
        console.print("[green]Configuration files loaded successfully.[/green]\n")

        # Check secrets
        missing = validate_required_secrets(config)
        if missing:
            console.print("[yellow]Missing environment variables:[/yellow]")
            for var in missing:
                console.print(f"  - {var}")
            console.print("")

        # Validate
        is_valid, issues = do_validate(config)

        if issues:
            console.print(format_validation_issues(issues))
        else:
            console.print("[green]All validation checks passed.[/green]")

        # Summary table
        from rich.table import Table

        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        table.add_row("Your Proxy Wallet", format_wallet(config.your_account.proxy_wallet))
        table.add_row("Target Accounts", str(len(config.targets)))
        table.add_row("Enabled Targets", str(len(config.enabled_targets)))
        table.add_row("Polling Interval", f"{config.polling.activity_interval_ms}ms")
        table.add_row("Max Daily Loss", f"${config.safety.max_daily_loss_usd}")

        console.print(table)

        if not is_valid:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        format_error("Configuration error", str(e))
        raise typer.Exit(1)


@app.command()
def reconcile(
    account: Optional[str] = typer.Option(None, "--account", "-a", help="Specific account"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't make changes"),
):
    """Force position reconciliation.

    Syncs local position state with API data.
    """
    setup_logging_for_cli(False)

    console.print("\n[bold]Running position reconciliation...[/bold]\n")

    if dry_run:
        format_warning("Dry-run mode: no changes will be made")

    # TODO: Implement actual reconciliation
    console.print("[yellow]Reconciliation not yet implemented.[/yellow]")
    console.print("This would:")
    console.print("  1. Fetch positions from Data API")
    console.print("  2. Compare with local database")
    console.print("  3. Update any discrepancies")
    console.print("  4. Report drift status")


# =============================================================================
# Simulation Commands
# =============================================================================

simulate_app = typer.Typer(help="Simulation and testing tools")
app.add_typer(simulate_app, name="simulate")


@simulate_app.command("run")
def simulate_run(
    wallet: Optional[str] = typer.Option(None, "--wallet", "-w", help="Target wallet to simulate"),
    mode: str = typer.Option("synthetic", "--mode", "-m", help="Mode: synthetic, historical, dry-run"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Speed multiplier for historical replay"),
    trades_per_hour: int = typer.Option(10, "--rate", "-r", help="Trades per hour (synthetic mode)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run copy trading simulation.

    Modes:
    - synthetic: Generate fake trades for testing
    - historical: Replay historical trades from wallet
    - dry-run: Connect to real WebSocket but don't execute
    """
    setup_logging_for_cli(verbose)

    # Default wallet if not provided
    if not wallet:
        wallet = "0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11"  # automatedaitradingbot
        console.print(f"[dim]Using default wallet: {format_wallet(wallet, short=True)}[/dim]")

    console.print(f"\n[bold]Starting {mode} simulation...[/bold]\n")
    console.print(f"  Target Wallet: {format_wallet(wallet, short=True)}")
    console.print(f"  Duration: {duration}s")
    if mode == "historical":
        console.print(f"  Speed: {speed}x")
    elif mode == "synthetic":
        console.print(f"  Rate: {trades_per_hour} trades/hour")
    console.print("")

    async def run_simulation():
        from ..simulation import SimulationHarness, SimulationConfig
        from ..simulation.harness import SimulationMode

        mode_map = {
            "synthetic": SimulationMode.SYNTHETIC,
            "historical": SimulationMode.HISTORICAL,
            "dry-run": SimulationMode.DRY_RUN,
            "dry_run": SimulationMode.DRY_RUN,
        }

        config = SimulationConfig(
            mode=mode_map.get(mode, SimulationMode.SYNTHETIC),
            target_wallets=[wallet],
            speed_multiplier=speed,
            trades_per_hour=trades_per_hour,
            save_results_path=str(output) if output else None,
            log_all_trades=verbose,
        )

        harness = SimulationHarness(config)

        # Track metrics for display
        trade_count = 0

        async def on_trade(trade):
            nonlocal trade_count
            trade_count += 1
            if not verbose:
                console.print(
                    f"  [cyan]#{trade_count}[/cyan] "
                    f"{trade.side} {float(trade.size):.2f} @ {float(trade.price):.4f} "
                    f"[dim]{trade.market_name[:40]}[/dim]"
                )

        harness.on_trade_detected(on_trade)

        await harness.start()
        await asyncio.sleep(duration)
        return await harness.stop()

    try:
        with get_spinner() as progress:
            task = progress.add_task("Running simulation...", total=duration)
            metrics = asyncio.run(run_simulation())

        # Display results
        console.print("\n[bold green]Simulation Complete[/bold green]\n")

        from rich.table import Table
        table = Table(title="Simulation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        results = metrics.to_dict()
        table.add_row("Trades Detected", str(results.get("trades_detected", 0)))
        table.add_row("Trades Executed", str(results.get("trades_executed", 0)))
        table.add_row("Trades Failed", str(results.get("trades_failed", 0)))
        table.add_row("Success Rate", f"{results.get('success_rate_pct', 0):.1f}%")
        table.add_row("Avg Detection Latency", f"{results.get('avg_detection_latency_ms', 0):.1f} ms")
        table.add_row("Avg Execution Latency", f"{results.get('avg_execution_latency_ms', 0):.1f} ms")
        table.add_row("Avg Slippage", f"{results.get('avg_slippage_pct', 0):.2f}%")

        console.print(table)

        if output:
            console.print(f"\nResults saved to: {output}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted.[/yellow]")
    except Exception as e:
        format_error("Simulation failed", str(e))
        raise typer.Exit(1)


@simulate_app.command("benchmark")
def simulate_benchmark(
    wallet: str = typer.Argument(..., help="Wallet address to benchmark"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
):
    """Benchmark trade detection latency.

    Connects to live Polymarket WebSocket and measures
    how quickly we can detect trades from the target wallet.
    """
    setup_logging_for_cli(False)

    console.print(f"\n[bold]Benchmarking detection latency...[/bold]")
    console.print(f"  Wallet: {format_wallet(wallet, short=True)}")
    console.print(f"  Duration: {duration}s")
    console.print("\n[dim]Waiting for trades from target wallet...[/dim]\n")

    async def run_benchmark():
        from ..simulation.mock_websocket import TradeDetectionBenchmark

        benchmark = TradeDetectionBenchmark(wallet)
        return await benchmark.run_benchmark(duration_s=duration)

    try:
        results = asyncio.run(run_benchmark())

        if results.get("trades_detected", 0) == 0:
            console.print("[yellow]No trades detected during benchmark period.[/yellow]")
            console.print("Try again when the target wallet is actively trading.")
            return

        from rich.table import Table
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Trades Detected", str(results.get("trades_detected", 0)))
        table.add_row("Avg Latency", f"{results.get('avg_latency_ms', 0):.1f} ms")
        table.add_row("Min Latency", f"{results.get('min_latency_ms', 0):.1f} ms")
        table.add_row("Max Latency", f"{results.get('max_latency_ms', 0):.1f} ms")
        table.add_row("P50 Latency", f"{results.get('p50_latency_ms', 0):.1f} ms")
        table.add_row("P95 Latency", f"{results.get('p95_latency_ms', 0):.1f} ms")

        console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted.[/yellow]")
    except Exception as e:
        format_error("Benchmark failed", str(e))
        raise typer.Exit(1)


@simulate_app.command("profile")
def simulate_profile(
    wallet: str = typer.Argument(..., help="Wallet address to analyze"),
    days: int = typer.Option(7, "--days", "-d", help="Days of history to analyze"),
):
    """Analyze trading profile of a wallet.

    Fetches historical trades and provides statistics
    about the trader's patterns and performance.
    """
    setup_logging_for_cli(False)

    console.print(f"\n[bold]Analyzing trader profile...[/bold]")
    console.print(f"  Wallet: {format_wallet(wallet, short=True)}")
    console.print(f"  Period: Last {days} days\n")

    async def analyze():
        from ..simulation.historical_replay import HistoricalReplay
        from datetime import datetime, timedelta

        replay = HistoricalReplay(
            wallets=[wallet],
            start_time=datetime.utcnow() - timedelta(days=days),
            end_time=datetime.utcnow(),
        )

        await replay.load_trades()
        return replay.get_trade_summary()

    try:
        with get_spinner() as progress:
            task = progress.add_task("Fetching trades...", total=None)
            summary = asyncio.run(analyze())

        if summary.get("trades", 0) == 0:
            console.print("[yellow]No trades found for this wallet.[/yellow]")
            return

        from rich.table import Table
        table = Table(title=f"Trader Profile: {format_wallet(wallet, short=True)}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Trades", str(summary.get("trades", 0)))
        table.add_row("Total Volume", format_usd(Decimal(str(summary.get("total_volume", 0)))))
        table.add_row("Buy Trades", str(summary.get("buy_trades", 0)))
        table.add_row("Sell Trades", str(summary.get("sell_trades", 0)))
        table.add_row("Unique Markets", str(summary.get("unique_markets", 0)))
        table.add_row("Time Span", f"{summary.get('time_span_hours', 0):.1f} hours")
        table.add_row("Trades/Hour", f"{summary.get('trades_per_hour', 0):.2f}")

        console.print(table)

    except Exception as e:
        format_error("Profile analysis failed", str(e))
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    try:
        from .. import __version__
        ver = __version__
    except ImportError:
        ver = "0.1.0"

    console.print(f"\nPolymarketBot v{ver}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
