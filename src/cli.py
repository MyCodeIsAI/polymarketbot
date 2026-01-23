"""Command-line interface for PolymarketBot.

This module provides the CLI entry point for all bot operations.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import load_config, AppConfig
from .config.loader import ConfigError, validate_required_secrets
from .config.validation import validate_config, format_validation_issues
from .api.connectivity import (
    test_all_endpoints,
    test_polygon_rpc,
    format_connectivity_results,
)
from .utils.logging import setup_logging

app = typer.Typer(
    name="polybot",
    help="Polymarket Copy-Trading Bot - High-performance position mirroring",
    add_completion=False,
)
console = Console()


def get_config_dir() -> Path:
    """Get the config directory path."""
    # Try current directory first
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    # Try relative to script
    script_config = Path(__file__).parent.parent / "config"
    if script_config.exists():
        return script_config

    return cwd_config  # Return default even if doesn't exist


@app.command()
def test_connection(
    timeout: int = typer.Option(10, help="Request timeout in seconds"),
    polygon_rpc: Optional[str] = typer.Option(None, help="Polygon RPC URL to test"),
):
    """Test connectivity to all Polymarket API endpoints.

    This command verifies that your network/VPN configuration allows
    access to Polymarket's servers.
    """
    setup_logging(level="WARNING", json_format=False)

    console.print("\n[bold]Testing Polymarket API Connectivity...[/bold]\n")

    async def run_tests():
        results = await test_all_endpoints(timeout_s=timeout)

        # Also test Polygon RPC if provided
        if polygon_rpc:
            rpc_result = await test_polygon_rpc(polygon_rpc, timeout_s=timeout)
            results.append(rpc_result)

        return results

    results = asyncio.run(run_tests())

    # Display results
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

    # Summary
    ok_count = sum(1 for r in results if r.is_ok)
    total = len(results)

    if ok_count == total:
        console.print(f"\n[green]All {total} endpoints reachable.[/green]")
    else:
        console.print(f"\n[yellow]{ok_count}/{total} endpoints reachable.[/yellow]")
        if any(r.status.value == "blocked" for r in results):
            console.print(
                "\n[red]Geo-blocking detected![/red] Ensure your VPN is connected "
                "with exit node in Amsterdam, Frankfurt, or other allowed region."
            )


@app.command()
def validate(
    config_dir: Optional[Path] = typer.Option(None, help="Config directory path"),
):
    """Validate configuration files.

    Checks that all configuration is valid and all required secrets
    are set in environment variables.
    """
    setup_logging(level="WARNING", json_format=False)

    config_path = config_dir or get_config_dir()

    console.print(f"\n[bold]Validating configuration in {config_path}...[/bold]\n")

    # Load configuration
    try:
        config = load_config(config_dir=config_path)
        console.print("[green]Configuration files loaded successfully.[/green]\n")
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    # Check required secrets
    missing_secrets = validate_required_secrets(config)
    if missing_secrets:
        console.print("[yellow]Missing environment variables:[/yellow]")
        for var in missing_secrets:
            console.print(f"  - {var}")
        console.print("\nSet these in your .env file or environment.")
        console.print("")

    # Run validation checks
    is_valid, issues = validate_config(config)

    if issues:
        console.print(format_validation_issues(issues))
    else:
        console.print("[green]All validation checks passed.[/green]")

    # Show summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Your Proxy Wallet", config.your_account.proxy_wallet)
    table.add_row("Signature Type", config.your_account.signature_type.name)
    table.add_row("Target Accounts", str(len(config.targets)))
    table.add_row("Enabled Targets", str(len(config.enabled_targets)))
    table.add_row("Polling Interval", f"{config.polling.activity_interval_ms}ms")
    table.add_row("Max Daily Loss", f"${config.safety.max_daily_loss_usd}")

    console.print(table)

    if config.enabled_targets:
        console.print("\n[bold]Enabled Targets:[/bold]")
        target_table = Table()
        target_table.add_column("Name", style="cyan")
        target_table.add_column("Wallet", style="dim")
        target_table.add_column("Ratio", style="yellow")
        target_table.add_column("Max USD", style="green")
        target_table.add_column("Slippage", style="magenta")

        for target in config.enabled_targets:
            target_table.add_row(
                target.name,
                f"{target.wallet[:10]}...{target.wallet[-6:]}",
                f"{float(target.position_ratio):.2%}",
                f"${target.max_position_usd}",
                f"{float(target.slippage_tolerance):.1%}",
            )

        console.print(target_table)

    if not is_valid:
        raise typer.Exit(1)


@app.command()
def status():
    """Show current bot status (placeholder for future implementation)."""
    console.print(
        Panel(
            "[yellow]Bot not running.[/yellow]\n\n"
            "Use [bold]polybot start[/bold] to begin copy-trading.",
            title="PolymarketBot Status",
        )
    )


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"PolymarketBot v{__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
