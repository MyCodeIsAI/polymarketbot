"""Output formatters for CLI using Rich.

Provides consistent formatting for:
- Tables (accounts, positions, trades)
- Status displays
- P&L reports
- Error messages
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Any, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def format_decimal(value: Optional[Decimal], decimals: int = 2) -> str:
    """Format a decimal value.

    Args:
        value: Decimal value
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if value is None:
        return "-"
    return f"{float(value):,.{decimals}f}"


def format_usd(value: Optional[Decimal]) -> str:
    """Format a USD value.

    Args:
        value: USD value

    Returns:
        Formatted string with $ prefix
    """
    if value is None:
        return "-"
    return f"${float(value):,.2f}"


def format_percent(value: Optional[Decimal], include_sign: bool = False) -> str:
    """Format a percentage value.

    Args:
        value: Decimal value (0.05 = 5%)
        include_sign: Include + for positive values

    Returns:
        Formatted percentage string
    """
    if value is None:
        return "-"

    pct = float(value) * 100
    if include_sign and pct > 0:
        return f"+{pct:.2f}%"
    return f"{pct:.2f}%"


def format_timestamp(dt: Optional[datetime]) -> str:
    """Format a datetime.

    Args:
        dt: Datetime object

    Returns:
        Formatted string
    """
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_relative_time(dt: Optional[datetime]) -> str:
    """Format datetime as relative time.

    Args:
        dt: Datetime object

    Returns:
        Relative time string (e.g., "5m ago")
    """
    if dt is None:
        return "-"

    now = datetime.utcnow()
    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    else:
        return f"{int(seconds / 86400)}d ago"


def format_wallet(wallet: str, short: bool = True) -> str:
    """Format a wallet address.

    Args:
        wallet: Full wallet address
        short: Truncate to short form

    Returns:
        Formatted wallet string
    """
    if not wallet:
        return "-"
    if short and len(wallet) > 16:
        return f"{wallet[:8]}...{wallet[-6:]}"
    return wallet


def format_table(
    title: str,
    columns: List[tuple],
    rows: List[List[Any]],
    show_header: bool = True,
) -> Table:
    """Create a formatted Rich table.

    Args:
        title: Table title
        columns: List of (name, style) tuples
        rows: List of row data
        show_header: Show column headers

    Returns:
        Rich Table object
    """
    table = Table(title=title, show_header=show_header)

    for col_name, col_style in columns:
        table.add_column(col_name, style=col_style)

    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    return table


def format_status(
    is_running: bool,
    state: dict,
) -> Panel:
    """Format bot status display.

    Args:
        is_running: Whether bot is running
        state: Current state dictionary

    Returns:
        Rich Panel with status
    """
    if not is_running:
        return Panel(
            "[yellow]Bot not running.[/yellow]\n\n"
            "Use [bold]polybot start[/bold] to begin copy-trading.",
            title="PolymarketBot Status",
        )

    # Build status text
    lines = []

    # Running status
    uptime = state.get("uptime", "unknown")
    lines.append(f"[green]Running[/green] - Uptime: {uptime}")
    lines.append("")

    # Balance
    balance = state.get("balance", {})
    lines.append("[bold]Balance:[/bold]")
    lines.append(f"  Total: {format_usd(balance.get('total'))}")
    lines.append(f"  Available: {format_usd(balance.get('available'))}")
    lines.append(f"  Reserved: {format_usd(balance.get('reserved'))}")
    lines.append("")

    # Positions
    positions = state.get("positions", {})
    lines.append("[bold]Positions:[/bold]")
    lines.append(f"  Open: {positions.get('open', 0)}")
    lines.append(f"  Value: {format_usd(positions.get('value'))}")
    lines.append(f"  Unrealized P&L: {format_usd(positions.get('unrealized_pnl'))}")
    lines.append("")

    # Today's stats
    today = state.get("today", {})
    lines.append("[bold]Today:[/bold]")
    lines.append(f"  Trades: {today.get('trades', 0)}")
    lines.append(f"  P&L: {format_usd(today.get('pnl'))}")
    lines.append("")

    # Health
    health = state.get("health", {})
    health_status = health.get("status", "unknown")
    if health_status == "healthy":
        health_text = "[green]Healthy[/green]"
    elif health_status == "degraded":
        health_text = "[yellow]Degraded[/yellow]"
    else:
        health_text = f"[red]{health_status}[/red]"

    lines.append(f"[bold]Health:[/bold] {health_text}")

    # Circuit breaker
    cb_state = state.get("circuit_breaker", {})
    if cb_state.get("is_open"):
        lines.append(f"[red]Circuit Breaker: OPEN[/red] - {cb_state.get('reason', 'unknown')}")

    return Panel("\n".join(lines), title="PolymarketBot Status")


def format_accounts(accounts: List[dict]) -> Table:
    """Format accounts table.

    Args:
        accounts: List of account dictionaries

    Returns:
        Rich Table
    """
    table = Table(title="Tracked Accounts")
    table.add_column("Name", style="cyan")
    table.add_column("Wallet", style="dim")
    table.add_column("Ratio", style="yellow", justify="right")
    table.add_column("Max USD", style="green", justify="right")
    table.add_column("Slippage", style="magenta", justify="right")
    table.add_column("Status", style="white")

    for acc in accounts:
        status = "[green]Enabled[/green]" if acc.get("enabled") else "[red]Disabled[/red]"

        table.add_row(
            acc.get("name", "-"),
            format_wallet(acc.get("target_wallet", "")),
            format_percent(Decimal(str(acc.get("position_ratio", 0)))),
            format_usd(Decimal(str(acc.get("max_position_usd", 0)))),
            format_percent(Decimal(str(acc.get("slippage_tolerance", 0)))),
            status,
        )

    return table


def format_positions(
    positions: List[dict],
    show_closed: bool = False,
) -> Table:
    """Format positions table.

    Args:
        positions: List of position dictionaries
        show_closed: Include closed positions

    Returns:
        Rich Table
    """
    table = Table(title="Positions")
    table.add_column("Market", style="cyan", max_width=30)
    table.add_column("Outcome", style="white")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Avg Price", style="white", justify="right")
    table.add_column("Cur Price", style="white", justify="right")
    table.add_column("P&L", style="green", justify="right")
    table.add_column("Status", style="dim")

    for pos in positions:
        if not show_closed and pos.get("status") == "closed":
            continue

        # Calculate P&L color
        pnl = pos.get("unrealized_pnl")
        if pnl:
            pnl_decimal = Decimal(str(pnl))
            if pnl_decimal > 0:
                pnl_str = f"[green]{format_usd(pnl_decimal)}[/green]"
            elif pnl_decimal < 0:
                pnl_str = f"[red]{format_usd(pnl_decimal)}[/red]"
            else:
                pnl_str = format_usd(pnl_decimal)
        else:
            pnl_str = "-"

        # Status indicator
        status = pos.get("status", "unknown")
        if status == "synced":
            status_str = "[green]●[/green]"
        elif status == "drift":
            status_str = "[yellow]●[/yellow]"
        elif status == "closed":
            status_str = "[dim]●[/dim]"
        else:
            status_str = "[red]●[/red]"

        market_name = pos.get("market_name") or pos.get("market_id", "-")[:20]

        table.add_row(
            market_name,
            pos.get("outcome", "-"),
            format_decimal(Decimal(str(pos.get("our_size", 0)))),
            format_decimal(Decimal(str(pos.get("average_price", 0))), 4),
            format_decimal(Decimal(str(pos.get("current_price", 0))), 4) if pos.get("current_price") else "-",
            pnl_str,
            status_str,
        )

    return table


def format_trades(trades: List[dict], limit: int = 20) -> Table:
    """Format trades table.

    Args:
        trades: List of trade dictionaries
        limit: Maximum rows to show

    Returns:
        Rich Table
    """
    table = Table(title=f"Recent Trades (last {limit})")
    table.add_column("Time", style="dim")
    table.add_column("Account", style="cyan")
    table.add_column("Side", style="white")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Price", style="white", justify="right")
    table.add_column("Slippage", style="magenta", justify="right")
    table.add_column("Status", style="white")
    table.add_column("Latency", style="dim", justify="right")

    for trade in trades[:limit]:
        # Side color
        side = trade.get("side", "-")
        if side == "BUY":
            side_str = "[green]BUY[/green]"
        else:
            side_str = "[red]SELL[/red]"

        # Status color
        status = trade.get("status", "unknown")
        if status == "filled":
            status_str = "[green]Filled[/green]"
        elif status == "failed":
            status_str = "[red]Failed[/red]"
        elif status.startswith("skipped"):
            status_str = f"[yellow]{status}[/yellow]"
        else:
            status_str = status

        # Slippage
        slippage = trade.get("slippage_percent")
        if slippage:
            slippage_str = format_percent(Decimal(str(slippage)))
        else:
            slippage_str = "-"

        # Latency
        latency = trade.get("total_latency_ms")
        latency_str = f"{latency}ms" if latency else "-"

        # Time
        detected_at = trade.get("detected_at")
        if detected_at:
            if isinstance(detected_at, str):
                time_str = detected_at[:19]
            else:
                time_str = format_relative_time(detected_at)
        else:
            time_str = "-"

        table.add_row(
            time_str,
            str(trade.get("account_id", "-")),
            side_str,
            format_decimal(Decimal(str(trade.get("execution_size") or trade.get("target_size", 0)))),
            format_decimal(Decimal(str(trade.get("execution_price") or trade.get("target_price", 0))), 4),
            slippage_str,
            status_str,
            latency_str,
        )

    return table


def format_pnl(pnl_data: dict) -> Panel:
    """Format P&L summary panel.

    Args:
        pnl_data: P&L data dictionary

    Returns:
        Rich Panel
    """
    lines = []

    # Overall P&L
    lines.append("[bold]Overall P&L[/bold]")
    lines.append("")

    realized = pnl_data.get("realized_pnl", 0)
    unrealized = pnl_data.get("unrealized_pnl", 0)
    total = Decimal(str(realized)) + Decimal(str(unrealized))

    realized_decimal = Decimal(str(realized))
    if realized_decimal >= 0:
        lines.append(f"  Realized:   [green]{format_usd(realized_decimal)}[/green]")
    else:
        lines.append(f"  Realized:   [red]{format_usd(realized_decimal)}[/red]")

    unrealized_decimal = Decimal(str(unrealized))
    if unrealized_decimal >= 0:
        lines.append(f"  Unrealized: [green]{format_usd(unrealized_decimal)}[/green]")
    else:
        lines.append(f"  Unrealized: [red]{format_usd(unrealized_decimal)}[/red]")

    if total >= 0:
        lines.append(f"  [bold]Total:      [green]{format_usd(total)}[/green][/bold]")
    else:
        lines.append(f"  [bold]Total:      [red]{format_usd(total)}[/red][/bold]")

    lines.append("")

    # ROI
    roi = pnl_data.get("roi_percent", 0)
    roi_decimal = Decimal(str(roi))
    if roi_decimal >= 0:
        lines.append(f"  ROI: [green]+{float(roi_decimal):.2f}%[/green]")
    else:
        lines.append(f"  ROI: [red]{float(roi_decimal):.2f}%[/red]")

    lines.append("")

    # Trade statistics
    lines.append("[bold]Trade Statistics[/bold]")
    lines.append("")
    lines.append(f"  Total Trades:  {pnl_data.get('total_trades', 0)}")
    lines.append(f"  Winning:       {pnl_data.get('winning_trades', 0)}")
    lines.append(f"  Losing:        {pnl_data.get('losing_trades', 0)}")

    win_rate = pnl_data.get("win_rate_percent", 0)
    lines.append(f"  Win Rate:      {float(win_rate):.1f}%")

    lines.append("")
    lines.append(f"  Total Volume:  {format_usd(Decimal(str(pnl_data.get('total_invested', 0))))}")

    return Panel("\n".join(lines), title="P&L Summary")


def format_error(message: str, details: Optional[str] = None) -> None:
    """Print formatted error message.

    Args:
        message: Error message
        details: Optional details
    """
    console.print(f"\n[red]Error:[/red] {message}")
    if details:
        console.print(f"[dim]{details}[/dim]")
    console.print("")


def format_success(message: str) -> None:
    """Print formatted success message.

    Args:
        message: Success message
    """
    console.print(f"\n[green]Success:[/green] {message}\n")


def format_warning(message: str) -> None:
    """Print formatted warning message.

    Args:
        message: Warning message
    """
    console.print(f"\n[yellow]Warning:[/yellow] {message}\n")


def get_spinner() -> Progress:
    """Get a spinner progress context.

    Returns:
        Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )
