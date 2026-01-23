#!/usr/bin/env python3
"""
Ghost Mode Dashboard - Real-time copy trade simulation.

This monitors REAL trades from tracked Polymarket accounts and simulates
what the copy trading bot would do - including timing of API calls.

Features:
- Real-time monitoring of tracked accounts via Polymarket API
- Keyword filtering for specific markets (e.g., weather, temperature)
- Drawdown-based stoploss protection
- Full API request simulation (ghost execution)
- NO retroactive position mirroring - only new trades after start

Usage:
    python run_ghost_mode.py                    # Default port 8765
    python run_ghost_mode.py --port 9000        # Custom port
    PORT=9000 python run_ghost_mode.py          # Via environment variable
"""

import argparse
import asyncio
import os
import sys
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Default port (changed from 8000 to avoid conflicts)
DEFAULT_PORT = int(os.environ.get("PORT", "8765"))

from ghost_mode import ghost_state, ghost_mode_monitor
from src.copytrade import CopyTradeAccount, DEFAULT_SLIPPAGE_TIERS
from src.copytrade.proxy import (
    ProxyConfig,
    ProxyType as ProxyTypeEnum,
    benchmark_all_endpoints,
    check_geo_restriction,
    get_infrastructure_info,
    load_proxy_config,
    save_proxy_config,
    get_proxy_config,
)
from src.utils.polymarket_api import (
    lookup_wallet_from_username,
    get_live_positions,
    get_live_portfolio_value,
)

# Import discovery router for API endpoints
try:
    from src.discovery.routes import router as discovery_router
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False


class ConnectionManager:
    """WebSocket connection manager for real-time updates."""

    def __init__(self):
        self.active_connections: set = set()

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.add(websocket)

        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to PolymarketBot Ghost Mode",
        })
        await websocket.send_json({
            "type": "ghost_mode_status",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"enabled": ghost_state.enabled, "ghost_mode": ghost_state.enabled},
        })
        await websocket.send_json({
            "type": "initial_status",
            "timestamp": datetime.utcnow().isoformat(),
            "data": ghost_state.get_status(),
        })
        await websocket.send_json({
            "type": "accounts_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": [acc.to_dict() for acc in ghost_state.accounts.values()],
        })

    def disconnect(self, websocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        for conn in disconnected:
            self.active_connections.discard(conn)


# Global connection manager
ws_manager = ConnectionManager()

# Ghost mode monitoring task
ghost_mode_task: Optional[asyncio.Task] = None


async def start_ghost_monitor():
    """Start the monitoring loop (ghost or live mode)."""
    global ghost_mode_task

    if ghost_mode_task and not ghost_mode_task.done():
        return  # Already running

    ghost_state.start()

    # Check for missed trades while system was offline
    await ghost_state.check_missed_trades(ws_manager.broadcast)

    ghost_mode_task = asyncio.create_task(
        ghost_mode_monitor(ws_manager.broadcast)
    )

    mode_name = "Live" if ghost_state.is_live_mode else "Ghost"
    await ws_manager.broadcast({
        "type": "mode_status",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "enabled": True,
            "ghost_mode": not ghost_state.is_live_mode,
            "live_mode": ghost_state.is_live_mode,
        },
    })


async def stop_ghost_monitor():
    """Stop the ghost mode monitoring loop."""
    global ghost_mode_task

    ghost_state.stop()

    if ghost_mode_task and not ghost_mode_task.done():
        ghost_mode_task.cancel()
        try:
            await ghost_mode_task
        except asyncio.CancelledError:
            pass

    ghost_mode_task = None

    await ws_manager.broadcast({
        "type": "ghost_mode_status",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {"enabled": False, "ghost_mode": False},
    })


async def run_dashboard(port: int = DEFAULT_PORT):
    """Run the web dashboard with ghost mode monitoring."""
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    app = FastAPI(title="PolymarketBot Ghost Mode Dashboard")

    # Store port for display
    dashboard_port = port

    # Pydantic models
    class AccountCreate(BaseModel):
        name: str
        wallet: str
        position_ratio: float = 0.01
        max_position_usd: float = 500
        use_tiered_slippage: bool = True
        slippage_tiers: Optional[Dict[str, float]] = None  # e.g., {"5": 300, "10": 200, ...}
        flat_slippage_tolerance: float = 0.05
        keywords: List[str] = []
        max_drawdown_percent: float = 15
        # Advanced risk settings
        take_profit_pct: float = 0  # 0 = disabled
        stop_loss_pct: float = 0  # 0 = disabled
        max_concurrent: int = 0  # 0 = unlimited
        max_holding_hours: int = 0  # 0 = disabled
        min_liquidity: float = 0  # 0 = no minimum
        cooldown_seconds: int = 10
        order_type: str = "market"  # "market" or "limit"

    class AccountUpdate(BaseModel):
        name: Optional[str] = None
        wallet: Optional[str] = None
        position_ratio: Optional[float] = None
        max_position_usd: Optional[float] = None
        use_tiered_slippage: Optional[bool] = None
        slippage_tiers: Optional[Dict[str, float]] = None
        flat_slippage_tolerance: Optional[float] = None
        keywords: Optional[List[str]] = None
        max_drawdown_percent: Optional[float] = None
        # Advanced risk settings
        take_profit_pct: Optional[float] = None
        stop_loss_pct: Optional[float] = None
        max_concurrent: Optional[int] = None
        max_holding_hours: Optional[int] = None
        min_liquidity: Optional[float] = None
        cooldown_seconds: Optional[int] = None
        enabled: Optional[bool] = None
        order_type: Optional[str] = None  # "market" or "limit"

    # Mount static files (cross-platform)
    static_dir = PROJECT_ROOT / "src" / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Dashboard route
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        template_path = PROJECT_ROOT / "src" / "web" / "templates" / "dashboard.html"
        if template_path.exists():
            return template_path.read_text()
        return "<h1>Dashboard template not found</h1>"

    # ==========================================================================
    # Status & Health APIs
    # ==========================================================================

    @app.get("/api/status")
    async def get_status():
        return ghost_state.get_status()

    @app.get("/api/health")
    async def get_health():
        return {
            "api_healthy": True,
            "websocket_healthy": len(ws_manager.active_connections) > 0 or True,
            "database_healthy": True,
            "circuit_breaker_open": False,
            "memory_percent": 35.0,
            "cpu_percent": 15.0,
            "ghost_mode": ghost_state.enabled,
        }

    # ==========================================================================
    # Wallet Lookup API
    # ==========================================================================

    @app.get("/api/lookup-wallet/{username}")
    async def api_lookup_wallet(username: str):
        """Look up a Polymarket wallet address from a username."""
        wallet = lookup_wallet_from_username(username)
        if wallet:
            return {"success": True, "username": username, "wallet": wallet}
        return {"success": False, "username": username, "wallet": None, "error": "User not found"}

    # ==========================================================================
    # Mode Control APIs (Ghost/Live)
    # ==========================================================================

    @app.post("/api/ghost-mode/start")
    async def api_start_ghost_mode():
        ghost_state.is_live_mode = False
        await start_ghost_monitor()
        return {"success": True, "ghost_mode": True, "live_mode": False, "message": "Ghost Mode started"}

    @app.post("/api/ghost-mode/stop")
    async def api_stop_ghost_mode():
        await stop_ghost_monitor()
        # Clear trading state when stopping ghost mode
        ghost_state.clear_trading_state()
        await ws_manager.broadcast({
            "type": "state_cleared",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"message": "Trading state cleared"},
        })
        return {"success": True, "ghost_mode": False, "message": "Ghost Mode stopped, state cleared"}

    @app.post("/api/live-mode/start")
    async def api_start_live_mode():
        """Start live mode (requires wallet connection)."""
        ghost_state.is_live_mode = True
        ghost_state.clear_trading_state()  # Clear any ghost state
        await start_ghost_monitor()  # Same monitoring loop, different execution
        return {"success": True, "live_mode": True, "ghost_mode": False, "message": "Live Mode started"}

    @app.post("/api/live-mode/stop")
    async def api_stop_live_mode():
        await stop_ghost_monitor()
        ghost_state.clear_trading_state()
        await ws_manager.broadcast({
            "type": "state_cleared",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"message": "Trading state cleared"},
        })
        return {"success": True, "live_mode": False, "message": "Live Mode stopped, state cleared"}

    @app.get("/api/mode/status")
    async def api_mode_status():
        return {
            "enabled": ghost_state.enabled,
            "ghost_mode": ghost_state.enabled and not ghost_state.is_live_mode,
            "live_mode": ghost_state.is_live_mode,
            **ghost_state.get_status(),
        }

    @app.get("/api/missed-trades")
    async def api_get_missed_trades():
        """Get trades that were missed while the system was offline."""
        return {
            "count": len(ghost_state.missed_trades),
            "trades": ghost_state.missed_trades[:50],
        }

    # ==========================================================================
    # Account APIs
    # ==========================================================================

    @app.get("/api/accounts")
    async def get_accounts():
        return [acc.to_dict() for acc in ghost_state.accounts.values()]

    def convert_slippage_tiers(tiers_dict: Optional[Dict[str, float]]) -> Optional[List[Tuple[Decimal, Decimal]]]:
        """Convert slippage tiers from dict format {\"5\": 300} to list format [(0.05, 3.00)]."""
        if not tiers_dict:
            return None

        # Map price keys to decimal values
        price_map = {
            '5': Decimal('0.05'),
            '10': Decimal('0.10'),
            '20': Decimal('0.20'),
            '35': Decimal('0.35'),
            '50': Decimal('0.50'),
            '70': Decimal('0.70'),
            '85': Decimal('0.85'),
            '100': Decimal('1.00'),
        }

        tiers = []
        for price_key, slippage_pct in sorted(tiers_dict.items(), key=lambda x: int(x[0])):
            if price_key in price_map:
                # Convert percentage (e.g., 300) to decimal (3.00)
                tiers.append((price_map[price_key], Decimal(str(slippage_pct)) / 100))

        return tiers if tiers else None

    @app.post("/api/accounts")
    async def create_account(data: AccountCreate):
        # Convert slippage tiers from dict to list format
        slippage_tiers = convert_slippage_tiers(data.slippage_tiers)

        account = ghost_state.add_account(
            name=data.name,
            wallet=data.wallet,
            position_ratio=Decimal(str(data.position_ratio)),
            max_position_usd=Decimal(str(data.max_position_usd)),
            use_tiered_slippage=data.use_tiered_slippage,
            slippage_tiers=slippage_tiers,
            flat_slippage_tolerance=Decimal(str(data.flat_slippage_tolerance)),
            keywords=data.keywords,
            max_drawdown_percent=Decimal(str(data.max_drawdown_percent)),
            # Advanced risk settings
            take_profit_pct=Decimal(str(data.take_profit_pct)),
            stop_loss_pct=Decimal(str(data.stop_loss_pct)),
            max_concurrent=data.max_concurrent,
            max_holding_hours=data.max_holding_hours,
            min_liquidity=Decimal(str(data.min_liquidity)),
            cooldown_seconds=data.cooldown_seconds,
            order_type=data.order_type,
        )

        # Broadcast update
        await ws_manager.broadcast({
            "type": "accounts_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": [acc.to_dict() for acc in ghost_state.accounts.values()],
        })

        return account.to_dict()

    @app.put("/api/accounts/{account_id}")
    async def update_account(account_id: int, data: AccountUpdate):
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}

        # Convert slippage_tiers from dict to list format if present
        if 'slippage_tiers' in update_data:
            update_data['slippage_tiers'] = convert_slippage_tiers(update_data['slippage_tiers'])

        account = ghost_state.update_account(account_id, **update_data)
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")

        # Broadcast update
        await ws_manager.broadcast({
            "type": "accounts_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": [acc.to_dict() for acc in ghost_state.accounts.values()],
        })

        return account.to_dict()

    @app.delete("/api/accounts/{account_id}")
    async def delete_account(account_id: int):
        if account_id not in ghost_state.accounts:
            raise HTTPException(status_code=404, detail="Account not found")

        del ghost_state.accounts[account_id]

        await ws_manager.broadcast({
            "type": "accounts_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": [acc.to_dict() for acc in ghost_state.accounts.values()],
        })

        return {"success": True}

    # ==========================================================================
    # Blockchain Configuration APIs
    # ==========================================================================

    @app.get("/api/blockchain/status")
    async def get_blockchain_status():
        """Get blockchain monitoring status."""
        monitor = ghost_state._blockchain_monitor if hasattr(ghost_state, '_blockchain_monitor') else None
        return {
            "available": monitor is not None or hasattr(ghost_state, 'polygon_rpc_url'),
            "enabled": getattr(ghost_state, 'blockchain_enabled', False),
            "rpc_configured": getattr(ghost_state, 'polygon_rpc_url', None) is not None,
            "trades_detected": getattr(ghost_state, 'blockchain_trades_detected', 0),
            "avg_latency_ms": monitor.avg_latency_ms if monitor else 0,
            "blocks_processed": monitor.blocks_processed if monitor else 0,
            "last_block": monitor.last_block if monitor else 0,
            "wallets_monitored": len(monitor.wallets) if monitor else len(ghost_state.accounts),
        }

    @app.post("/api/blockchain/configure")
    async def configure_blockchain(config: dict):
        """
        Configure Polygon RPC URL for blockchain monitoring.

        This enables ~2-5s detection latency vs ~15-21s with API polling.

        Body: {"rpc_url": "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"}
        """
        rpc_url = config.get("rpc_url")
        if not rpc_url:
            raise HTTPException(status_code=400, detail="rpc_url is required")

        ghost_state.set_polygon_rpc(rpc_url)

        await ws_manager.broadcast({
            "type": "blockchain_config",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"rpc_configured": True},
        })

        return {
            "success": True,
            "message": "RPC URL configured. Restart ghost mode to enable blockchain monitoring.",
            "rpc_url": rpc_url[:50] + "...",
        }

    @app.post("/api/blockchain/test")
    async def test_blockchain_connection(config: dict):
        """Test a Polygon RPC connection without saving it."""
        rpc_url = config.get("rpc_url")
        if not rpc_url:
            raise HTTPException(status_code=400, detail="rpc_url is required")

        try:
            from web3 import Web3
            # Handle different web3 versions
            try:
                from web3.middleware import ExtraDataToPOAMiddleware as poa_middleware
            except ImportError:
                from web3.middleware import geth_poa_middleware as poa_middleware

            # Convert WebSocket URL to HTTP for connection test
            if rpc_url.startswith("wss://"):
                http_url = rpc_url.replace("wss://", "https://")
            elif rpc_url.startswith("ws://"):
                http_url = rpc_url.replace("ws://", "http://")
            else:
                http_url = rpc_url

            w3 = Web3(Web3.HTTPProvider(http_url, request_kwargs={'timeout': 10}))
            w3.middleware_onion.inject(poa_middleware, layer=0)

            if w3.is_connected():
                block_number = w3.eth.block_number
                return {
                    "success": True,
                    "message": "Connection successful",
                    "block_number": block_number,
                    "chain_id": w3.eth.chain_id,
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to connect to RPC endpoint",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ==========================================================================
    # Position & Trade APIs
    # ==========================================================================

    @app.get("/api/positions")
    async def get_positions():
        # In LIVE MODE: Fetch positions directly from Polymarket API
        if ghost_state.is_live_mode and ghost_state.accounts:
            all_positions = []
            for account in ghost_state.accounts.values():
                try:
                    live_positions = get_live_positions(account.wallet)
                    for pos in live_positions:
                        if pos.get("status") == "open" and pos.get("size", 0) > 0:
                            pos["account_name"] = account.name
                            pos["account_id"] = account.id
                            all_positions.append(pos)
                except Exception as e:
                    print(f"[Live Positions] Error fetching for {account.name}: {e}")
            return all_positions

        # In GHOST MODE: Return simulated positions (filter out zero-size phantom entries)
        return [
            {**p, "size": float(p["size"]), "avg_price": float(p["avg_price"]), "cost_basis": float(p["cost_basis"])}
            for p in ghost_state.positions.values()
            if p["status"] == "open" and float(p["size"]) > 0
        ]

    @app.get("/api/positions/debug")
    async def get_positions_debug():
        """Debug endpoint to see all positions including closed ones."""
        return [
            {**p, "size": float(p["size"]), "avg_price": float(p["avg_price"]), "cost_basis": float(p["cost_basis"])}
            for p in ghost_state.positions.values()
        ]

    @app.get("/api/positions/summary")
    async def get_positions_summary():
        # In LIVE MODE: Fetch summary from Polymarket API
        if ghost_state.is_live_mode and ghost_state.accounts:
            total_count = 0
            total_value = 0
            total_unrealized_pnl = 0

            for account in ghost_state.accounts.values():
                try:
                    portfolio = get_live_portfolio_value(account.wallet)
                    total_count += portfolio.get("positions_count", 0)
                    total_value += portfolio.get("total_value", 0)
                    total_unrealized_pnl += portfolio.get("unrealized_pnl", 0)
                except Exception as e:
                    print(f"[Live Summary] Error fetching for {account.name}: {e}")

            return {
                "count": total_count,
                "total_value": total_value,
                "total_unrealized_pnl": total_unrealized_pnl,
            }

        # In GHOST MODE: Return simulated summary (filter out zero-size)
        open_positions = [p for p in ghost_state.positions.values() if p["status"] == "open" and float(p["size"]) > 0]
        total_value = sum(float(p["size"]) * float(p.get("current_price", p["avg_price"])) for p in open_positions)
        return {
            "count": len(open_positions),
            "total_value": total_value,
            "total_unrealized_pnl": float(ghost_state.simulated_pnl),
        }

    @app.get("/api/trades")
    async def get_trades(limit: int = 20):
        return [t.to_dict() for t in ghost_state.ghost_trades[:limit]]

    @app.get("/api/trades/stats")
    async def get_trade_stats():
        status = ghost_state.get_status()
        return {
            "total_trades": status["trades_detected"],
            "would_execute": status["trades_would_execute"],
            "filtered_keyword": status["trades_filtered_keyword"],
            "filtered_stoploss": status["trades_filtered_stoploss"],
            "filtered_slippage": status["trades_filtered_slippage"],
            "filtered_limit": status.get("trades_filtered_limit", 0),
            "win_rate": 0.6,  # Placeholder
            "avg_latency_ms": status["avg_detection_ms"],
            "avg_slippage": 0.005,
        }

    # ==========================================================================
    # P&L APIs
    # ==========================================================================

    @app.get("/api/pnl")
    async def get_pnl():
        # In LIVE MODE: Fetch P&L directly from Polymarket API
        if ghost_state.is_live_mode and ghost_state.accounts:
            total_realized_pnl = 0
            total_unrealized_pnl = 0
            total_invested = 0
            total_value = 0

            for account in ghost_state.accounts.values():
                try:
                    portfolio = get_live_portfolio_value(account.wallet)
                    total_realized_pnl += portfolio.get("realized_pnl", 0)
                    total_unrealized_pnl += portfolio.get("unrealized_pnl", 0)
                    total_invested += portfolio.get("total_invested", 0)
                    total_value += portfolio.get("total_value", 0)
                except Exception as e:
                    print(f"[Live P&L] Error fetching for {account.name}: {e}")

            roi_percent = (total_realized_pnl + total_unrealized_pnl) / total_invested * 100 if total_invested > 0 else 0
            return {
                "realized_pnl": total_realized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "total_invested": total_invested,
                "total_value": total_value,
                "roi_percent": roi_percent,
            }

        # In GHOST MODE: Return simulated P&L
        return {
            "realized_pnl": float(ghost_state.simulated_pnl),
            "unrealized_pnl": 0,
            "total_invested": float(Decimal("10000") - ghost_state.simulated_balance),
            "roi_percent": float(ghost_state.simulated_pnl / ghost_state.simulated_balance * 100) if ghost_state.simulated_balance > 0 else 0,
        }

    @app.get("/api/pnl/daily")
    async def get_daily_pnl(days: int = 30):
        # Return empty for ghost mode (no historical data)
        return []

    # ==========================================================================
    # Latency APIs
    # ==========================================================================

    @app.get("/api/latency")
    async def get_latency():
        det_lats = ghost_state.detection_latencies[-50:] if ghost_state.detection_latencies else []
        api_lats = ghost_state.api_latencies[-50:] if ghost_state.api_latencies else []

        def stats(data):
            if not data:
                return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}
            sorted_data = sorted(data)
            n = len(sorted_data)
            return {
                "count": n,
                "avg_ms": round(sum(data) / n, 2),
                "recent_avg_ms": round(sum(data[-20:]) / len(data[-20:]), 2) if len(data) >= 20 else round(sum(data) / n, 2),
                "min_ms": round(min(data), 2),
                "max_ms": round(max(data), 2),
                "p95_ms": round(sorted_data[int(n * 0.95)] if n >= 20 else max(data), 2),
            }

        return {
            "stages": {
                "detection": stats(det_lats),
                "e2e": stats(api_lats),
            },
            "health": {"status": "healthy", "health_score": 100},
        }

    # ==========================================================================
    # Bot Control APIs (for compatibility)
    # ==========================================================================

    @app.post("/api/control/{action}")
    async def control_bot(action: str):
        if action == "start":
            await start_ghost_monitor()
            return {"success": True, "action": action, "status": "running", "ghost_mode": True}
        elif action == "stop":
            await stop_ghost_monitor()
            return {"success": True, "action": action, "status": "stopped", "ghost_mode": False}
        elif action == "pause":
            await stop_ghost_monitor()
            return {"success": True, "action": action, "status": "paused"}
        elif action == "resume":
            await start_ghost_monitor()
            return {"success": True, "action": action, "status": "running"}
        return {"success": False, "action": action}

    # ==========================================================================
    # Infrastructure & Proxy APIs
    # ==========================================================================

    @app.get("/infrastructure", response_class=HTMLResponse)
    async def infrastructure_page():
        """Infrastructure and proxy configuration page."""
        template_path = PROJECT_ROOT / "src" / "web" / "templates" / "infrastructure.html"
        if template_path.exists():
            return template_path.read_text()
        return "<h1>Infrastructure template not found</h1>"

    # ==========================================================================
    # Discovery Page & API
    # ==========================================================================

    @app.get("/discovery", response_class=HTMLResponse)
    async def discovery_page():
        """Account discovery page."""
        template_path = PROJECT_ROOT / "src" / "web" / "templates" / "discovery.html"
        if template_path.exists():
            return template_path.read_text()
        return "<h1>Discovery template not found</h1>"

    # Include discovery API router if available
    if DISCOVERY_AVAILABLE:
        app.include_router(discovery_router, prefix="/api")

    @app.get("/api/infrastructure/info")
    async def api_infrastructure_info():
        """Get comprehensive infrastructure information."""
        return get_infrastructure_info()

    @app.get("/api/infrastructure/geo-check")
    async def api_geo_check():
        """Check if current IP is geo-restricted."""
        proxy_config = get_proxy_config()
        return check_geo_restriction(proxy_config if proxy_config.enabled else None)

    @app.post("/api/infrastructure/benchmark")
    async def api_run_benchmark():
        """Run latency benchmark against Polymarket endpoints."""
        import asyncio
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        proxy_config = get_proxy_config()
        results = await loop.run_in_executor(
            None,
            benchmark_all_endpoints,
            proxy_config if proxy_config.enabled else None,
            10,  # 10 samples
        )
        return {name: result.to_dict() for name, result in results.items()}

    @app.get("/api/infrastructure/proxy")
    async def api_get_proxy_config():
        """Get current proxy configuration."""
        config = get_proxy_config()
        return config.to_dict()

    @app.put("/api/infrastructure/proxy")
    async def api_update_proxy_config(data: dict):
        """Update proxy configuration."""
        try:
            # Convert proxy_type string to enum
            proxy_type_str = data.get("proxy_type", "none")
            try:
                proxy_type = ProxyTypeEnum(proxy_type_str)
            except ValueError:
                proxy_type = ProxyTypeEnum.NONE

            config = ProxyConfig(
                enabled=data.get("enabled", False),
                proxy_type=proxy_type,
                host=data.get("host", ""),
                port=data.get("port", 0),
                username=data.get("username"),
                password=data.get("password"),
                timeout_seconds=data.get("timeout_seconds", 5.0),
                max_retries=data.get("max_retries", 3),
                pool_connections=data.get("pool_connections", 10),
                pool_maxsize=data.get("pool_maxsize", 20),
                keep_alive=data.get("keep_alive", True),
            )
            save_proxy_config(config)
            return {"success": True, "message": "Proxy configuration saved"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/infrastructure/proxy/test")
    async def api_test_proxy():
        """Test proxy connection."""
        import time
        proxy_config = get_proxy_config()

        try:
            start = time.perf_counter()
            geo_result = check_geo_restriction(proxy_config if proxy_config.enabled else None)
            latency = (time.perf_counter() - start) * 1000

            return {
                "success": True,
                "latency_ms": round(latency, 2),
                "geo_status": "Blocked" if geo_result.get("restricted") else "Allowed",
                "country": geo_result.get("country", "Unknown"),
                "ip": geo_result.get("ip", "Unknown"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==========================================================================
    # WebSocket
    # ==========================================================================

    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("command") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                elif data.get("command") == "subscribe":
                    await websocket.send_json({"type": "subscribed", "channel": data.get("channel")})
                elif data.get("command") == "get_positions":
                    # In LIVE MODE: Fetch positions from Polymarket API
                    if ghost_state.is_live_mode and ghost_state.accounts:
                        all_positions = []
                        for account in ghost_state.accounts.values():
                            try:
                                live_positions = get_live_positions(account.wallet)
                                for pos in live_positions:
                                    if pos.get("status") == "open" and pos.get("size", 0) > 0:
                                        pos["account_name"] = account.name
                                        all_positions.append(pos)
                            except Exception:
                                pass
                        await websocket.send_json({
                            "type": "positions",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": all_positions,
                        })
                    else:
                        # In GHOST MODE: Return simulated positions (filter out zero-size)
                        await websocket.send_json({
                            "type": "positions",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": [
                                {**p, "size": float(p["size"]), "avg_price": float(p["avg_price"])}
                                for p in ghost_state.positions.values()
                                if p["status"] == "open" and float(p["size"]) > 0
                            ],
                        })
                elif data.get("command") == "get_trades":
                    await websocket.send_json({
                        "type": "trades",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": [t.to_dict() for t in ghost_state.ghost_trades[:20]],
                    })
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    # ==========================================================================
    # Startup
    # ==========================================================================

    print("\n" + "="*70)
    print("  PolymarketBot Ghost Mode Dashboard")
    print("="*70)
    print(f"\n  Dashboard URL: \033[1;36mhttp://localhost:{port}\033[0m")
    print(f"\n  Tracked Accounts:")
    for acc in ghost_state.accounts.values():
        kw = ", ".join(acc.keywords[:3]) if acc.keywords else "All markets"
        print(f"    - {acc.name}: {acc.wallet[:10]}...")
        print(f"      Keywords: {kw}")
        print(f"      Max Drawdown: {acc.max_drawdown_percent}%")
    print(f"\n  Mode: GHOST MODE (Real monitoring, simulated execution)")
    print(f"  No wallet connected - API calls will fail but timing is measured")
    print("\n  Press Ctrl+C to stop")
    print("\n" + "-"*70)
    print("  Toggle Ghost Mode in the UI to start monitoring real trades")
    print("-"*70 + "\n")

    # Run server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except asyncio.CancelledError:
        pass
    finally:
        await stop_ghost_monitor()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ghost Mode Dashboard - Real-time copy trade simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ghost_mode.py                    # Default port 8765
    python run_ghost_mode.py --port 9000        # Custom port
    PORT=9000 python run_ghost_mode.py          # Via environment variable
        """
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the dashboard on (default: {DEFAULT_PORT}, or PORT env var)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_dashboard(port=args.port))
    except KeyboardInterrupt:
        print("\n\nGhost Mode Dashboard stopped.")
