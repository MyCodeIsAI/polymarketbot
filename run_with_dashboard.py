#!/usr/bin/env python3
"""
Full simulation with Web Dashboard.

Starts the web dashboard and runs a realistic simulation,
pushing updates to the dashboard in real-time via WebSocket.

Access the dashboard at: http://localhost:8000
"""

import asyncio
import sys
import signal
import random
import time
import uuid
from collections import deque
from datetime import datetime
from decimal import Decimal
from statistics import mean
from typing import Dict, Any, List, Optional
from pathlib import Path

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# Sample markets for realistic simulation
SAMPLE_MARKETS = [
    {"id": "0x001", "name": "Will Trump win the 2024 election?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.52, "No": 0.48}},
    {"id": "0x002", "name": "Will Fed cut rates in Q1 2025?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.65, "No": 0.35}},
    {"id": "0x003", "name": "Will Bitcoin reach $100K in 2024?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.42, "No": 0.58}},
    {"id": "0x004", "name": "Super Bowl LIX Winner", "outcomes": ["Chiefs", "Eagles"], "prices": {"Chiefs": 0.55, "Eagles": 0.45}},
    {"id": "0x005", "name": "Best Picture Oscar 2025", "outcomes": ["Oppenheimer", "Other"], "prices": {"Oppenheimer": 0.68, "Other": 0.32}},
    {"id": "0x006", "name": "Will AI pass the Turing test by 2025?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.35, "No": 0.65}},
    {"id": "0x007", "name": "Nvidia stock above $150 by EOY?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.72, "No": 0.28}},
    {"id": "0x008", "name": "Will there be a government shutdown?", "outcomes": ["Yes", "No"], "prices": {"Yes": 0.25, "No": 0.75}},
]


class LatencyTracker:
    """Track latencies for simulation with realistic metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._measurements: Dict[str, deque] = {
            "detection": deque(maxlen=window_size),
            "validation": deque(maxlen=window_size),
            "signing": deque(maxlen=window_size),
            "submission": deque(maxlen=window_size),
            "e2e": deque(maxlen=window_size),
        }

    def record(self, stage: str, latency_ms: float):
        """Record a latency measurement."""
        if stage in self._measurements:
            self._measurements[stage].append(latency_ms)

    def get_stats(self, stage: str) -> Dict[str, float]:
        """Get statistics for a stage."""
        data = list(self._measurements.get(stage, []))
        if not data:
            return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}

        sorted_data = sorted(data)
        n = len(sorted_data)
        return {
            "count": n,
            "avg_ms": round(mean(data), 2),
            "recent_avg_ms": round(mean(data[-20:]) if len(data) >= 20 else mean(data), 2),
            "min_ms": round(min(data), 2),
            "max_ms": round(max(data), 2),
            "p50_ms": round(sorted_data[n // 2], 2),
            "p95_ms": round(sorted_data[int(n * 0.95)] if n >= 20 else max(data), 2),
            "p99_ms": round(sorted_data[int(n * 0.99)] if n >= 100 else max(data), 2),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all stage statistics."""
        return {stage: self.get_stats(stage) for stage in self._measurements}

    def get_health(self) -> Dict[str, Any]:
        """Get health assessment."""
        e2e_data = list(self._measurements.get("e2e", []))
        if not e2e_data:
            return {"status": "unknown", "health_score": 100}

        avg = mean(e2e_data)
        # Target: <200ms, Warning: <500ms, Critical: >500ms
        if avg < 150:
            return {"status": "healthy", "health_score": 100}
        elif avg < 200:
            return {"status": "healthy", "health_score": 90}
        elif avg < 300:
            return {"status": "warning", "health_score": 70}
        elif avg < 500:
            return {"status": "degraded", "health_score": 50}
        else:
            return {"status": "critical", "health_score": 20}


class SimulatedBotState:
    """Simulated bot state that mimics real bot behavior."""

    def __init__(self):
        self.status = "stopped"
        self.uptime_start = None
        self.positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.balance = Decimal("10000.00")
        self.available_balance = Decimal("10000.00")
        self.total_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.trade_count = 0
        self.win_count = 0
        self.latency_tracker = LatencyTracker()
        self.accounts = [
            {
                "id": 1,
                "name": "automatedaitradingbot",
                "target_wallet": "0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11",
                "position_ratio": "0.01",
                "max_position_usd": "500",
                "slippage_tolerance": "0.05",
                "enabled": True,
            }
        ]

    def get_state(self) -> Dict[str, Any]:
        """Get current state for API."""
        uptime = 0
        if self.uptime_start:
            uptime = (datetime.utcnow() - self.uptime_start).total_seconds()

        return {
            "status": self.status,
            "uptime": uptime,
            "balance": float(self.balance),
            "available_balance": float(self.available_balance),
            "positions_count": len([p for p in self.positions if p.get("status") == "open"]),
            "positions_value": sum(float(p.get("value", 0)) for p in self.positions if p.get("status") == "open"),
            "total_pnl": float(self.total_pnl + self.unrealized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "trade_count": self.trade_count,
        }

    def start(self):
        """Start the bot."""
        self.status = "running"
        self.uptime_start = datetime.utcnow()

    def stop(self):
        """Stop the bot."""
        self.status = "stopped"
        self.uptime_start = None

    def add_trade(self, trade: Dict[str, Any]):
        """Add a trade and update positions."""
        self.trades.insert(0, trade)
        if len(self.trades) > 100:
            self.trades.pop()

        self.trade_count += 1

        # Update balance
        trade_value = Decimal(str(trade.get("size", 0))) * Decimal(str(trade.get("price", 0)))
        if trade.get("side") == "BUY":
            self.available_balance -= trade_value
        else:
            self.available_balance += trade_value

        # Update or create position
        # Match on market_id + outcome (a Yes position is different from a No position)
        market_id = trade.get("market_id")
        outcome = trade.get("outcome")
        position_key = f"{market_id}_{outcome}"

        existing = next(
            (p for p in self.positions
             if p.get("market_id") == market_id and p.get("outcome") == outcome and p.get("status") == "open"),
            None
        )

        if existing and trade.get("side") == "SELL":
            # SELL reduces an existing long position in the same outcome
            existing["size"] = float(Decimal(str(existing.get("size", 0))) - Decimal(str(trade.get("size", 0))))

            if existing["size"] <= 0.01:  # Close if nearly zero
                existing["status"] = "closed"
                existing["size"] = 0
            existing["current_price"] = trade.get("price")
            existing["value"] = existing["size"] * existing["current_price"]
        elif existing and trade.get("side") == "BUY":
            # BUY adds to existing position
            old_size = Decimal(str(existing.get("size", 0)))
            old_avg = Decimal(str(existing.get("average_price", 0)))
            new_size = Decimal(str(trade.get("size", 0)))
            new_price = Decimal(str(trade.get("price", 0)))

            # Calculate new average price
            total_cost = (old_size * old_avg) + (new_size * new_price)
            total_size = old_size + new_size
            new_avg = total_cost / total_size if total_size > 0 else new_price

            existing["size"] = float(total_size)
            existing["average_price"] = float(new_avg)
            existing["current_price"] = float(new_price)
            existing["value"] = float(total_size * new_price)
        else:
            # New position (BUY creates long position)
            if trade.get("side") == "BUY":
                self.positions.append({
                    "id": len(self.positions) + 1,
                    "market_id": market_id,
                    "market_name": trade.get("market_name"),
                    "outcome": outcome,
                    "side": "LONG",
                    "size": float(trade.get("size", 0)),
                    "average_price": float(trade.get("price", 0)),
                    "current_price": float(trade.get("price", 0)),
                    "value": float(trade.get("size", 0)) * float(trade.get("price", 0)),
                    "unrealized_pnl": 0,
                    "status": "open",
                })

        # Simulate some P&L
        if random.random() > 0.4:  # 60% win rate
            self.win_count += 1
            pnl = trade_value * Decimal(str(random.uniform(0.02, 0.15)))
        else:
            pnl = -trade_value * Decimal(str(random.uniform(0.01, 0.08)))

        self.total_pnl += pnl


# Global state
bot_state = SimulatedBotState()


async def generate_trades(ws_manager):
    """Generate trades and broadcast to WebSocket clients."""
    print("\n  [Trade Generator] Starting...")

    # Start the bot
    bot_state.start()
    await ws_manager.broadcast({
        "type": "status",
        "timestamp": datetime.utcnow().isoformat(),
        "data": bot_state.get_state(),
    })

    trade_num = 0
    while True:
        try:
            # Random delay between trades (3-15 seconds for demo)
            delay = random.uniform(3, 15)
            await asyncio.sleep(delay)

            # Generate a trade
            trade_num += 1
            market = random.choice(SAMPLE_MARKETS)
            outcome = random.choice(market["outcomes"])
            side = "BUY" if random.random() < 0.55 else "SELL"
            size = random.uniform(50, 500)
            price = market["prices"].get(outcome, 0.5)
            price = price + random.uniform(-0.03, 0.03)  # Price movement
            price = max(0.01, min(0.99, price))

            slippage = random.uniform(0, 0.01)

            # Simulate realistic latencies using perf_counter
            t_start = time.perf_counter()

            # Detection (WebSocket receive + parse): 20-80ms
            detection_ms = random.uniform(20, 80)
            bot_state.latency_tracker.record("detection", detection_ms)

            # Validation: 2-8ms
            validation_ms = random.uniform(2, 8)
            bot_state.latency_tracker.record("validation", validation_ms)

            # Signing (EIP-712): 10-25ms
            signing_ms = random.uniform(10, 25)
            bot_state.latency_tracker.record("signing", signing_ms)

            # Submission (HTTP POST): 40-120ms
            submission_ms = random.uniform(40, 120)
            bot_state.latency_tracker.record("submission", submission_ms)

            # E2E total
            e2e_ms = detection_ms + validation_ms + signing_ms + submission_ms
            bot_state.latency_tracker.record("e2e", e2e_ms)

            trade = {
                "id": f"trade_{uuid.uuid4().hex[:8]}",
                "trade_id": f"sim_{trade_num}",
                "account_id": 1,
                "account_name": "automatedaitradingbot",
                "market_id": market["id"],
                "market_name": market["name"],
                "outcome": outcome,
                "side": side,
                "target_size": size,
                "target_price": price,
                "execution_size": size,
                "execution_price": price * (1 + slippage if side == "BUY" else 1 - slippage),
                "size": size,
                "price": price,
                "slippage_percent": slippage,
                "status": "filled",
                "detected_at": datetime.utcnow().isoformat(),
                "executed_at": datetime.utcnow().isoformat(),
                "total_latency_ms": round(e2e_ms, 2),
                "latency_breakdown": {
                    "detection_ms": round(detection_ms, 2),
                    "validation_ms": round(validation_ms, 2),
                    "signing_ms": round(signing_ms, 2),
                    "submission_ms": round(submission_ms, 2),
                },
            }

            # Update bot state
            bot_state.add_trade(trade)

            # Broadcast trade
            await ws_manager.broadcast({
                "type": "trade",
                "timestamp": datetime.utcnow().isoformat(),
                "data": trade,
            })

            # Broadcast updated status
            await ws_manager.broadcast({
                "type": "status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": bot_state.get_state(),
            })

            # Broadcast position update (this triggers UI refresh)
            open_positions = [p for p in bot_state.positions if p.get("status") == "open"]
            await ws_manager.broadcast({
                "type": "position_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "positions": open_positions,
                    "count": len(open_positions),
                    "total_value": sum(p.get("value", 0) for p in open_positions),
                },
            })

            # Broadcast latency update (CRITICAL for monitoring)
            await ws_manager.broadcast({
                "type": "latency_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "stages": bot_state.latency_tracker.get_all_stats(),
                    "health": bot_state.latency_tracker.get_health(),
                    "last_trade": {
                        "e2e_ms": round(e2e_ms, 2),
                        "detection_ms": round(detection_ms, 2),
                        "submission_ms": round(submission_ms, 2),
                    },
                },
            })

            # Log to console with latency
            side_color = "\033[32m" if side == "BUY" else "\033[31m"
            latency_color = "\033[32m" if e2e_ms < 150 else "\033[33m" if e2e_ms < 300 else "\033[31m"
            print(f"  [Trade #{trade_num}] {side_color}{side}\033[0m ${size:.2f} @ {price:.4f} | {latency_color}{e2e_ms:.0f}ms\033[0m - {market['name'][:35]}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [Error] {e}")


async def run_dashboard():
    """Run the web dashboard with simulation."""
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path

    # Create FastAPI app
    app = FastAPI(title="PolymarketBot Dashboard (Simulation)")

    # Connection manager for WebSocket
    class ConnectionManager:
        def __init__(self):
            self.active_connections: set = set()

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.add(websocket)
            # Send initial state
            await websocket.send_json({
                "type": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to PolymarketBot Simulation",
            })
            await websocket.send_json({
                "type": "initial_status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": bot_state.get_state(),
            })
            await websocket.send_json({
                "type": "initial_positions",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"count": len(bot_state.positions), "positions": bot_state.positions[:10]},
            })

        def disconnect(self, websocket: WebSocket):
            self.active_connections.discard(websocket)

        async def broadcast(self, message: dict):
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.add(connection)
            for conn in disconnected:
                self.active_connections.discard(conn)

    manager = ConnectionManager()

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

    # API endpoints
    @app.get("/api/status")
    async def get_status():
        return bot_state.get_state()

    @app.get("/api/accounts")
    async def get_accounts():
        return bot_state.accounts

    @app.get("/api/positions")
    async def get_positions():
        return [p for p in bot_state.positions if p.get("status") == "open"]

    @app.get("/api/positions/summary")
    async def get_positions_summary():
        open_positions = [p for p in bot_state.positions if p.get("status") == "open"]
        return {
            "count": len(open_positions),
            "total_value": sum(p.get("value", 0) for p in open_positions),
            "total_unrealized_pnl": float(bot_state.unrealized_pnl),
        }

    @app.get("/api/trades")
    async def get_trades(limit: int = 20):
        return bot_state.trades[:limit]

    @app.get("/api/trades/stats")
    async def get_trade_stats():
        return {
            "total_trades": bot_state.trade_count,
            "win_rate": bot_state.win_count / bot_state.trade_count if bot_state.trade_count > 0 else 0,
            "avg_latency_ms": 85,
            "avg_slippage": 0.005,
        }

    @app.get("/api/pnl")
    async def get_pnl():
        return {
            "realized_pnl": float(bot_state.total_pnl),
            "unrealized_pnl": float(bot_state.unrealized_pnl),
            "total_invested": float(bot_state.balance - bot_state.available_balance),
            "roi_percent": float(bot_state.total_pnl / bot_state.balance * 100) if bot_state.balance > 0 else 0,
        }

    @app.get("/api/pnl/daily")
    async def get_daily_pnl(days: int = 30):
        # Generate mock daily P&L data
        from datetime import timedelta
        data = []
        cumulative = 0
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days-i-1)).strftime("%Y-%m-%d")
            daily = random.uniform(-50, 100)
            cumulative += daily
            data.append({"date": date, "daily_pnl": daily, "cumulative_pnl": cumulative})
        return data

    @app.get("/api/health")
    async def get_health():
        return {
            "api_healthy": True,
            "websocket_healthy": True,
            "database_healthy": True,
            "circuit_breaker_open": False,
            "memory_percent": random.uniform(30, 50),
            "cpu_percent": random.uniform(10, 30),
        }

    @app.get("/api/latency")
    async def get_latency():
        """Get latency metrics - CRITICAL for copy trading performance."""
        return {
            "stages": bot_state.latency_tracker.get_all_stats(),
            "health": bot_state.latency_tracker.get_health(),
            "thresholds": {
                "e2e_target_ms": 200,
                "e2e_warning_ms": 300,
                "e2e_critical_ms": 500,
                "detection_target_ms": 50,
                "submission_target_ms": 100,
            },
        }

    @app.post("/api/control/{action}")
    async def control_bot(action: str):
        if action == "start":
            bot_state.start()
        elif action == "stop":
            bot_state.stop()
        elif action == "pause":
            bot_state.status = "paused"
        elif action == "resume":
            bot_state.status = "running"
        return {"success": True, "action": action, "status": bot_state.status}

    # WebSocket endpoint
    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("command") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                elif data.get("command") == "subscribe":
                    await websocket.send_json({"type": "subscribed", "channel": data.get("channel")})
                elif data.get("command") == "get_positions":
                    await websocket.send_json({
                        "type": "positions",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": bot_state.positions,
                    })
                elif data.get("command") == "get_trades":
                    await websocket.send_json({
                        "type": "trades",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": bot_state.trades[:20],
                    })
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # Print startup message
    print("\n" + "="*60)
    print("  PolymarketBot Dashboard (Simulation Mode)")
    print("="*60)
    print(f"\n  Dashboard URL: \033[1;36mhttp://localhost:8000\033[0m")
    print(f"  Target Wallet: 0xd8f8...0f11 (@automatedaitradingbot)")
    print(f"  Mode: Synthetic Trade Generation")
    print("\n  Press Ctrl+C to stop\n")
    print("-"*60)

    # Start trade generator in background
    trade_task = asyncio.create_task(generate_trades(manager))

    # Run the server
    # Bind to localhost only for security - access via SSH tunnel
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except asyncio.CancelledError:
        pass
    finally:
        trade_task.cancel()
        try:
            await trade_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        print("\n\nSimulation stopped.")
