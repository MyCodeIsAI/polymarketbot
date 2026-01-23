# Polymarket Copy-Trading Bot: Multi-Stage Execution Plan

## Executive Summary

This document outlines a comprehensive multi-stage execution plan for building a high-performance Polymarket copy-trading bot. The system will monitor target wallets, detect position changes in real-time, and execute proportional copy-trades with configurable slippage protection and position sizing.

---

## Critical Architecture Decisions Based on Research

### Infrastructure Reality Check

| Component | Location/Specification | Latency Impact |
|-----------|----------------------|----------------|
| **Polymarket CLOB Servers** | AWS eu-west-2 (London) | Primary target |
| **Backup Region** | AWS eu-west-1 (Ireland) | ~280 miles, fiber-connected |
| **WebSocket Feed Latency** | ~100ms typical | Best real-time option |
| **Gamma API Latency** | ~1 second | Too slow for copy-trading |
| **Co-located VPS (London)** | 1-5ms | Optimal for speed |
| **Professional co-location** | 0.36-0.56ms | Maximum performance |

### Key APIs We Will Use

| API | Base URL | Purpose |
|-----|----------|---------|
| **Data API** | `https://data-api.polymarket.com` | Track target wallet positions/activity |
| **CLOB API** | `https://clob.polymarket.com` | Execute orders, manage positions |
| **CLOB WebSocket** | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Real-time order book, fills |
| **Gamma API** | `https://gamma-api.polymarket.com` | Market metadata (non-latency-critical) |

### Rate Limits We Must Respect

| Endpoint | Limit | Implication |
|----------|-------|-------------|
| `/positions` | 150 req/10s | Can poll 15 wallets/second max |
| `/trades` | 200 req/10s | 20 requests/second |
| `/activity` | Under general 1000/10s | ~100/second |
| POST `/order` | 3,500 burst/10s | Plenty of headroom |
| WebSocket | 500 instruments/connection | Multiple connections if needed |

---

## Identified Bottlenecks & Speed Considerations

### 1. **Position Detection Latency** (CRITICAL)
- **Problem**: No WebSocket for target wallet positions - must poll Data API
- **Best case**: Poll `/activity` endpoint every ~100ms (within rate limits)
- **Mitigation**: Use `/activity` with `type=TRADE` filter, timestamp tracking
- **Alternative**: Monitor Polygon blockchain events directly via RPC (sub-second detection)

### 2. **Order Execution Latency**
- **Problem**: VPN routing adds latency
- **Mitigation**:
  - Use VPN exit node in Amsterdam/London (closest to Polymarket servers)
  - Consider split-tunnel: VPN for API calls only, direct for other traffic
  - Use connection pooling and keep-alive connections

### 3. **Price Movement During Detection-to-Execution Window**
- **Problem**: Price can move significantly in 100-500ms
- **Mitigation**: Slippage protection with configurable thresholds per account

### 4. **Processing Overhead**
- **Mitigation**:
  - Use Rust or Python with async/await (no GIL blocking)
  - Pre-compute order signatures
  - Maintain local order book state via WebSocket
  - Use connection pools for API calls

---

## Critical Considerations You May Have Overlooked

### 1. **Proxy Wallet Architecture**
Polymarket users don't trade directly from their EOA wallet. A 1-of-1 multisig proxy wallet is deployed:
- **Gnosis Safe** for MetaMask/browser wallets
- **Custom Proxy** for MagicLink/email users

**Implication**: You need to track the PROXY wallet address (shown on Polymarket.com), not the signing EOA. The target accounts you're copying likely use proxy wallets.

### 2. **Token Allowances Setup**
Before your bot can trade, you must approve:
- USDC contract for CTF Exchange
- Conditional Token Framework for both CTF exchanges
- This is a one-time setup but MUST be done before first trade

### 3. **Negative Risk Markets**
Polymarket has two exchange types:
- **CTF Exchange** (`0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e`): Simple binary YES/NO
- **NegRisk CTF Exchange** (`0xc5d563a36ae78145c45a50134d48a1215220f80a`): Multi-outcome markets

Your bot must detect which exchange a market uses and route orders accordingly.

### 4. **Order Signing Complexity**
Orders require EIP-712 signatures. The signature must include:
- Token ID, price, size, side
- Expiration timestamp
- Nonce management

**Pre-signing orders is NOT possible** because price/size change per trade. However, you can:
- Pre-fetch market metadata
- Maintain hot wallet with unlocked key in memory
- Use efficient signing libraries

### 5. **Position Sizing Edge Cases**
Your 1:100 ratio example needs careful handling:
- **Minimum order sizes**: Polymarket has minimums; 1/100th of a small position may be too small
- **Rounding**: Share quantities may need rounding
- **Dust positions**: Very small positions may not be worth copying
- **Balance checks**: Must verify sufficient USDC before each order

### 6. **Exit Position Handling**
When target closes a position:
- Detect the SELL activity
- Calculate proportional exit size
- Handle partial exits (target may scale out)
- Handle market resolution (positions auto-settle)

### 7. **Failed Order Recovery**
What if your order fails (insufficient balance, slippage exceeded, rate limit)?
- Queue for retry with backoff
- Alert mechanism
- Position drift tracking (your position vs. target's)

### 8. **VPN Considerations**
- Polymarket blocks US/UK entirely
- VPN exit node location affects latency
- Choose exit node in Netherlands/Germany/Ireland for best balance
- Some VPNs have unstable connections - need reconnection logic

### 9. **Multiple Target Account Conflicts**
If two targets trade the same market simultaneously:
- Order aggregation or sequential execution?
- Position accounting per target must remain separate
- Total exposure limits across all targets?

### 10. **Market Hours & Liquidity**
- Prediction markets have varying liquidity
- Some markets are nearly illiquid
- Need liquidity checks before executing large orders

---

## Recommended Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Core Runtime** | Python 3.11+ with asyncio | Official SDK support, rapid development |
| **HTTP Client** | `httpx` with connection pooling | Async, fast, maintains connections |
| **WebSocket** | `websockets` library | Native async support |
| **Signing** | `eth-account`, `web3.py` | EIP-712 signature support |
| **Database** | SQLite (development) → PostgreSQL (production) | Position tracking, audit logs |
| **Config** | YAML files + environment variables | Secrets in env, config in files |
| **Queue** | In-memory asyncio.Queue | Order execution queue |
| **Monitoring** | Prometheus metrics + Grafana | Latency, success rates |
| **Logging** | Structured JSON logging | Audit trail, debugging |

---

## Multi-Stage Execution Plan

---

## STAGE 1: Foundation & Configuration System
**Goal**: Establish project structure, configuration management, and basic connectivity testing

### 1.1 Project Structure
```
polymarketbot/
├── config/
│   ├── accounts.yaml          # Target accounts & your account config
│   ├── settings.yaml          # Global settings (slippage, polling rates)
│   └── .env.example           # Template for secrets
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py          # YAML + env loading
│   │   ├── models.py          # Pydantic config models
│   │   └── validation.py      # Config validation
│   ├── core/
│   │   ├── __init__.py
│   │   └── exceptions.py      # Custom exceptions
│   └── utils/
│       ├── __init__.py
│       └── logging.py         # Structured logging setup
├── tests/
│   └── ...
├── .env                       # Secrets (gitignored)
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 1.2 Configuration Schema
```yaml
# accounts.yaml
your_account:
  private_key_env: "POLY_PRIVATE_KEY"    # Reference to env var
  proxy_wallet: "0x..."                   # Your Polymarket proxy wallet
  signature_type: 2                       # 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE

targets:
  - name: "whale_trader_1"
    wallet: "0x..."                       # Target's proxy wallet address
    enabled: true
    position_ratio: 0.01                  # 1/100th of their size
    max_position_usd: 500                 # Max per-position exposure
    slippage_tolerance: 0.05              # 5% max slippage
    min_position_usd: 5                   # Ignore positions smaller than $5

  - name: "whale_trader_2"
    wallet: "0x..."
    enabled: true
    position_ratio: 0.02                  # 2% of their size
    max_position_usd: 1000
    slippage_tolerance: 0.03              # 3% max slippage
    min_position_usd: 10

# settings.yaml
polling:
  activity_interval_ms: 200              # How often to poll /activity
  positions_sync_interval_s: 60          # Full position reconciliation

execution:
  order_timeout_s: 30
  max_retries: 3
  retry_delay_ms: 500

safety:
  max_daily_loss_usd: 1000               # Circuit breaker
  max_open_positions: 50
  require_liquidity_check: true
  min_book_depth_usd: 100                # Minimum liquidity to trade

network:
  api_timeout_s: 10
  connection_pool_size: 20
  keep_alive: true
```

### 1.3 Deliverables
- [ ] Project structure created
- [ ] Pydantic models for all configuration
- [ ] Environment variable loading with validation
- [ ] Basic connectivity test to all 4 APIs
- [ ] Structured logging setup
- [ ] Unit tests for config loading

### 1.4 Success Criteria
- Can load and validate configuration
- Can successfully ping all Polymarket endpoints
- Logs are structured JSON with timestamps

---

## STAGE 2: API Client Layer
**Goal**: Build robust, async API clients for all Polymarket endpoints

### 2.1 Components
```
src/
├── api/
│   ├── __init__.py
│   ├── base.py                # Base async HTTP client with retry logic
│   ├── clob.py                # CLOB API client (orders, books)
│   ├── data.py                # Data API client (positions, activity)
│   ├── gamma.py               # Gamma API client (markets metadata)
│   ├── auth.py                # Authentication (L1 EIP-712, L2 HMAC)
│   └── rate_limiter.py        # Token bucket rate limiter
```

### 2.2 Key Features
- **Connection pooling**: Reuse HTTP connections
- **Automatic retries**: Exponential backoff on 429/5xx
- **Rate limiting**: Proactive rate limiting per endpoint
- **Request signing**: L1 (EIP-712) and L2 (HMAC-SHA256) authentication
- **Metrics**: Track latency, success/failure rates

### 2.3 Authentication Implementation
```python
# L2 Auth (HMAC) - used for most operations
class ClobAuth:
    def __init__(self, api_key: str, secret: str, passphrase: str):
        self.api_key = api_key
        self.secret = base64.b64decode(secret)
        self.passphrase = passphrase

    def sign_request(self, method: str, path: str, body: str = "") -> dict:
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method + path + body
        signature = hmac.new(
            self.secret,
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "POLY_ADDRESS": self.address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_API_KEY": self.api_key,
            "POLY_PASSPHRASE": self.passphrase
        }
```

### 2.4 Deliverables
- [ ] Base async HTTP client with retry/timeout
- [ ] Rate limiter (token bucket per endpoint)
- [ ] Data API client (positions, activity, trades)
- [ ] CLOB API client (orders, order book)
- [ ] Gamma API client (markets, events)
- [ ] Full authentication flow (derive creds, sign requests)
- [ ] Integration tests against live APIs

### 2.5 Success Criteria
- Can fetch positions for any wallet
- Can fetch order book for any market
- Rate limiting prevents 429 errors
- Authentication works for order operations

---

## STAGE 3: Position Monitoring Engine
**Goal**: Real-time detection of target wallet position changes

### 3.1 Architecture
```
src/
├── monitoring/
│   ├── __init__.py
│   ├── activity_monitor.py    # Polls /activity for new trades
│   ├── position_tracker.py    # Maintains local position state
│   ├── change_detector.py     # Detects position deltas
│   └── event_bus.py           # Publishes position change events
```

### 3.2 Detection Strategy

**Primary Method: Activity Polling**
```python
async def monitor_activity(target_wallet: str):
    last_timestamp = get_last_seen_timestamp(target_wallet)

    while True:
        activities = await data_api.get_activity(
            user=target_wallet,
            type="TRADE",
            start=last_timestamp,
            sortBy="TIMESTAMP",
            sortDirection="ASC"
        )

        for activity in activities:
            if activity.timestamp > last_timestamp:
                await event_bus.publish(TradeDetected(
                    wallet=target_wallet,
                    market=activity.condition_id,
                    side=activity.side,
                    size=activity.size,
                    price=activity.price,
                    timestamp=activity.timestamp
                ))
                last_timestamp = activity.timestamp

        await asyncio.sleep(0.2)  # 200ms polling
```

**Secondary Method: Position Reconciliation**
- Every 60 seconds, fetch full positions
- Compare with local state
- Detect any missed trades
- Log discrepancies

### 3.3 State Management
```python
@dataclass
class TrackedPosition:
    target_wallet: str
    market_id: str
    token_id: str
    outcome: str
    target_size: Decimal
    our_size: Decimal
    average_price: Decimal
    last_updated: datetime
    status: PositionStatus  # SYNCED, PENDING, DRIFT
```

### 3.4 Deliverables
- [ ] Activity polling loop with configurable interval
- [ ] Local position state tracking
- [ ] Position change detection (new, increased, decreased, closed)
- [ ] Event bus for publishing detected changes
- [ ] Position reconciliation (full sync)
- [ ] Metrics: detection latency, missed trades

### 3.5 Success Criteria
- Detects new trades within 500ms of occurrence
- Correctly identifies position increases/decreases
- No duplicate detections
- Handles target trading multiple markets simultaneously

---

## STAGE 4: Order Execution Engine
**Goal**: Execute copy-trades with slippage protection

### 4.1 Architecture
```
src/
├── execution/
│   ├── __init__.py
│   ├── order_builder.py       # Build & sign orders
│   ├── slippage.py            # Slippage calculation & protection
│   ├── executor.py            # Order submission & tracking
│   ├── position_sizer.py      # Calculate proportional sizes
│   └── queue.py               # Order execution queue
```

### 4.2 Slippage Protection Logic
```python
@dataclass
class SlippageCheck:
    target_price: Decimal       # Price target got
    current_price: Decimal      # Current market price
    slippage_percent: Decimal   # Calculated slippage
    max_allowed: Decimal        # Config threshold
    passed: bool                # Whether to proceed

async def check_slippage(
    market_id: str,
    side: str,
    target_price: Decimal,
    max_slippage: Decimal
) -> SlippageCheck:
    # Get current best price from order book
    book = await clob_api.get_order_book(market_id)

    if side == "BUY":
        current_price = book.best_ask
    else:
        current_price = book.best_bid

    slippage = abs(current_price - target_price) / target_price

    return SlippageCheck(
        target_price=target_price,
        current_price=current_price,
        slippage_percent=slippage,
        max_allowed=max_slippage,
        passed=slippage <= max_slippage
    )
```

### 4.3 Position Sizing
```python
def calculate_copy_size(
    target_size: Decimal,
    position_ratio: Decimal,
    max_position_usd: Decimal,
    current_price: Decimal,
    available_balance: Decimal
) -> Optional[Decimal]:
    """Calculate the size to copy, respecting all constraints."""

    # Apply ratio
    desired_size = target_size * position_ratio

    # Calculate USD value
    desired_usd = desired_size * current_price

    # Cap at max position
    if desired_usd > max_position_usd:
        desired_size = max_position_usd / current_price

    # Check balance (with buffer for fees)
    cost = desired_size * current_price * Decimal("1.01")  # 1% buffer
    if cost > available_balance:
        desired_size = (available_balance * Decimal("0.99")) / current_price

    # Check minimum viable size
    if desired_size * current_price < MIN_ORDER_USD:
        return None  # Too small to execute

    return desired_size
```

### 4.4 Order Execution Flow
```
1. Trade detected for target wallet
2. Calculate proportional size
3. Fetch current order book
4. Check slippage against threshold
5. IF slippage OK:
   a. Build limit order at current best price
   b. Sign order with EIP-712
   c. Submit to CLOB
   d. Monitor for fill via WebSocket
   e. Update local position state
6. IF slippage EXCEEDED:
   a. Log warning with details
   b. Optionally: place limit order at target's price (wait for fill)
   c. Optionally: skip trade entirely
```

### 4.5 Order Types Strategy
- **Immediate execution**: Use marketable limit orders (cross the spread)
- **Slippage exceeded**: Use passive limit orders at target price
- **Time in force**: GTD with 5-minute expiry for passive orders

### 4.6 Deliverables
- [ ] Order builder with EIP-712 signing
- [ ] Slippage calculator with order book integration
- [ ] Position sizer with all constraints
- [ ] Execution queue (prioritized, async)
- [ ] Order status tracking
- [ ] Fill confirmation via WebSocket
- [ ] Retry logic for failed orders
- [ ] Metrics: execution latency, fill rates, slippage realized

### 4.7 Success Criteria
- Orders execute within 200ms of detection (when slippage OK)
- Slippage protection correctly blocks bad trades
- Position sizing respects all constraints
- Failed orders are retried appropriately

---

## STAGE 5: WebSocket Integration
**Goal**: Real-time order book updates and fill confirmations

### 5.1 Architecture
```
src/
├── websocket/
│   ├── __init__.py
│   ├── client.py              # WebSocket connection manager
│   ├── market_feed.py         # Order book updates
│   ├── user_feed.py           # Order/fill updates (authenticated)
│   └── reconnection.py        # Auto-reconnect with backoff
```

### 5.2 Dual Feed Strategy
1. **Market Feed** (unauthenticated): Real-time order book for active markets
2. **User Feed** (authenticated): Your order status and fill confirmations

### 5.3 Local Order Book Maintenance
```python
class LocalOrderBook:
    def __init__(self, token_id: str):
        self.token_id = token_id
        self.bids: SortedDict[Decimal, Decimal] = SortedDict()  # price -> size
        self.asks: SortedDict[Decimal, Decimal] = SortedDict()
        self.sequence: int = 0

    def apply_update(self, update: BookUpdate):
        if update.sequence <= self.sequence:
            return  # Stale update

        for change in update.changes:
            book = self.bids if change.side == "BUY" else self.asks
            if change.size == 0:
                book.pop(change.price, None)
            else:
                book[change.price] = change.size

        self.sequence = update.sequence

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids.keys()[-1] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks.keys()[0] if self.asks else None
```

### 5.4 Connection Management
- Heartbeat every 5 seconds (PING)
- Auto-reconnect on disconnect with exponential backoff
- Resubscribe to all markets on reconnect
- Detect stale data (sequence gaps)

### 5.5 Deliverables
- [ ] WebSocket client with auto-reconnect
- [ ] Market feed subscription manager
- [ ] User feed with authentication
- [ ] Local order book maintenance
- [ ] Sequence number tracking
- [ ] Heartbeat/keepalive
- [ ] Metrics: connection uptime, message latency

### 5.6 Success Criteria
- Maintains connection for 24+ hours
- Recovers from disconnects within 5 seconds
- Order book matches REST API within tolerance
- Fill confirmations received in <100ms

---

## STAGE 6: Account & Balance Management
**Goal**: Track balances, positions, and P&L per tracked account

### 6.1 Architecture
```
src/
├── accounts/
│   ├── __init__.py
│   ├── balance.py             # USDC balance tracking
│   ├── positions.py           # Position management per account
│   ├── pnl.py                 # P&L calculation
│   └── reconciliation.py      # Sync with on-chain state
```

### 6.2 Per-Account Tracking
```python
@dataclass
class AccountState:
    account_name: str
    target_wallet: str

    # Balance
    usdc_balance: Decimal
    reserved_for_orders: Decimal
    available_balance: Decimal

    # Positions (keyed by market_id)
    positions: Dict[str, CopyPosition]

    # P&L
    total_invested: Decimal
    current_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal

    # Statistics
    trades_executed: int
    trades_skipped_slippage: int
    trades_failed: int

    # Timestamps
    last_sync: datetime
    last_trade: Optional[datetime]
```

### 6.3 Position Drift Detection
When your position doesn't match expected ratio:
```python
def check_drift(target_pos: Decimal, our_pos: Decimal, ratio: Decimal) -> DriftStatus:
    expected = target_pos * ratio
    actual = our_pos
    drift_percent = abs(actual - expected) / expected if expected > 0 else 0

    if drift_percent > 0.1:  # 10% drift
        return DriftStatus.SIGNIFICANT
    elif drift_percent > 0.05:  # 5% drift
        return DriftStatus.MINOR
    return DriftStatus.SYNCED
```

### 6.4 Deliverables
- [ ] Per-account state management
- [ ] Balance tracking with reserved amounts
- [ ] Position tracking with drift detection
- [ ] P&L calculation (realized + unrealized)
- [ ] Periodic reconciliation with chain state
- [ ] Statistics and metrics per account

### 6.5 Success Criteria
- Accurate balance tracking within 1%
- Position drift detected within 60 seconds
- P&L calculation matches manual verification

---

## STAGE 7: Safety Systems & Circuit Breakers
**Goal**: Protect against losses and system failures

### 7.1 Architecture
```
src/
├── safety/
│   ├── __init__.py
│   ├── circuit_breaker.py     # Halt trading on anomalies
│   ├── limits.py              # Position/loss limits
│   ├── health_check.py        # System health monitoring
│   └── alerts.py              # Alert notifications
```

### 7.2 Circuit Breakers
```python
class CircuitBreaker:
    conditions = [
        MaxDailyLoss(threshold_usd=1000),
        MaxConsecutiveFailures(count=5),
        MaxSlippageEvents(count=10, window_minutes=5),
        APIErrorRate(threshold=0.5, window_minutes=1),
        PositionDriftTooHigh(threshold=0.2),
        BalanceTooLow(threshold_usd=50),
    ]

    async def check_all(self) -> Optional[TripReason]:
        for condition in self.conditions:
            if await condition.is_triggered():
                return condition.reason
        return None

    async def trip(self, reason: TripReason):
        # Cancel all open orders
        await order_manager.cancel_all()

        # Disable all monitoring
        await monitor.pause()

        # Send alert
        await alerts.send_critical(f"Circuit breaker tripped: {reason}")

        # Log state snapshot
        await state_manager.snapshot()
```

### 7.3 Health Monitoring
- API connectivity (ping every 10s)
- WebSocket connection status
- Rate limit headroom
- Memory/CPU usage
- Order queue depth

### 7.4 Deliverables
- [ ] Circuit breaker framework
- [ ] All circuit breaker conditions implemented
- [ ] Health check system
- [ ] Alert system (webhook/email)
- [ ] Graceful shutdown handling
- [ ] State snapshot on trip

### 7.5 Success Criteria
- Circuit breaker trips within 1 second of condition
- All open orders cancelled on trip
- Alerts delivered within 30 seconds

---

## STAGE 8: Database & Persistence
**Goal**: Persist state, enable recovery, maintain audit trail

### 8.1 Schema
```sql
-- Tracked accounts configuration
CREATE TABLE tracked_accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    target_wallet VARCHAR(42) NOT NULL,
    position_ratio DECIMAL(10,6) NOT NULL,
    max_position_usd DECIMAL(20,2) NOT NULL,
    slippage_tolerance DECIMAL(5,4) NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Position tracking
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES tracked_accounts(id),
    market_id VARCHAR(66) NOT NULL,
    token_id VARCHAR(66) NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    target_size DECIMAL(30,18) NOT NULL,
    our_size DECIMAL(30,18) NOT NULL,
    average_price DECIMAL(20,18) NOT NULL,
    status VARCHAR(20) NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(account_id, token_id)
);

-- Trade execution log (audit trail)
CREATE TABLE trade_log (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES tracked_accounts(id),
    market_id VARCHAR(66) NOT NULL,
    side VARCHAR(4) NOT NULL,
    target_price DECIMAL(20,18) NOT NULL,
    execution_price DECIMAL(20,18),
    target_size DECIMAL(30,18) NOT NULL,
    execution_size DECIMAL(30,18),
    slippage_percent DECIMAL(10,6),
    status VARCHAR(20) NOT NULL,
    order_id VARCHAR(100),
    tx_hash VARCHAR(66),
    error_message TEXT,
    detected_at TIMESTAMP NOT NULL,
    executed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- System events log
CREATE TABLE system_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for query performance
CREATE INDEX idx_positions_account_status ON positions(account_id, status);
CREATE INDEX idx_trade_log_account_created ON trade_log(account_id, created_at);
CREATE INDEX idx_trade_log_status ON trade_log(status);
CREATE INDEX idx_system_events_type_created ON system_events(event_type, created_at);
```

### 8.2 Recovery on Restart
1. Load last known positions from database
2. Fetch current positions from Data API
3. Reconcile and detect drift
4. Resume monitoring from last known timestamp

### 8.3 Deliverables
- [ ] Database schema migration system
- [ ] Repository pattern for all entities
- [ ] Trade logging with full context
- [ ] Position persistence
- [ ] Recovery on restart
- [ ] Audit trail queries

### 8.4 Success Criteria
- No data loss on restart
- Full audit trail of all decisions
- Can reconstruct state from logs

---

## STAGE 9: CLI Interface & Operations
**Goal**: Command-line interface for operations and monitoring

### 9.1 Commands
```
polybot start                  # Start the bot
polybot stop                   # Graceful shutdown
polybot status                 # Show current state
polybot accounts               # List tracked accounts
polybot accounts add           # Add new account to track
polybot positions              # Show all positions
polybot positions <account>    # Show positions for account
polybot trades                 # Recent trade log
polybot pnl                    # P&L summary
polybot config validate        # Validate configuration
polybot test-connection        # Test API connectivity
polybot reconcile              # Force position reconciliation
polybot pause <account>        # Pause tracking for account
polybot resume <account>       # Resume tracking
```

### 9.2 Deliverables
- [ ] CLI framework (Click or Typer)
- [ ] All operational commands
- [ ] Status display with tables
- [ ] Interactive mode for sensitive input
- [ ] Systemd service file for 24/7 operation

### 9.3 Success Criteria
- All operations accessible via CLI
- Clear, formatted output
- Non-zero exit codes on errors

---

## STAGE 10: Web Dashboard (Future)
**Goal**: Real-time web UI for monitoring and control

### 10.1 Features
- Real-time position display
- P&L charts
- Trade history
- Account management
- Alert configuration
- System health dashboard

### 10.2 Technology
- FastAPI backend
- WebSocket for real-time updates
- React or Vue.js frontend
- TailwindCSS styling

### 10.3 Deliverables
- [ ] FastAPI REST + WebSocket API
- [ ] Frontend dashboard
- [ ] Authentication
- [ ] Real-time updates

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Proxmox VM (Dedicated)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    VPN Container                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  WireGuard/OpenVPN → Exit in Amsterdam/Netherlands  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Bot Container (Python)                    │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Activity Monitor → Execution Engine → WebSocket    │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  PostgreSQL (positions, trades, events)              │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Monitoring Stack                          │  │
│  │  Prometheus → Grafana → AlertManager                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Targets

| Metric | Target | Method to Achieve |
|--------|--------|-------------------|
| Trade detection latency | <300ms | 200ms polling + optimized parsing |
| Order execution latency | <200ms | Connection pooling, pre-computed data |
| End-to-end copy latency | <500ms | Async pipeline, no blocking |
| WebSocket uptime | >99.9% | Auto-reconnect, health checks |
| Order fill rate | >95% | Proper slippage settings, liquidity checks |
| System uptime | >99.5% | Health monitoring, auto-restart |

---

## Risk Mitigation Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Target makes large trade, we can't match | Medium | Medium | Position size limits, balance checks |
| Slippage exceeds threshold repeatedly | Medium | Low | Configurable thresholds, logging |
| API rate limiting | Low | Medium | Proactive rate limiting, backoff |
| WebSocket disconnection | Medium | Low | Auto-reconnect, REST fallback |
| VPN connection drops | Medium | High | VPN health check, pause trading |
| Database corruption | Low | High | Backups, WAL replication |
| Private key compromise | Low | Critical | HSM/secure enclave consideration |
| Polymarket API changes | Low | High | Version checking, abstraction layer |

---

## Next Steps

Upon your approval of this plan, I will proceed with **Stage 1: Foundation & Configuration System**.

This includes:
1. Creating the full project structure
2. Implementing Pydantic configuration models
3. Setting up structured logging
4. Building basic API connectivity tests
5. Creating example configuration files

Please review this plan and let me know:
1. Any modifications to the staging order
2. Additional requirements I may have missed
3. Questions about any specific component
4. Approval to proceed with Stage 1
