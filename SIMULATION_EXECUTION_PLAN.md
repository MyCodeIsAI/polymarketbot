# PolymarketBot Simulation Execution Plan

## Objective

Simulate every aspect of the copy-trading utility as close to live production as possible,
measuring latency at each stage to validate performance requirements.

---

## Target Performance Requirements

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Trade Detection Latency | < 100ms | < 300ms |
| Order Submission Latency | < 50ms | < 150ms |
| End-to-End Latency | < 200ms | < 500ms |
| Slippage | < 0.5% | < 2% |
| Success Rate | > 98% | > 95% |

---

## Simulation Phases

### Phase 1: Component Latency Benchmarking

Test each component in isolation to establish baseline latencies.

#### 1.1 WebSocket Connection Latency
```
Measure: Time to establish WebSocket connection to Polymarket
Target: < 500ms
```

#### 1.2 Trade Detection Latency
```
Measure: Time from trade appearing on chain to detection
Components:
  - WebSocket message receive: ~10-50ms
  - Message parsing: ~1-5ms
  - Trade validation: ~1-5ms
Target: < 100ms total
```

#### 1.3 Order Building Latency
```
Measure: Time to construct and sign order
Components:
  - Position sizing calculation: ~1ms
  - Order construction: ~5ms
  - EIP-712 signing: ~10-20ms
Target: < 50ms total
```

#### 1.4 Order Submission Latency
```
Measure: Time from order built to confirmation
Components:
  - HTTP POST to CLOB API: ~30-100ms
  - Order matching: ~10-50ms
  - Confirmation receive: ~10-30ms
Target: < 150ms total
```

### Phase 2: Integration Simulation

Test the full pipeline with synthetic trades.

#### 2.1 Synthetic Trade Generation
- Generate trades matching target wallet patterns
- Vary trade sizes: $50 - $5,000
- Mix of markets: political, sports, crypto, news
- Realistic timing: bursts and quiet periods

#### 2.2 Full Pipeline Test
```
Flow:
  1. Synthetic trade generated (simulates target wallet trade)
  2. Detection system picks up trade
  3. Position sizing calculated
  4. Order built and signed
  5. Order submitted (dry-run: logged, not sent)
  6. Latency measured at each stage
```

### Phase 3: Historical Replay

Replay actual historical trades from target wallet.

#### 3.1 Data Collection
- Fetch last 7-30 days of trades from: 0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11
- Include: timestamp, market, outcome, side, size, price
- Total expected: 500-2000 trades

#### 3.2 Replay Modes
| Mode | Speed | Purpose |
|------|-------|---------|
| Real-time (1x) | Actual timing | Validate sustained operation |
| Accelerated (10x) | 10x faster | Stress test detection |
| Burst (100x) | Maximum speed | Find breaking points |

### Phase 4: Live Dry-Run

Connect to real Polymarket systems but don't execute.

#### 4.1 Setup
- Connect to live Polymarket WebSocket
- Monitor target wallet in real-time
- Build orders but log instead of submit

#### 4.2 Metrics Collection
- Detection latency vs block timestamp
- Order build time
- Simulated fill price vs actual market
- Would-be slippage calculation

---

## Measurement Points

```
Timeline of a Copy Trade:

t0: Target trade appears on blockchain
    |
    | [DETECTION_LATENCY: target < 100ms]
    v
t1: Trade detected by our system
    |
    | [PROCESSING_LATENCY: target < 10ms]
    v
t2: Trade validated and position sized
    |
    | [BUILD_LATENCY: target < 50ms]
    v
t3: Order built and signed
    |
    | [SUBMISSION_LATENCY: target < 100ms]
    v
t4: Order submitted to CLOB
    |
    | [MATCHING_LATENCY: varies by liquidity]
    v
t5: Order filled (full or partial)

TOTAL E2E: t5 - t0, target < 300ms
```

---

## Simulation Commands

### Quick Synthetic Test (60 seconds)
```bash
python3 run_with_dashboard.py
# Access: http://localhost:8000
```

### CLI Simulation Commands
```bash
# Synthetic simulation with metrics
polybot simulate run --mode synthetic --duration 300 --rate 30 --output results.json

# Historical replay at 10x speed
polybot simulate run --mode historical -w 0xd8f8...0f11 --speed 10 --duration 600

# Benchmark detection latency (requires live connection)
polybot simulate benchmark 0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11 --duration 300

# Analyze target trader profile
polybot simulate profile 0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11 --days 30
```

### Full Integration Test
```bash
# Start dashboard
python3 run_with_dashboard.py &

# Run benchmark script
python3 scripts/benchmark_latency.py --target 0xd8f8...0f11 --duration 300
```

---

## Expected Results

### Synthetic Simulation (5 min @ 30 trades/hr)
```
Expected:
  - Trades Generated: ~2-3
  - Detection Latency: 0ms (synthetic, no network)
  - Execution Latency: 50-150ms (simulated)
  - Success Rate: 97%+
  - Slippage: 0.5% avg
```

### Historical Replay (10 min @ 10x speed)
```
Expected:
  - Trades Replayed: 50-200 (depends on history)
  - Replay Accuracy: Timestamps preserved
  - Detection Latency: 0ms (replay, no network)
  - Simulated Slippage: Based on historical liquidity
```

### Live Dry-Run (30 min)
```
Expected:
  - Real trades detected: depends on target activity
  - Detection Latency: 50-200ms (real WebSocket)
  - Would-be Slippage: calculated vs current book
  - No actual orders placed
```

---

## Latency Breakdown Targets

| Stage | Component | Target | Measurement Method |
|-------|-----------|--------|-------------------|
| T0→T1 | WebSocket receive | < 50ms | timestamp diff |
| T1→T2 | Parse + validate | < 5ms | code timing |
| T2→T3 | Position sizing | < 5ms | code timing |
| T3→T4 | Order building | < 20ms | code timing |
| T4→T5 | EIP-712 signing | < 20ms | code timing |
| T5→T6 | HTTP submission | < 100ms | request timing |
| T6→T7 | Order matching | variable | response timing |

---

## Risk Scenarios to Test

### 1. Rapid Fire Trades
- Target executes 5+ trades in 10 seconds
- System must queue and execute all

### 2. Large Position
- Target opens $50,000 position
- System must respect max_position_usd limits

### 3. Opposing Trades
- Target buys then immediately sells
- System must handle position reversal

### 4. Market with Low Liquidity
- Small orderbook depth
- System must detect and skip or reduce size

### 5. Network Interruption
- WebSocket disconnect mid-trade
- System must reconnect and reconcile

### 6. API Rate Limiting
- Exceed Polymarket rate limits
- System must backoff gracefully

---

## Success Criteria

| Criteria | Threshold | Status |
|----------|-----------|--------|
| E2E latency < 300ms | 95th percentile | ⬜ |
| Detection latency < 100ms | 95th percentile | ⬜ |
| Slippage < 1% | average | ⬜ |
| Success rate > 95% | all trades | ⬜ |
| No missed trades | 100% detection | ⬜ |
| Graceful degradation | on errors | ⬜ |
| Position accuracy | matches target ratio | ⬜ |

---

## Files Created for Simulation

```
polymarketbot/
├── run_with_dashboard.py      # Full dashboard + simulation
├── run_simulation.py          # CLI simulation runner
├── src/simulation/
│   ├── __init__.py
│   ├── harness.py             # Main simulation orchestrator
│   ├── trade_generator.py     # Synthetic trade generation
│   ├── historical_replay.py   # Historical trade replay
│   └── mock_websocket.py      # WebSocket mock/live toggle
└── config/
    └── accounts.example.yaml  # Target wallet configuration
```

---

## Next Steps

1. **Run synthetic simulation** - Validate dashboard and trade flow
2. **Run historical replay** - Test with real trade patterns
3. **Live dry-run** - Measure real detection latency
4. **Stress test** - Find system limits
5. **Production deployment** - With real execution enabled
