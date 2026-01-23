# PolymarketBot - Copy Trading System

A real-time copy trading bot for [Polymarket](https://polymarket.com) prediction markets. Monitor successful traders and automatically mirror their positions with configurable risk management.

## Features

- **Real-time Trade Detection** - Monitors tracked accounts via Polymarket API polling (~3 second intervals)
- **Ghost Mode** - Test your strategy with simulated execution before going live
- **Tiered Slippage Control** - Dynamic slippage tolerance based on entry price (low odds = more slippage allowed)
- **Keyword Filtering** - Only copy trades on markets matching your keywords
- **Drawdown Protection** - Automatic stoploss when tracked account exceeds drawdown threshold
- **Advanced Risk Settings** - Take profit, stop loss, max concurrent positions, holding time limits
- **Web Dashboard** - Real-time monitoring UI with WebSocket updates
- **Cross-Platform** - Works on Linux and Windows
- **Auto Wallet Lookup** - Enter a username and automatically find their wallet address

## System Architecture

```
polymarketbot/
├── src/
│   ├── copytrade/              # Core copy trading infrastructure
│   │   ├── account.py          # CopyTradeAccount dataclass, slippage tiers
│   │   └── manager.py          # Account CRUD and state persistence
│   ├── utils/
│   │   ├── polymarket_api.py   # Polymarket API utilities (wallet lookup, etc.)
│   │   └── logging.py          # Logging configuration
│   └── web/
│       ├── static/js/app.js    # Dashboard frontend
│       └── templates/          # HTML templates
│
├── ghost_mode.py               # Ghost mode simulation engine
├── run_ghost_mode.py           # Main entry point with web server
└── ghost_state.json            # Persistent state (auto-generated)
```

## Requirements

- Python 3.10+ (3.11 recommended)
- pip (Python package manager)
- Internet connection
- VPN if in geo-restricted region

---

## Installation

### Linux

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/polymarketbot.git
cd polymarketbot

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install fastapi uvicorn websockets requests pydantic

# Or if requirements.txt exists:
pip install -r requirements.txt
```

### Windows

```powershell
# 1. Clone the repository
git clone https://github.com/yourusername/polymarketbot.git
cd polymarketbot

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\activate

# 4. Install dependencies
pip install fastapi uvicorn websockets requests pydantic

# Or if requirements.txt exists:
pip install -r requirements.txt
```

### Dependencies

Core packages needed:
```
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
requests>=2.31.0
pydantic>=2.0.0
```

---

## Quick Start

### Step 1: Start the Dashboard

**Linux:**
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start the bot
python3 run_ghost_mode.py
```

**Windows:**
```powershell
# Activate virtual environment (if not already active)
.\venv\Scripts\activate

# Start the bot
python run_ghost_mode.py
```

You should see:
```
======================================================================
  PolymarketBot Ghost Mode Dashboard
======================================================================

  Dashboard URL: http://localhost:8765

  Tracked Accounts:
    - automatedaitradingbot: 0xd8f8c136...
      Keywords: temperature, tempature, weather
      Max Drawdown: 15%

  Mode: GHOST MODE (Real monitoring, simulated execution)
```

### Step 2: Open the Dashboard

Navigate to **http://localhost:8765** in your web browser.

### Step 3: Add Accounts to Track

1. Click **"Add Account"** in the Tracked Accounts section
2. Enter the Polymarket **username** - the wallet address will auto-lookup
3. Or paste the **wallet address** directly (0x...)
4. Configure your copy settings (see below)
5. Click **Save**

### Step 4: Enable Ghost Mode

Click the **Ghost Mode toggle** in the top-right corner to start monitoring.

### Step 5: Monitor Trades

Watch as the dashboard shows:
- Detected trades from tracked accounts
- Which trades would be copied (pass filters)
- Which trades are filtered out (keyword, slippage, drawdown)
- Real-time latency statistics

---

## Configuration Guide

### Account Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **Username/Name** | Display name (also used for wallet lookup) | Required |
| **Wallet Address** | Polymarket wallet to track (0x...) | Required |
| **Position Ratio** | Copy this % of their trade size | 0.01 (1%) |
| **Max Position USD** | Maximum dollars per copied trade | $500 |
| **Keywords** | Only copy markets containing these terms | Empty (all) |
| **Max Drawdown %** | Stop copying if account drops this much | 15% |

### Tiered Slippage System

The tiered slippage system allows more slippage on low-probability bets (which have more edge) and tighter slippage on high-probability bets:

| Entry Price | Max Slippage | Rationale |
|-------------|--------------|-----------|
| 0-5¢ | 300% | Huge edge on long shots, price movement expected |
| 5-10¢ | 200% | Still significant edge at low prices |
| 10-20¢ | 100% | Good edge, moderate slippage tolerance |
| 20-35¢ | 50% | Decent edge, need tighter execution |
| 35-50¢ | 30% | Moderate odds, moderate slippage |
| 50-70¢ | 20% | Getting expensive, tighter tolerance |
| 70-85¢ | 12% | Thin margins, need good fills |
| 85-100¢ | 6% | Near-certainties, very tight slippage |

**Example:** If you're copying a BUY at 5¢, the system allows the fill price to be up to 20¢ (300% slippage) because the potential edge on a 5¢ bet is massive. But copying a BUY at 90¢ only allows fills up to 95.4¢ (6% slippage) because the margin is thin.

### Advanced Risk Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **Take Profit %** | Auto-close position at this profit | 0 (disabled) |
| **Stop Loss %** | Auto-close position at this loss | 0 (disabled) |
| **Max Concurrent** | Maximum open positions at once | 0 (unlimited) |
| **Max Holding Hours** | Auto-close after N hours | 0 (disabled) |
| **Min Liquidity** | Skip markets below this liquidity | 0 (no minimum) |
| **Cooldown Seconds** | Wait between trade attempts | 10 |

---

## Ghost Mode vs Live Mode

### Ghost Mode (Default - Safe for Testing)

- ✅ Monitors **real trades** from tracked accounts
- ✅ Simulates what **would happen** if you copied them
- ✅ API calls are made but **safely fail** (no wallet connected)
- ✅ Measures **timing and latency** accurately
- ✅ **No money at risk** - perfect for testing

### Live Mode (Requires Wallet Setup)

- ⚠️ **Actually executes trades** with real money
- ⚠️ Requires Polymarket wallet connection and API credentials
- ⚠️ Real money at risk
- ⚠️ Test thoroughly in Ghost Mode first!

---

## Custom Port Configuration

The default port is **8765**. To use a different port:

```bash
# Via command line argument
python3 run_ghost_mode.py --port 9000

# Via environment variable (Linux)
PORT=9000 python3 run_ghost_mode.py

# Via environment variable (Windows PowerShell)
$env:PORT=9000; python run_ghost_mode.py
```

---

## Understanding the Dashboard

### Status Panel
- **Connection**: WebSocket status (green = connected)
- **Mode**: Ghost/Live indicator
- **Uptime**: How long the monitor has been running

### Statistics Panel
- **Trades Detected**: Total trades seen from tracked accounts
- **Would Execute**: Trades that passed all filters
- **Filtered (Keyword)**: Trades skipped - no keyword match
- **Filtered (Stoploss)**: Trades skipped - drawdown exceeded
- **Filtered (Slippage)**: Trades skipped - too much slippage
- **Missed Offline**: Trades that occurred while system was down

### Trades Table
Real-time feed showing:
- Timestamp
- Account name
- Market name
- Side (BUY/SELL)
- Size and price
- Slippage percentage
- Status (ghost, filtered, executed)

### Latency Panel
- **Detection**: Time to poll and detect a trade
- **End-to-End**: Total processing time
- Critical for copy trading - lower is better!

---

## Account Discovery Scanner

The Discovery Scanner helps you find under-the-radar profitable accounts worth copy-trading. Rather than copying well-known leaderboard accounts that everyone else is watching, it finds smaller consistent performers in niche categories.

### Accessing the Scanner

Navigate to **http://localhost:8765/discovery** or click "Discovery" in the dashboard navigation.

### Scan Types

| Scan Type | Description | Best For |
|-----------|-------------|----------|
| **Niche Market Holders** | Finds traders holding positions in weather, economics, tech markets | Finding under-the-radar specialists |
| **Profit Threshold** | Collects anyone with >$1k profit, then filters | Casting a wide net |
| **Category Leaderboards** | Pulls from category-specific leaderboards | Finding proven performers |

### Three-Phase Filtering Pipeline

The scanner uses efficient multi-phase filtering to minimize API calls:

```
PHASE 1: SEED COLLECTION (0 extra API calls)
├── Collect candidates from chosen source
├── Quick filter using existing data
└── Keep top N for Phase 2

PHASE 2: LIGHT SCAN (1 API call per candidate)
├── Fetch recent activity
├── Check if account is active
├── Basic pattern analysis
└── Keep top N for Phase 3

PHASE 3: DEEP ANALYSIS (2-3 API calls per candidate)
├── Full trade history
├── P/L curve analysis
├── Pattern metrics
├── Red flag detection
└── Comprehensive scoring
```

### Analysis Modes

| Mode | Description | Hard Filters |
|------|-------------|--------------|
| **Niche Specialist** | Traders focused on weather/economics/tech | >50 trades, profitable, <40% mainstream |
| **Micro-Bet Hunter** | Many small bets at extreme long odds | >100 trades, >30% under 10¢, avg <$100 |
| **Insider Detection** | Suspicious patterns (fresh accounts, huge bets) | Inverted scoring - flags suspicious behavior |
| **Similar To** | Find accounts trading like a reference wallet | Requires reference wallet input |

### Key Metrics Analyzed

**P/L Metrics:**
- Total realized P/L
- Sharpe/Sortino/Calmar ratios
- Max drawdown percentage
- Win rate and profit factor
- Largest win concentration

**Trading Patterns:**
- Average position size
- % trades at <10¢ odds (long shots)
- Trades per day frequency
- Account age
- Category specialization

**Red Flags:**
- Fresh account with large positions
- Single win dominates P/L
- >85% win rate (suspicious)
- Heavy concentration in one bet

### Configuring Thresholds

Click "Advanced Filters" to customize scoring thresholds:

- **Hard Filters**: Must pass to be considered (e.g., min 50 trades)
- **Soft Filters**: Contribute to score (e.g., +15 points for >60% win rate)
- **Weights**: Adjust importance of different factors

### Research-Backed Thresholds

Based on analysis of Polymarket copy-trading strategies:

| Metric | Good Value | Excellent Value |
|--------|------------|-----------------|
| Total P/L | >$5,000 | >$50,000 |
| Win Rate | >55% | >60% |
| Trade Count | >100 | >500 |
| Account Age | >90 days | >120 days |
| Mainstream % | <40% | <20% |

### Tips for Finding Quality Accounts

1. **Avoid the obvious** - Top leaderboard accounts are already heavily copied
2. **Look for consistency** - Smooth P/L curves beat lucky one-time wins
3. **Category expertise** - Specialists outperform generalists
4. **Reasonable size** - $20-200 avg positions are more copyable than $10k whales
5. **Active but not hyperactive** - ~100 trades/month shows real research

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/discovery/scan/start` | POST | Start a new discovery scan |
| `/api/discovery/scan/{id}` | GET | Get scan status and progress |
| `/api/discovery/scan/{id}/results` | GET | Get scan results |
| `/api/discovery/accounts` | GET | List all discovered accounts |
| `/api/discovery/accounts/{wallet}` | GET | Get detailed analysis |
| `/api/discovery/config/{mode}` | GET | Get scoring configuration |
| `/api/discovery/config/{mode}` | PUT | Update scoring thresholds |

### Sources

Research on copy-trading strategies:
- [Copytrade Wars - Polymarket Oracle](https://news.polymarket.com/p/copytrade-wars)
- [Wallet Basket Strategy - Phemex](https://phemex.com/news/article/innovative-strategy-emerges-for-polymarket-copy-trading-50622)
- [10 Rules for Choosing Wallets - Medium](https://medium.com/@michalstefanow.marek/how-to-%D1%81hoose-wallet-for-copy-trading-on-polymarket-10-main-rules-ct-will-never-share-with-you-0351ad82faac)
- [PolyTrack Copy Trading Guide](https://www.polytrackhq.app/blog/polymarket-copy-trading-guide)

---

## API Reference

The dashboard exposes a REST API for programmatic access:

### Status & Health
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current bot status and statistics |
| `/api/health` | GET | Health check endpoint |
| `/api/mode/status` | GET | Current mode (ghost/live) |

### Account Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/accounts` | GET | List all tracked accounts |
| `/api/accounts` | POST | Add new tracked account |
| `/api/accounts/{id}` | PUT | Update account settings |
| `/api/accounts/{id}` | DELETE | Remove tracked account |
| `/api/lookup-wallet/{username}` | GET | Lookup wallet from username |

### Mode Control
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ghost-mode/start` | POST | Start ghost mode monitoring |
| `/api/ghost-mode/stop` | POST | Stop ghost mode |
| `/api/live-mode/start` | POST | Start live mode (requires wallet) |
| `/api/live-mode/stop` | POST | Stop live mode |

### Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trades` | GET | Recent trades (add `?limit=N`) |
| `/api/positions` | GET | Current open positions |
| `/api/latency` | GET | Latency statistics |
| `/api/missed-trades` | GET | Trades missed while offline |

### Infrastructure & Proxy
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/infrastructure/info` | GET | Get infrastructure info and VPS recommendations |
| `/api/infrastructure/geo-check` | GET | Check if current IP is geo-restricted |
| `/api/infrastructure/benchmark` | POST | Run latency benchmark against Polymarket |
| `/api/infrastructure/proxy` | GET | Get current proxy configuration |
| `/api/infrastructure/proxy` | PUT | Update proxy configuration |
| `/api/infrastructure/proxy/test` | POST | Test proxy connection |

### Pages
| URL | Description |
|-----|-------------|
| `/` | Main dashboard |
| `/discovery` | Account discovery scanner |
| `/infrastructure` | Infrastructure & proxy settings |

### WebSocket
Connect to `ws://localhost:8765/ws/live` for real-time updates.

---

## Troubleshooting

### "Port already in use"

**Linux/Mac:**
```bash
# Find process using the port
lsof -i :8765

# Kill it
kill <PID>
```

**Windows:**
```powershell
# Find process using the port
netstat -ano | findstr :8765

# Kill it (replace PID with actual number)
taskkill /PID <PID> /F
```

### "Module not found" errors

Ensure virtual environment is activated:
```bash
# Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate

# Then verify
which python  # Linux - should show venv path
where python  # Windows - should show venv path
```

### Wallet lookup fails

The username lookup queries Polymarket's public API. If it fails:
1. Verify the username is spelled correctly (case-sensitive)
2. The user may have a different display name than username
3. Enter the wallet address manually from their Polymarket profile URL

### No trades detected

1. ✅ Is Ghost Mode enabled? (toggle should be ON)
2. ✅ Does the tracked account have recent activity?
3. ✅ Are your keywords too restrictive?
4. ✅ Check browser console (F12) for WebSocket errors
5. ✅ Check terminal output for API errors

### WebSocket keeps disconnecting

1. Check your internet connection
2. The server may have crashed - check terminal for errors
3. Try refreshing the browser page

---

## Files & Data

### ghost_state.json

Auto-generated file storing:
- Tracked account configurations
- Seen trade hashes (prevents duplicate processing)
- Last shutdown timestamp

This file persists between restarts. Delete it to reset all state.

### Logs

Console output includes:
- `[State]` - Persistence operations
- `[GHOST]` - Simulated trade executions
- `[FILTERED]` - Trades filtered out
- `[MISSED]` - Trades that occurred while offline
- `[Error]` - Error messages

---

## Infrastructure & Speed Optimization

**Speed is critical for copy trading.** Every millisecond of latency means worse fill prices and potential missed trades. This section covers optimal infrastructure setup.

### Polymarket Server Locations

Polymarket's trading infrastructure runs on AWS:

| Component | Location | AWS Region | Endpoint |
|-----------|----------|------------|----------|
| **CLOB API** (Primary) | London, UK | eu-west-2 | `clob.polymarket.com` |
| **Backup** | Dublin, Ireland | eu-west-1 | - |
| **WebSocket** | London, UK | eu-west-2 | `ws-subscriptions-clob.polymarket.com` |
| **Data API** | London, UK | eu-west-2 | `data-api.polymarket.com` |

### Optimal VPS Locations

Choose a VPS location that's close to London but NOT geo-blocked:

| Location | Latency | Geo-Blocked | Recommended |
|----------|---------|-------------|-------------|
| **Amsterdam, Netherlands** | 2-5ms | No | **Best choice** |
| **Frankfurt, Germany** | 8-12ms | No | Good alternative |
| **Dublin, Ireland** | 5-8ms | No | Close to backup |
| Paris, France | 10-15ms | No | Acceptable |
| London, UK | 1-2ms | **YES** | Do NOT use |
| United States | 80-120ms | **YES** | Do NOT use |

### Recommended VPS Providers

**Amsterdam (Recommended):**
| Provider | Price | Notes |
|----------|-------|-------|
| [QuantVPS](https://quantvps.com) | $29-59/mo | Trading-optimized, lowest latency |
| [Hetzner](https://hetzner.com) | €4-20/mo | Best value in EU |
| [Vultr](https://vultr.com) | $5-20/mo | Easy setup, good network |
| [DigitalOcean](https://digitalocean.com) | $6-24/mo | Reliable, good docs |

**Frankfurt:**
| Provider | Price | Notes |
|----------|-------|-------|
| [Hetzner](https://hetzner.com) | €4-20/mo | Excellent value |
| [AWS](https://aws.amazon.com) | $10-50/mo | eu-central-1 region |
| [Contabo](https://contabo.com) | €5-15/mo | Budget option |

### Speed Optimization Tips

1. **Network:**
   - Deploy VPS in **Amsterdam** for 2-5ms latency to Polymarket
   - Use **WebSocket** instead of HTTP polling for real-time data
   - Enable **HTTP/2** and **keep-alive** connections
   - Use **connection pooling** (10+ persistent connections)

2. **Code:**
   - Use **async/await** for non-blocking I/O
   - Cache DNS lookups and API responses where safe
   - Pre-sign orders to reduce execution time
   - Profile hot paths and optimize bottlenecks

3. **Infrastructure:**
   - Use **SSD/NVMe** storage for fast I/O
   - Consider **dedicated/bare-metal** for lowest jitter
   - Apply TCP kernel optimizations (done automatically by deploy script)

### Proxy Configuration

If you're in a geo-restricted region, use the built-in proxy support:

1. Navigate to **http://localhost:8765/infrastructure**
2. Enable proxy and configure SOCKS5/HTTP proxy settings
3. Test connection to verify it works
4. Run latency benchmark to measure performance

Supported proxy types: HTTP, HTTPS, SOCKS5, SOCKS5H (DNS on proxy)

---

## One-Command Deployment

Deploy to a fresh VPS with a single command:

### Linux/Ubuntu (Recommended)

```bash
# Option 1: Direct script execution
curl -fsSL https://raw.githubusercontent.com/yourusername/polymarketbot/main/deploy.sh | bash

# Option 2: Clone and run
git clone https://github.com/yourusername/polymarketbot.git
cd polymarketbot
chmod +x deploy.sh
./deploy.sh
```

### Docker (Easiest)

```bash
# Single container
docker run -d --name polymarketbot -p 8765:8765 \
  -v polybot-data:/app/data polymarketbot:latest

# With Docker Compose (recommended)
curl -fsSL https://raw.githubusercontent.com/yourusername/polymarketbot/main/docker-compose.yml -o docker-compose.yml
docker compose up -d
```

### Windows

```powershell
# Clone repository
git clone https://github.com/yourusername/polymarketbot.git
cd polymarketbot

# Run deployment script
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\deploy.ps1
```

### Post-Deployment

After deployment:

1. Open **http://your-server-ip:8765** in your browser
2. Go to **Infrastructure** page to verify geo-access and latency
3. Configure proxy if needed
4. Add accounts to track
5. Enable Ghost Mode to start monitoring

---

## Security Notes

- ⚠️ **Never commit private keys** to version control
- ⚠️ Ghost Mode is safe - no wallet connection needed
- ⚠️ Live Mode requires secure credential handling
- ✅ `ghost_state.json` contains only configurations, no secrets
- ✅ API calls in Ghost Mode fail safely (no authentication)

---

## Development

### Running in Background (Linux)

```bash
# Start in background
nohup python3 run_ghost_mode.py > bot.log 2>&1 &

# Check if running
ps aux | grep run_ghost_mode

# View logs
tail -f bot.log
```

### Running as Service (Linux systemd)

Create `/etc/systemd/system/polymarketbot.service`:
```ini
[Unit]
Description=PolymarketBot Copy Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/polymarketbot
Environment=PORT=8765
ExecStart=/root/polymarketbot/venv/bin/python run_ghost_mode.py
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

Then:
```bash
systemctl daemon-reload
systemctl enable polymarketbot
systemctl start polymarketbot

# Check status
systemctl status polymarketbot

# View logs
journalctl -u polymarketbot -f
```

**Important Security Note:** The dashboard binds to `localhost:8765` only. Access it securely via SSH tunnel:
```bash
# From your local machine:
ssh -L 8765:localhost:8765 root@your-vps-ip

# Then open http://localhost:8765 in your browser
```
This ensures the dashboard is never exposed to the public internet.

---

## License

MIT License - See LICENSE file for details.

---

## Disclaimer

**This software is for educational and research purposes only.**

- Trading on prediction markets involves significant risk of loss
- Past performance of tracked accounts does not guarantee future results
- Always test thoroughly in Ghost Mode before risking real funds
- The authors are not responsible for any financial losses
- Ensure compliance with local laws and Polymarket terms of service
