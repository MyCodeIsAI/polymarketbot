# Polymarket Insider Trading Scanner: Research Outline

## Executive Summary

Building a real-time insider trading detection system for Polymarket is **highly feasible**. The platform operates entirely on-chain (Polygon), with transparent transactions, and provides robust APIs and WebSocket streams. Multiple competitors (Polysights, Polywhaler, PolyTrack, Unusual Whales) are already doing this commercially.

---

## Part 1: Competitor Analysis

### 1.1 Unusual Whales "Unusual Predictions" Tool

**Source:** https://unusualwhales.com/predictions, https://phemex.com/news/article/unusual-whales-launches-tool-to-track-insider-trading-on-polymarket-54994

**How They Do It:**
- Analyzes **traders' profitability**, **historical records**, and **timing of bets**
- Identifies "smart money movements and whale activity"
- Tracks prediction markets alongside their core options flow product
- Limited public technical documentation, appears proprietary

### 1.2 Polysights "Insider Finder" - The Market Leader

**Source:** https://gizmodo.com/tracking-insider-trading-on-polymarket-is-turning-into-a-business-of-its-own-2000709286, https://app.polysights.xyz/insider-finder

**Technical Methodology:**
- Created by Tre Upshaw (29, former memecoin trader)
- **85% accuracy rate** on flagged trades
- 24,000 users, $2M funding round in progress, $25K Polymarket grant
- Uses **Vertex AI and Gemini** for AI-powered summaries
- 30+ custom metrics

**Detection Criteria:**
| Factor | Threshold |
|--------|-----------|
| Wallet Age | Created recently (days) |
| Market Participation | Only 2-3 specific events |
| Bet Size | Single bets > $10,000 |
| Timing | Positions completed hours to days before event |
| Market Type | Very specific/niche outcomes |

### 1.3 Open Source Implementation - polymarket-insider-tracker

**Source:** https://github.com/pselamy/polymarket-insider-tracker

**Four Core Detection Algorithms:**

1. **Fresh Wallet Detection**
   - Flags wallets with < 5 lifetime transactions making trades > $1,000
   - Traces funding sources to connect wallets

2. **Liquidity Impact Analysis**
   - Flags trades consuming > 2% of visible order book
   - Niche markets receive higher severity weighting

3. **Sniper Cluster Detection (DBSCAN)**
   - Identifies wallets entering markets within minutes of creation
   - Recognizes coordinated behavior patterns

4. **Event Correlation**
   - Cross-references trading with news
   - Detects positions opened 1-4 hours before news breaks

**Configuration Thresholds:**
```
MIN_TRADE_SIZE_USDC = 1000
FRESH_WALLET_MAX_NONCE = 5
LIQUIDITY_IMPACT_THRESHOLD = 0.02  # 2%
```

**Tech Stack:** PostgreSQL 15, Redis 7, SQLAlchemy, WebSocket ingestion

### 1.4 Polywhaler

**Source:** https://www.polywhaler.com/

**Metrics:**
- Whale threshold: **$10,000+** trades
- **Insider Score (0-100)** based on: low probability bets, unusual trade size, suspicious timing, high-risk market context
- Trade Impact Scoring
- Market Sentiment Analysis

---

## Part 2: API & Real-Time Capabilities

### 2.1 API Rate Limits

**Source:** https://docs.polymarket.com/quickstart/introduction/rate-limits

| Endpoint Category | Rate (per 10 seconds) |
|-------------------|----------------------|
| General API | 15,000 |
| GAMMA API | 4,000 |
| CLOB API General | 9,000 |
| Data API General | 1,000 |
| `/trades` | 200 |
| `/positions` | 150 |
| POST `/order` | 3,500 burst / 60 sustained |

### 2.2 WebSocket Real-Time Streams

**Source:** https://www.polytrackhq.app/blog/polymarket-websocket-tutorial

| Service | URL | Purpose |
|---------|-----|---------|
| CLOB WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Orderbook, trades |
| RTDS | `wss://ws-live-data.polymarket.com` | Market activity, prices |

**Channels:**
- `market` - Public orderbook/price updates
- `user` - Authenticated order status
- `trades` - Trade stream
- `activity` - On-chain activity

### 2.3 Proxy Wallet Factory (New Account Detection)

**Source:** https://docs.polymarket.com/developers/proxy-wallet

**Factory Contracts:**
| Factory | Address | For |
|---------|---------|-----|
| Safe Proxy Factory | `0xaacfeea03eb1561c4e67d661e40682bd20e3541b` | MetaMask users |
| Polymarket Proxy Factory | `0xaB45c5A4B0c941a2F231C04C3f49182e1A254052` | MagicLink users |

**Stats:** 2,220,447 total transactions (wallets created)

**Monitoring:** Subscribe to `ProxyCreation` events via WebSocket

### 2.4 GraphQL Subgraphs

**Source:** https://thegraph.com/docs/en/subgraphs/guides/polymarket/

| Subgraph | Purpose | Endpoint |
|----------|---------|----------|
| Orders | Orderbook analytics | Goldsky hosted |
| Positions | User positions/PnL | Goldsky hosted |
| Activity | Trades/events | Goldsky hosted |
| Open Interest | Market OI | Goldsky hosted |
| PnL | Profit/loss | Goldsky hosted |

**Free Tier:** 100K queries/month

### 2.5 Core Smart Contracts

| Contract | Address | Function |
|----------|---------|----------|
| CTF | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` | ERC-1155 outcome tokens |
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Binary market settlement |
| NegRisk_CTFExchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` | Multi-outcome settlement |
| NegRiskAdapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` | CTF adapter |
| USDC.e Collateral | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | Stablecoin |

---

## Part 3: Data Sources Summary

| Source | URL | Purpose |
|--------|-----|---------|
| CLOB API | `https://clob.polymarket.com` | Trades, orderbook |
| Gamma API | `https://gamma-api.polymarket.com` | Markets, events |
| Data API | `https://data-api.polymarket.com` | Positions, activity |
| CLOB WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Real-time trades |
| RTDS WebSocket | `wss://ws-live-data.polymarket.com` | Real-time activity |
| Polygon RPC | Any provider (Infura, Chainstack, Alchemy) | On-chain events |
| GraphQL Subgraphs | Goldsky/The Graph endpoints | Historical queries |
| PolygonScan | `https://polygonscan.com` | Transaction lookup |

---

## Part 4: Competitive Landscape

| Tool | Monthly Cost | Features |
|------|--------------|----------|
| Polysights | Free + Premium | Insider Finder, 30+ metrics, AI summaries |
| Polywhaler | $10/month | Insider score, whale tracking |
| PolyTrack | Free | Cluster detection, volume spikes |
| Unusual Whales | Subscription | Cross-platform (stocks + crypto) |

---

## Part 5: Technical Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | SQLite | Simple, portable, sufficient for our scale. Can migrate later. |
| Real-time | WebSocket + Polling fallback | WebSocket primary, REST polling for rate-limited scenarios |
| Hosting | Cloud primary, local support | Seamless switching via config |
| Alerts | Browser + Discord + Email | All three supported |
| Historical scope | Full platform | Maximum detection coverage |

---

## References

- Polymarket Docs: https://docs.polymarket.com/
- polymarket-insider-tracker: https://github.com/pselamy/polymarket-insider-tracker
- Polysights: https://app.polysights.xyz/
- The Graph Polymarket: https://thegraph.com/docs/en/subgraphs/guides/polymarket/
- PolygonScan: https://polygonscan.com/
