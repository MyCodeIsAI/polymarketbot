# Insider Footprint: Théo Cluster (French Trader)

## Summary
- **Real Identity:** French national, "Théo" (pseudonym)
- **Background:** "Extensive trading experience and financial services background"
- **Accounts:** 4 confirmed, 11 total (per Chainalysis)
- **Event:** 2024 US Presidential Election
- **Total Wagered:** ~$70M
- **Profit:** ~$85M
- **Status:** No insider trading allegation (claimed superior analysis)

---

## Confirmed Account Cluster

| Username | Holdings (Pre-Election) | Funding Source |
|----------|------------------------|----------------|
| Fredi9999 | $12,300,000 | Kraken |
| Theo4 | $5,000,000 | Kraken |
| PrincessCaro | $3,900,000 | Kraken |
| Michie | $3,500,000 | Kraken |
| **TOTAL** | ~$24,700,000 | Same exchange |

Chainalysis identified **11 total accounts** linked to same entity via wallet clustering.

---

## Cluster Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Accounts Linked | 11 | Multi-account operation |
| Common Funding Source | Kraken | Same exchange |
| Market Concentration | Trump election | 100% focused |
| Position in Market | 25% of Trump Electoral College | Whale |
| Position in Popular Vote | 40%+ | Market mover |
| Trading Strategy | Contrarian to polls | "Shy Trump voter" thesis |

---

## Wallet Clustering Signals

### Chainalysis Methodology Applied:
1. **Same funding source:** All wallets funded from Kraken
2. **Temporal correlation:** Accounts active in similar timeframes
3. **Proportional sizing:** Position sizes suggest coordinated allocation
4. **Behavioral synchronization:** Similar trading patterns
5. **Market overlap:** 100% overlap on same markets

### Transaction Patterns

| Account | Trading Style | Typical Tx Size |
|---------|--------------|-----------------|
| Fredi9999 | Large single trades | $4,302+ |
| PrincessCaro | Frequent small trades | $0.30 - $187 |
| Theo4 | Medium trades | Varied |
| Michie | Medium trades | Varied |

**Note:** Different trading styles may be intentional to avoid clustering detection.

---

## Detection Footprint

```json
{
  "cluster_signals": {
    "common_funding_source": {"same_exchange": true, "weight": 30},
    "market_overlap": {"min": 0.90, "weight": 25},
    "temporal_correlation": {"same_week_creation": true, "weight": 20},
    "position_proportionality": {"similar_allocation": true, "weight": 15},
    "combined_market_share": {"threshold": 0.10, "weight": 10}
  },
  "individual_signals": {
    "large_position_size": {"min_usd": 1000000, "weight": 20},
    "single_market_focus": {"concentration": 0.95, "weight": 15},
    "contrarian_position": {"against_consensus": true, "weight": 10}
  }
}
```

---

## Differentiation: Insider vs Whale

**Why This May NOT Be Insider Trading:**
1. **Public Thesis:** Claimed "shy Trump voter" polling methodology
2. **Commissioned Own Polls:** Funded independent research
3. **Long Accumulation:** Weeks of trading (Oct 7+)
4. **Market Making Role:** Provided liquidity, not front-running

**Why This IS Relevant for Detection:**
1. **Multi-account patterns:** Same methods used by actual insiders
2. **Funding source correlation:** Technique to identify Sybil accounts
3. **Market concentration:** Pattern shared with insiders
4. **Scale:** Demonstrates whale behavior detection

---

## API Queries for Cluster Detection

### Step 1: Identify Suspicious Wallet
```
GET https://data-api.polymarket.com/trades?market={trump_election_market}
```
Filter for large trades (> $100K).

### Step 2: Extract Funding Sources
For each wallet, query PolygonScan:
```
GET https://api.polygonscan.com/api?module=account&action=tokentx&address={wallet}
```
Find first USDC deposit → record `from` address.

### Step 3: Cluster by Funding Source
Group wallets by:
- Same `from` address
- Same exchange deposit address pattern
- Temporal proximity of funding

### Step 4: Validate Cluster
Check for:
- Market overlap > 90%
- Combined position size
- Trading pattern similarity

---

## Kraken Detection Pattern

Kraken deposit addresses often follow patterns:
1. Hot wallet addresses rotated periodically
2. Large deposits may come from same hot wallet
3. Withdrawal addresses can be correlated

---

## Sources

- https://www.cnbc.com/2024/10/24/polymarket-trump-french-election-bet.html
- https://fortune.com/2024/10/24/polymarket-crypto-trump-win-presidential-election/
- https://cointelegraph.com/news/polymarket-french-whale-donald-trump-election-odds
- https://www.dlnews.com/articles/defi/4-crypto-whales-are-skewing-trump-polymarket-election-bet/
