# Insider Footprint: bigwinner01 (Trump Tariff Short + CZ Pardon)

## Summary
- **Username:** bigwinner01
- **Linked Identity:** Possibly Garrett Jin (former BitForex CEO) - disputed
- **Platform:** Hyperliquid (shorts) + Polymarket (pardons)
- **Total Profit:** ~$250M+ combined
- **Notable Trades:** Bitcoin tariff short ($192M), CZ pardon bet ($56K+)
- **Date Range:** October 2025
- **Status:** Active, denies insider trading

---

## Account Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Trade Size | $100M-$340M positions | Extreme whale |
| Timing Precision | 30 minutes before announcement | Pre-announcement trading |
| Cross-Platform | Hyperliquid + Polymarket | Diversified strategy |
| Win Correlation | Multiple wins on Trump-related events | Pattern specialization |
| Public Defense | Published market analysis | Attempted legitimization |

---

## Major Trades

### Trade 1: Bitcoin Tariff Short (October 10, 2025)

| Attribute | Value |
|-----------|-------|
| Platform | Hyperliquid |
| Position | Short BTC + ETH |
| Size | ~$190M |
| Entry Time | 20:20 UTC |
| Trump Post | 20:50 UTC (30 min later) |
| Event | 100% China tariff announcement |
| Profit | **$192 million** |
| Market Impact | $19.1B liquidations (largest ever) |

**Timeline:**
- 20:20 UTC: Trader opens massive short
- 20:50 UTC: Trump posts tariff announcement on Truth Social
- Minutes later: BTC drops $124K → $105K
- Next day: $192M profit realized

### Trade 2: CZ Pardon Bet (October 2025)

| Attribute | Value |
|-----------|-------|
| Platform | Polymarket |
| Market | Trump pardons CZ in 2025? |
| Position | YES |
| Profit | **$56,000+** |
| Outcome | Trump pardoned CZ on Oct 23 |

**Connection:**
- Same wallet linked to tariff short
- Traced by on-chain analyst "Euan" via Etherscan
- Coffeezilla flagged as "obvious insider knowledge"

---

## Behavioral Signals

### Critical Patterns
1. **30-minute foreknowledge:** Positions opened precisely before announcements
2. **Cross-platform coordination:** Uses derivatives + prediction markets
3. **Extreme position sizing:** $100M+ single trades
4. **Trump-event specialization:** Both trades tied to Trump administration
5. **Public denial + defense:** Published "analysis" to justify trades
6. **Connected identity:** Garrett Jin link (disputed)

### Defense Claims
- "No connection with Trump family"
- "Risk-management trade based on macro factors"
- Published analysis citing: tariffs, weak sentiment, tech correlation, overleveraged markets
- Claims wallet belongs to "clients' fund"

---

## Detection Footprint

```json
{
  "signals": {
    "trade_size_usd": {"min": 1000000, "weight": 20},
    "timing_precision_minutes": {"max": 60, "weight": 30},
    "cross_platform_correlation": {"same_theme": true, "weight": 20},
    "event_category": {"value": "government_policy", "weight": 15},
    "public_denial_after": {"defense_published": true, "weight": 15}
  },
  "category": {
    "type": "policy_announcements",
    "entity": "trump_administration",
    "events": ["tariffs", "pardons", "executive_orders"]
  },
  "behavior": {
    "whale_size": true,
    "pre_announcement": true,
    "cross_platform": true
  }
}
```

---

## Intelligence Assessment

### Why This Is Suspicious

1. **30-minute timing:** Near-impossible without advance knowledge
2. **Largest liquidation event ever:** Suggests intentional market impact
3. **Two Trump-related wins:** Pattern of administration access
4. **CZ-Trump connection:** Pardon linked to $2B stablecoin deal (pay-for-pardon)
5. **Disputed identity:** Multiple name connections suggest obfuscation

### Possible Information Sources
- Trump administration insider
- Crypto industry connection (CZ relationship)
- Policy advisor with announcement timing
- Social media pre-access (Truth Social)

---

## Regulatory Response

- **Senator Warren:** Urged SEC investigation
- **Coffeezilla:** Public "insider trading" allegation
- **Arkham Intelligence:** Labeled wallet "Trump insider whale"
- **CFTC:** Has jurisdiction over Bitcoin derivatives

---

## Wallet Tracking

### Hyperliquid Position Monitoring
Track large shorts on BTC/ETH before:
- Trump speeches
- Policy announcements
- Executive orders
- Tariff changes

### Polymarket Cross-Reference
When same wallet appears on Polymarket:
- Flag immediately
- Check for Trump-related markets
- Monitor position timing vs news cycle

---

## Sources

- https://cryptoslate.com/the-big-bitcoin-short-who-shorted-btc-before-trumps-tariff-post-to-bank-200-million/
- https://www.ccn.com/news/crypto/coffeezilla-trump-cz-pardon-insider-trading-trader/
- https://finance.yahoo.com/news/alleged-trump-insider-whale-denies-211326404.html
- https://decrypt.co/344137/alleged-trump-insider-whale-denies-insider-trading-opens-340-million-bitcoin-short
