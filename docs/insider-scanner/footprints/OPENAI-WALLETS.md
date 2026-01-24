# Insider Footprint: OpenAI Insider Wallets

## Summary
- **Accounts:** Multiple (specific addresses unknown)
- **Events:** OpenAI product launches (Browser, GPT-5.2, Gemini 3.0)
- **Combined Profit:** $20,000+ documented
- **Date Range:** October - December 2025
- **Status:** No investigation, corporate policy changes in progress

---

## Documented Cases

### Case 1: OpenAI Browser (October 2025)

| Attribute | Value |
|-----------|-------|
| Wallet | New (brand new at time of bet) |
| Bet | OpenAI launches AI browser by Oct 31 |
| Amount | $40,000 |
| Entry Odds | Low (exact unknown) |
| Profit | $7,000 |
| Outcome | Browser launched, bet won |

### Case 2: GPT-5.2 Release (December 2025)

| Attribute | Value |
|-----------|-------|
| Wallets | 4 accounts total |
| Bet | OpenAI releases new LLM by Dec 13 |
| Combined Bet | Unknown |
| Combined Profit | $13,000+ |
| Timing | Bets placed "one week before" release |
| Outcome | GPT-5.2 released Dec 11, bet won |

### Case 3: Gemini 3.0 Launch (0xafEe)

| Attribute | Value |
|-----------|-------|
| Wallet | 0xafEe (same as Google insider) |
| Bet | Gemini 3.0 exact launch day |
| Profit | $150,000 |
| Accuracy | Exact date predicted |

---

## Pattern Analysis

### Common Characteristics

| Signal | OpenAI Browser | GPT-5.2 | Gemini 3.0 |
|--------|----------------|---------|------------|
| New Wallet | Yes | Yes | No (repeat) |
| Large Single Bet | $40K | Varied | Yes |
| Timing Before Event | Days | ~1 week | Days |
| Exact Date Prediction | No | ~2 days early | Yes |
| Category | Tech/AI | Tech/AI | Tech/AI |

### Tech Insider Profile
1. **Focus:** AI/tech product launches
2. **Bet Type:** Release dates, product announcements
3. **Information Source:** Likely internal roadmaps
4. **Pattern:** New wallet per major bet
5. **Win Rate:** 100% on documented cases

---

## Detection Footprint

```json
{
  "signals": {
    "market_category": {"value": "tech_product_launch", "weight": 25},
    "company_focus": {"values": ["openai", "google", "anthropic", "meta"], "weight": 20},
    "bet_type": {"value": "release_date", "weight": 20},
    "wallet_freshness": {"new_for_large_bet": true, "weight": 20},
    "timing_precision": {"exact_or_near_date": true, "weight": 15}
  },
  "behavior": {
    "entry_timing": "days_to_week_before",
    "exit_strategy": "hold_to_resolution",
    "category_expertise": "ai_tech"
  }
}
```

---

## Tech Company Response

### Corporate Policy Changes

> "Discussions with corporate clients about whether to include prediction markets in insider trading policies have at least doubled in the past six months." - Conway Dodge, KPMG Partner

| Company | Policy Status |
|---------|---------------|
| Robinhood | Updated > 1 year ago to cover prediction markets |
| Coinbase | Expanded to prohibit employee participation |
| OpenAI | Unknown |
| Google | Unknown |

---

## AI Product Launch Market Monitoring

### High-Value Target Markets
1. OpenAI model releases (GPT-6, etc.)
2. Google AI releases (Gemini updates)
3. Anthropic releases (Claude updates)
4. Meta AI releases (LLaMA updates)
5. Apple AI announcements
6. Microsoft AI announcements

### Detection Strategy
1. Monitor all AI/tech launch markets
2. Flag new accounts entering with > $10K
3. Track accounts with prior AI market wins
4. Watch for timing clusters (multiple accounts betting within hours)
5. Cross-reference with tech company event calendars

---

## API Queries

### Find AI-Related Markets
```
GET https://gamma-api.polymarket.com/markets?tag=AI
GET https://gamma-api.polymarket.com/markets?query=openai
GET https://gamma-api.polymarket.com/markets?query=release
```

### Monitor Large Trades on AI Markets
```
# For each AI market:
GET https://data-api.polymarket.com/trades?conditionId={ai_market_id}&size_gte=10000
```

### Profile New Entrants
```
# For each large trader:
GET https://data-api.polymarket.com/trades?user={wallet}&limit=100

# If < 5 prior trades AND current bet > $10K → FLAG
```

---

## Risk Assessment

### Why Tech Markets Are High Risk
1. **Large companies:** Many employees with inside info
2. **Clear timelines:** Product roadmaps are known internally
3. **Binary outcomes:** Did X launch by Y date?
4. **High liquidity:** Large bets can be placed
5. **Low enforcement:** No SEC oversight

### Recommended Monitoring Priority
| Market Type | Priority |
|-------------|----------|
| OpenAI releases | HIGH |
| Google AI releases | HIGH |
| Apple announcements | MEDIUM |
| Startup launches | LOW |

---

## Sources

- https://www.theinformation.com/articles/polymarket-bets-openai-google-raise-insider-trading-suspicions
- https://gizmodo.com/tracking-insider-trading-on-polymarket-is-turning-into-a-business-of-its-own-2000709286
- https://www.kucoin.com/news/flash/polymarket-accounts-profit-from-openai-and-google-product-launches-sparking-insider-trading-concerns
