# Zied Quant v2

OKX Spot (USDT, no stablecoins) scanner with a self-learning pipeline.
**Paper trading only. No real order is ever sent. Not financial advice.**

## Pipeline
1. **Data**: all live `*-USDT` spot pairs, stablecoins removed, 5 timeframes (5m, 15m, 1H, 4H, 1D).
2. **150 features** = 28 technical x 5 timeframes + 10 market-context features
   (beta/correlation to BTC, relative strength, BTC regime/volatility, hour/day, liquidity).
3. **Ranking** of every coin, best to worst.
4. **Models**: 20-layer residual net + gradient boosting (LightGBM, or scikit-learn fallback) + their ensemble.
   The evolved runner chooses the model and confidence threshold on a chronological tuning block, then promotes it only if a separate later holdout has at least 30 selected trades, positive net expectancy after costs, PF > 1.05, and acceptable AUC. AUC alone cannot promote a model.
5. **Labels** = triple barrier (same ATR stop/target as the paper trader, 24h max, costs included).
6. **Paper trading** with realistic costs (fees, slippage, spread, size impact) and risk management:
   volatility-based sizing, max open trades, correlation filter, daily loss limit, loss-streak kill-switch, BTC regime filter.
7. **Self-learning**: every scan logs the top coins; after 24h each logged signal is labelled with what really happened.
   The models retrain on history + labelled signals + real paper-trade outcomes (x3 weight).
8. **Walk-forward backtest** (`backtest`) to check the edge out-of-sample before trusting anything.
9. **Strategy discovery**: daily candidate generation across trend-pullback, breakout and RSI-reversal rule families; parameter mutations are evaluated chronologically with a 24-hour embargo, a 20 bps round-trip cost reserve, and an untouched holdout. A candidate is recorded as champion only with at least 30 holdout trades, positive average net return and PF > 1.05. Discovery is research-only and never changes live entries automatically.
10. **Telegram**: alerts and commands `/stats /positions /top /model /signals /pause /resume`.
11. **Paper LIMIT entries**: signals place discounted limit orders based on volatility and spread; cash is reserved while pending, only a later closed 5m candle can fill an order, and unfilled orders expire after 6 hours with the reserve returned. No real exchange orders are sent.
12. **Portfolio-level risk**: pending orders count toward the five-position limit, correlated assets are filtered against open and pending positions, and aggregate stop-loss risk is capped at 2% of equity with notional exposure capped at 75%.
13. **Leakage control**: feature normalization is fitted only on the model-fit segment, with chronological validation and an embargo before future outcome windows. Walk-forward simulation applies the aggregate stop-risk budget; pairwise correlation still uses live return histories because the stored backtest dataset does not contain synchronized raw return series for every coin.


## Commands
```bash
pip install -r requirements.txt
export TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=...      # optional
python evolved_runner.py train     # guarded train/tune/holdout + model promotion
python evolved_runner.py backtest 5 # walk-forward out-of-sample report
python evolved_runner.py run       # endless loop with guarded model promotion
python evolved_runner.py once      # one guarded cycle
python evolved_runner.py scan | stats | label | refresh
python -m unittest discover -s tests -v # offline tests
```

## Deploy
* **GitHub Actions** (hourly): `.github/workflows/zied-quant.yml`. Add repo secrets `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.
  State is kept in the Actions cache (+ DB backup artifact). Durable state is only overwritten after a successful cycle. The workflow runs offline tests before trading, and Telegram commands are read once per hour.
* **Docker / VPS** (recommended, always on, instant commands, durable state; uses the guarded evolution runner):
  `cp .env.example .env`, fill it, then `docker compose up -d --build`.
* If `www.okx.com` is blocked from your host, set `OKX_BASE=https://aws.okx.com`.

Read the backtest first: if it shows no out-of-sample edge, the signals are not worth following.

## Candlestick patterns and crypto-news intelligence

The runtime entry point `evolved_runner.py` installs `candle_news_intelligence.py`.
The overlay analyzes closed 1H OHLC candles for common bullish, bearish, and neutral
patterns (including Doji variants, Hammer/Shooting Star, Marubozu, Engulfing, Harami,
Piercing Line, Dark Cloud Cover, Tweezer Top/Bottom, Inside/Outside Bar, Morning/Evening
Star, Three White Soldiers, and Three Black Crows). It also reads public crypto-news RSS
feeds, caches headlines for 15 minutes, and applies a small headline-sentiment adjustment
to the top 40 base-ranked coins.

Diagnostic output is saved in `zied_data/candle_patterns.json` and
`zied_data/news_sentiment.json` and is preserved by the existing state persistence flow.
The overlay does **not** change the 300-feature model schema, does not treat missing news
as negative, and cannot bypass AI selection or portfolio risk limits. News sentiment is a
simple keyword heuristic, and chart patterns are uncertain context—not reliable predictions.
The strategy remains paper trading only.
