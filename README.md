# Zied Quant v2

OKX Spot (USDT, no stablecoins) scanner with a self-learning pipeline.
**Paper trading only. No real order is ever sent. Not financial advice.**

## Pipeline
1. **Data**: all live `*-USDT` spot pairs, stablecoins removed, 5 timeframes (5m, 15m, 1H, 4H, 1D).
2. **150 features** = 28 technical x 5 timeframes + 10 market-context features
   (beta/correlation to BTC, relative strength, BTC regime/volatility, hour/day, liquidity).
3. **Ranking** of every coin, best to worst.
4. **Models**: 20-layer residual net + gradient boosting (LightGBM, or scikit-learn fallback) + their ensemble.
   The best one is chosen on a time-ordered validation block (with an embargo) and the buy threshold comes from it.
5. **Labels** = triple barrier (same ATR stop/target as the paper trader, 24h max, costs included).
6. **Paper trading** with realistic costs (fees, slippage, spread, size impact) and risk management:
   volatility-based sizing, max open trades, correlation filter, daily loss limit, loss-streak kill-switch, BTC regime filter.
7. **Self-learning**: every scan logs the top coins; after 24h each logged signal is labelled with what really happened.
   The models retrain on history + labelled signals + real paper-trade outcomes (x3 weight).
8. **Walk-forward backtest** (`backtest`) to check the edge out-of-sample before trusting anything.
9. **Telegram**: alerts and commands `/stats /positions /top /model /signals /pause /resume`.

## Commands
```bash
pip install -r requirements.txt
export TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=...      # optional
python zied_quant.py train          # build dataset (30 days, cached) + train
python zied_quant.py backtest 5     # walk-forward out-of-sample report
python zied_quant.py run            # endless loop (VPS / Docker)
python zied_quant.py once           # one cycle (used by the hourly GitHub workflow)
python zied_quant.py scan | stats | label | refresh
python -m pytest -q                 # offline tests
```

## Deploy
* **GitHub Actions** (hourly): `.github/workflows/zied-quant.yml`. Add repo secrets `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.
  State is kept in the Actions cache (+ DB backup artifact). Telegram commands are read once per hour.
* **Docker / VPS** (recommended, always on, instant commands, durable state):
  `cp .env.example .env`, fill it, then `docker compose up -d --build`.
* If `www.okx.com` is blocked from your host, set `OKX_BASE=https://aws.okx.com`.

Read the backtest first: if it shows no out-of-sample edge, the signals are not worth following.
