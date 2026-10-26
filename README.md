# Zied Quant

OKX Spot (USDT, no stablecoins) scanner: 150 features (30 x 5 timeframes) -> ranking -> 20-layer deep net -> buy signal -> paper trading + self-learning + Telegram alerts.

**Paper trading only. No real orders are sent. Not financial advice.**

```bash
pip install -r requirements.txt
export TELEGRAM_TOKEN=...      # optional
export TELEGRAM_CHAT_ID=...    # optional
python zied_quant.py train     # first training
python zied_quant.py run       # full loop
python zied_quant.py scan      # one scan, no trading
python zied_quant.py stats     # paper-trading stats
```
