# Financial Agent

A machine-learning agent that studies a stock's price/volume history and tells you
**what trade to make next**: direction (BUY / SELL / HOLD), entry, ATR-based stop loss,
take profit, and risk-based position size. Every recommendation is backed by an
out-of-sample backtest so you can see how the strategy actually performed.

> ⚠️ **Disclaimer** — This is an educational tool, not financial advice. Daily
direction prediction is extremely hard; models here typically score only slightly
above a coin flip. Never trade money you can't afford to lose.

## Quick start (no terminal needed)

You don't have to use the command line at all:

- **`FinancialAgent.exe`** — double-click it. A small window opens: type a ticker
  (e.g. `NBIS`), pick an action, hit **RUN**. The trade call, backtest and model
  stats print in the window. Works on any Windows PC, no Python required.
- **`Launch Financial Agent.bat`** — same GUI, but launched through your installed
  Python (use this if you ever edit the code and want to test without rebuilding).

Notes:
- First run for a new ticker downloads ~6 years of price history and trains a
  model, so it can take a minute or two. After that it reuses cached data and
  saved models in `data/` and `models/`.
- The exe stores its cache/models next to itself (in the project folder).
- Rebuild the exe after code changes with:
  `python -m PyInstaller --noconfirm --onefile --windowed --name FinancialAgent gui.py`


## How it works

```
┌───────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
│  Data     │ → │  Features  │ → │  Model       │ → │  Strategy  │ → │  Backtest +  │
│ yfinance  │   │ RSI, MACD, │   │ GBM / RF /   │   │ prob →     │   │  Report      │
│ + cache   │   │ ATR, vol…  │   │ Logistic     │   │ trade plan │   │              │
└───────────┘   └────────────┘   └──────────────┘   └────────────┘   └──────────────┘
```

1. **Data** — daily OHLCV from Yahoo Finance (cached in `data/`), with an offline
   synthetic-data fallback so the pipeline still runs without a connection.
2. **Features** — 19 hand-crafted indicators (momentum, trend, volatility,
   volume). No look-ahead: every feature at day *t* uses only data up to *t*.
3. **Model** — predicts P(close tomorrow > close today). Trained on the first 80%
   of history, evaluated on the untouched last 20% (time-ordered split, no shuffling).
4. **Strategy** — probability → action with configurable thresholds; ATR-based
   stop/target; position sized so a stopped-out trade risks ~1% of equity.
5. **Backtest** — simulates the strategy on the held-out window with commission +
   slippage, reports return, Sharpe, max drawdown, win rate vs. buy & hold.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

```bash
# Full pipeline: train → backtest → recommendation (default command)
python main.py analyze AAPL

# Just the trade call for tomorrow (trains automatically if no saved model)
python main.py recommend MSFT --days 1000

# Out-of-sample backtest only
python main.py backtest TSLA

# Train & save a specific model
python main.py train NVDA --model random_forest

# Options: --model {gradient_boosting,random_forest,logistic}
#          --days N (history length), --capital 25000, --refresh (re-download data)
```

### Example output (`python main.py analyze AAPL`)

```
┌────────────────────────── TRAINING REPORT ──────────────────────────
│ Ticker          : AAPL
│ Model           : gradient_boosting
│ Accuracy        : 53.4%   (baseline: 51.0%)
│ ROC-AUC         : 0.552
└──────────────────────────────────────────────────────────────────────

╔══════════════════════ TRADE RECOMMENDATION ══════════════════════╗
║ Ticker           : AAPL                                         ║
║ Last close       : $231.45                                      ║
║ ACTION           : ▲ BUY      confidence: moderate              ║
║ P(up next day)   : 58.2%   (buy ≥ 55%, sell ≤ 45%)             ║
║ Entry (est.)     : $231.90                                      ║
║ Stop loss        : $226.10  (-2.0×ATR)                          ║
║ Take profit      : $241.20  (+3.0×ATR)                          ║
║ Position size    : 171 shares ≈ $39,656                         ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Using it as a library

```python
from financial_agent import FinancialAgent, Config

agent = FinancialAgent(Config(initial_capital=50_000, buy_threshold=0.57))
out = agent.analyze("AMD")          # dict with bundle / backtest / signal
print(out["signal"].action)         # 'BUY' | 'SELL' | 'HOLD'
```

## Project layout

```
Financial Agent/
├── main.py                  # CLI entry point
├── requirements.txt
├── financial_agent/
│   ├── config.py            # all tunable parameters in one place
│   ├── data.py              # yfinance fetch + CSV cache + offline fallback
│   ├── features.py          # indicator / feature engineering (no look-ahead)
│   ├── models.py            # model registry, train/evaluate, save/load
│   ├── strategy.py          # probability → trade plan (stop/target/sizing)
│   ├── backtest.py          # event-loop backtester with costs + metrics
│   └── agent.py             # FinancialAgent orchestrator + reports
├── data/                    # cached OHLCV (auto-created, git-ignored)
└── models/                  # saved model artifacts (auto-created, git-ignored)
```

## Tuning

Everything lives in `financial_agent/config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `buy_threshold` / `sell_threshold` | 0.55 / 0.45 | P(up) needed to act |
| `atr_stop_mult` / `atr_target_mult` | 2.0 / 3.0 | Stop & target in ATRs (1.5 R:R) |
| `risk_per_trade` | 0.01 | Fraction of equity risked per trade |
| `commission_pct` / `slippage_pct` | 5 bps each | Backtest costs |
| `train_test_split` | 0.8 | Time-ordered train/test split |

## Known limitations

- **Daily direction is near-random**: out-of-sample accuracy hovering around 50–55%
  is normal. Judge the *strategy* (backtest Sharpe, drawdown), not raw accuracy.
- No news/fundamentals/macro inputs — price & volume only.
- One position at a time, long-only; stop/target checked on daily bars
  (intraday path assumed stop-first when both are touched).
- Yahoo Finance data quality varies by ticker; use `--refresh` after market close.
