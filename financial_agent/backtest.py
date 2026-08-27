"""Event-driven backtester for the agent's long-only strategy.

Execution model (no look-ahead):
  * The probability computed from bar t is known at the close of day t.
  * Entries/exits therefore happen on day t+1: entries at the open, and
    stop / target / signal exits are checked intrabar (stop assumed to fill
    before target when both are touched — conservative).
  * Commission and slippage are charged on every fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series                       # daily equity curve
    trades: List[Dict]                      # one dict per round trip
    metrics: Dict[str, float]
    window: Dict[str, str]


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    """df must contain: Open/High/Low/Close, prob_up (aligned to signal day), atr_pct."""
    cash = cfg.initial_capital
    shares = 0
    entry = stop = target = None
    entry_date = None
    trades: List[Dict] = []
    curve: Dict[pd.Timestamp, float] = {}

    comm, slip = cfg.commission_pct, cfg.slippage_pct

    for i in range(1, len(df)):
        bar = df.iloc[i]
        sig = df.iloc[i - 1]              # signal known at previous close
        p = sig["prob_up"]
        atr_dist = float(sig["atr_pct"]) * float(sig["Close"]) * cfg.atr_stop_mult

        exit_price: float | None = None
        reason = ""

        if shares > 0:
            # Intrabar exits — stop takes priority (conservative).
            if bar["Low"] <= stop:
                exit_price, reason = stop * (1 - slip), "stop_loss"
            elif bar["High"] >= target:
                exit_price, reason = target * (1 - slip), "take_profit"
            elif not np.isnan(p) and p <= cfg.sell_threshold:
                exit_price, reason = bar["Close"] * (1 - slip), "signal_exit"

            if exit_price is not None:
                cash += shares * exit_price * (1 - comm)
                pnl_pct = (exit_price / entry - 1.0) * 100
                trades.append(
                    {
                        "entry_date": str(entry_date.date()),
                        "exit_date": str(bar.name.date()),
                        "entry": round(entry, 4),
                        "exit": round(exit_price, 4),
                        "shares": shares,
                        "pnl_pct": round(pnl_pct, 2),
                        "reason": reason,
                    }
                )
                shares, entry, stop, target, entry_date = 0, None, None, None, None

        if shares == 0 and not np.isnan(p) and p >= cfg.buy_threshold:
            fill = bar["Open"] * (1 + slip)
            stop = fill - atr_dist
            target = fill + atr_dist * (cfg.atr_target_mult / cfg.atr_stop_mult)
            risk_amt = cash * cfg.risk_per_trade
            n = int(risk_amt // max(atr_dist, 1e-9))
            n = min(n, int((cash * cfg.max_position_fraction) // fill))
            if n > 0:
                cost = n * fill * (1 + comm)
                if cost <= cash:
                    cash -= cost
                    shares, entry, entry_date = n, fill, bar.name

        curve[bar.name] = cash + shares * bar["Close"]

    # Mark any open position to the last close.
    equity = pd.Series(curve).sort_index()
    if len(equity) < 10:
        raise ValueError("Backtest window too short.")

    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / cfg.initial_capital - 1.0
    sharpe = float(
        (rets.mean() / rets.std(ddof=0) * np.sqrt(TRADING_DAYS)) if rets.std(ddof=0) > 0 else 0.0
    )
    dd = equity / equity.cummax() - 1.0
    wins = [t for t in trades if t["pnl_pct"] > 0]
    bh_first, bh_last = df.iloc[1]["Open"], df.iloc[-1]["Close"]

    metrics = {
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round((bh_last / bh_first - 1.0) * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(dd.min() * 100, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "avg_trade_pct": round(float(np.mean([t["pnl_pct"] for t in trades])), 2) if trades else 0.0,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }

    window = {"start": str(df.index[1].date()), "end": str(df.index[-1].date())}
    return BacktestResult(equity=equity, trades=trades, metrics=metrics, window=window)
