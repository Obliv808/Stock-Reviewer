"""Turn a probability estimate into a concrete, sized trade recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config


@dataclass
class TradeSignal:
    ticker: str
    action: str                      # BUY | SELL (exit) | HOLD
    prob_up: float                   # model probability of an up next bar
    last_close: float
    expected_move_pct: float         # (prob_up - 0.5) * recent daily vol, as a %
    entry_estimate: Optional[float]  # expected fill price (next open ~ last close)
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_shares: int
    position_value: float
    risk_amount: float               # $ lost if the stop is hit
    confidence: str                  # LOW / MEDIUM / HIGH
    note: str = ""


def make_signal(
    ticker: str,
    prob_up: float,
    last_close: float,
    atr_pct: float,
    recent_vol: float,
    equity: float,
    cfg: Config,
) -> TradeSignal:
    """Apply the decision rules and ATR-based risk management to a raw probability."""
    expected_move_pct = (prob_up - 0.5) * recent_vol * 100.0

    if prob_up >= cfg.buy_threshold:
        action = "BUY"
    elif prob_up <= cfg.sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"

    stop: Optional[float] = None
    tp: Optional[float] = None
    entry_estimate: Optional[float] = None
    shares = 0
    value = 0.0
    risk = 0.0
    note = ""

    if action == "BUY":
        entry = last_close  # next open is unknown; estimate at last close
        stop_dist = max(atr_pct, 1e-6) * cfg.atr_stop_mult * entry
        stop = entry - stop_dist
        tp = entry + stop_dist * (cfg.atr_target_mult / cfg.atr_stop_mult)
        risk = equity * cfg.risk_per_trade
        shares = int(risk // stop_dist)
        value = shares * entry
        # cap so the position fits in available capital
        max_shares = int((equity * cfg.max_position_fraction) // entry)
        if shares > max_shares:
            shares = max_shares
            value = shares * entry
            risk = shares * stop_dist
        entry_estimate = entry
        if shares == 0:
            action, note = "HOLD", "Position size rounds to zero — increase capital or reduce risk."
    elif action == "SELL":
        note = "Bearish: exit any long position (long-only)."

    dist = abs(prob_up - 0.5)
    confidence = "HIGH" if dist >= 0.12 else ("MEDIUM" if dist >= 0.07 else "LOW")

    return TradeSignal(
        ticker=ticker.upper(),
        action=action,
        prob_up=prob_up,
        last_close=last_close,
        expected_move_pct=expected_move_pct,
        entry_estimate=entry_estimate,
        stop_loss=stop,
        take_profit=tp,
        position_shares=shares,
        position_value=value,
        risk_amount=risk,
        confidence=confidence,
        note=note,
    )
