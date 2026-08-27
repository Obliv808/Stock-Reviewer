"""Feature engineering: technical indicators + target variable.

Every feature at row t uses information available at or before the close of day t.
The target looks ahead one bar (next-day direction), which is only used as a label —
never as an input — so there is no look-ahead leakage in training or live signals.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "sma_20_ratio", "sma_50_ratio",
    "macd_norm", "macd_hist_norm",
    "rsi_14",
    "bb_pctb",
    "atr_pct",
    "vol_20", "vol_60",
    "volume_ratio",
    "hl_range",
    "gap_open",
    "dow",
]

TARGET = "y_dir"          # 1 if next close > today's close, else 0
TARGET_RET = "y_ret"      # next-day simple return (for diagnostics)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the OHLCV frame with all model features + targets added."""
    out = df.copy()
    c = out["Close"]

    # Momentum / returns
    for n in (1, 2, 3, 5, 10):
        out[f"ret_{n}"] = c.pct_change(n)

    # Trend
    out["sma_20_ratio"] = c / c.rolling(20).mean() - 1.0
    out["sma_50_ratio"] = c / c.rolling(50).mean() - 1.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_norm"] = macd / c
    out["macd_hist_norm"] = (macd - signal) / c

    # Oscillators
    out["rsi_14"] = _rsi(c)

    mid = c.rolling(20).mean()
    std = c.rolling(20).std(ddof=0)
    upper, lower = mid + 2 * std, mid - 2 * std
    out["bb_pctb"] = ((c - lower) / (upper - lower)).replace([np.inf, -np.inf], np.nan)

    # Volatility / volume
    atr = _atr(out)
    out["atr_pct"] = atr / c
    ret1 = c.pct_change(1)
    out["vol_20"] = ret1.rolling(20).std(ddof=0)
    out["vol_60"] = ret1.rolling(60).std(ddof=0)
    vol_ma = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = (out["Volume"] / vol_ma).replace([np.inf, -np.inf], np.nan)

    # Bar shape / calendar
    out["hl_range"] = (out["High"] - out["Low"]) / c
    out["gap_open"] = out["Open"] / c.shift(1) - 1.0
    out["dow"] = out.index.dayofweek.astype(float)

    # Targets (label only — never fed to the model)
    out[TARGET] = (c.shift(-1) > c).astype(float)
    out.loc[out.index[-1], TARGET] = np.nan  # last bar has no future
    out[TARGET_RET] = c.pct_change(1).shift(-1)

    return out


def make_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop warm-up NaNs and rows without a label; return (X, y)."""
    full = build_features(df)
    mask = full[FEATURES + [TARGET]].notna().all(axis=1)
    full = full[mask]
    if len(full) < 60:
        raise ValueError(f"Not enough usable rows to build a dataset ({len(full)}).")
    return full[FEATURES], full[TARGET]
