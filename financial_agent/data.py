"""Market data acquisition: Yahoo Finance with local CSV cache + offline fallback."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cache_dir: str, ticker: str, period: str, interval: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir, f"{ticker.upper()}_{period}_{interval}.csv"
    )


def _fetch_yahoo(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf  # lazy import so offline mode works without it

    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker!r}")
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def _synthetic_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Deterministic GBM-with-regimes fallback so the agent still runs offline.

    Clearly labelled as synthetic — never use it for real decisions.
    """
    seed = int(hashlib.md5(ticker.upper().encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504,
            "3y": 756, "5y": 1260, "max": 2520}.get(period, 1260)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    # Regime-switching drift so the series has structure to learn from.
    price = 100.0 * (1 + rng.uniform(0, 8))
    prices, vols = [], []
    mu = 0.0005
    for _ in range(days):
        if rng.random() < 0.02:                      # regime switch
            mu = rng.choice([-0.0015, -0.0005, 0.0005, 0.0015])
        sigma = 0.015 + 0.01 * abs(np.sin(_ / 40))
        ret = mu + sigma * rng.standard_normal()
        price *= np.exp(ret)
        o = price * np.exp(0.2 * sigma * rng.standard_normal())
        hi = max(o, price) * (1 + abs(rng.normal(0, 0.5)) * sigma)
        lo = min(o, price) * (1 - abs(rng.normal(0, 0.5)) * sigma)
        prices.append([o, hi, lo, price])
        vols.append(int(rng.lognormal(mean=16.5, sigma=0.3)))

    df = pd.DataFrame(prices, index=idx, columns=["Open", "High", "Low", "Close"])
    df["Volume"] = vols
    return df


def load_ohlcv(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    cache_dir: str = "data",
    max_cache_age_hours: float = 12.0,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Return (ohlcv_df, source_label). Tries cache, then Yahoo, then synthetic."""
    path = _cache_path(cache_dir, ticker, period, interval)

    # 1) Fresh-enough cache?
    if not refresh and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < max_cache_age_hours:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df, f"cache ({path}, {age_h:.1f}h old)"

    # 2) Yahoo Finance
    try:
        df = _fetch_yahoo(ticker, period, interval)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)
        return df, f"Yahoo Finance ({len(df)} bars)"
    except Exception as exc:  # network down / bad ticker
        print(f"[data] Yahoo fetch failed for {ticker!r}: {exc}")

    # 3) Stale cache if we have one
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df, "stale cache (offline)"

    # 4) Synthetic fallback
    print("[data] WARNING: using SYNTHETIC offline data — results are for demo only.")
    return _synthetic_ohlcv(ticker, period, interval), "synthetic (offline)"
