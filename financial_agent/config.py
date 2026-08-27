"""Central configuration for the Financial Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # --- Data ---------------------------------------------------------------
    default_period: str = "5y"          # initial fetch window when no --days given
    interval: str = "1d"
    cache_dir: str = os.path.join(PROJECT_ROOT, "data")
    models_dir: str = os.path.join(PROJECT_ROOT, "models")
    max_cache_age_hours: float = 12.0   # re-download if cache is older

    # --- Features / target ----------------------------------------------------
    min_bars: int = 300                 # minimum history required to train

    # --- Model ----------------------------------------------------------------
    model_name: str = "gradient_boosting"   # gradient_boosting | random_forest | logistic
    train_test_split: float = 0.80          # chronological split (no shuffling)

    # --- Strategy -------------------------------------------------------------
    buy_threshold: float = 0.55           # go long if P(up) >= this
    sell_threshold: float = 0.45          # exit / short signal if P(up) <= this
    atr_stop_mult: float = 2.0            # stop distance = ATR * mult
    atr_target_mult: float = 3.0          # target distance = ATR * mult
    risk_per_trade: float = 0.01          # risk 1% of equity per trade
    max_position_fraction: float = 0.95   # never deploy more than 95% of cash

    # --- Backtest -------------------------------------------------------------
    initial_capital: float = 100_000.0
    commission_pct: float = 0.0005        # 5 bps per side
    slippage_pct: float = 0.0005          # 5 bps adverse fill assumption

    # --- Misc -----------------------------------------------------------------
    random_state: int = 42
