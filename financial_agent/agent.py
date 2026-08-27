"""FinancialAgent — orchestrates data → features → model → trade signal.

Typical usage:
    agent = FinancialAgent()
    agent.analyze("AAPL")          # train + backtest + live recommendation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .data import load_ohlcv
from .features import FEATURES, build_features, make_dataset
from .models import ModelBundle, feature_importance, train_model
from .strategy import TradeSignal, make_signal


def days_to_period(days: int) -> str:
    for d, p in [(21, "1mo"), (63, "3mo"), (126, "6mo"), (252, "1y"),
                 (504, "2y"), (756, "3y"), (1260, "5y"), (2520, "10y")]:
        if days <= d:
            return p
    return "max"


class FinancialAgent:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()

    # ------------------------------------------------------------------ data
    def load_data(self, ticker: str, period: str = "5y", refresh: bool = False):
        return load_ohlcv(
            ticker,
            period=period,
            interval=self.cfg.interval,
            cache_dir=self.cfg.cache_dir,
            max_cache_age_hours=self.cfg.max_cache_age_hours,
            refresh=refresh,
        )

    # ----------------------------------------------------------------- model
    def train(self, ticker: str, model_name: Optional[str] = None,
              period: str = "5y", refresh: bool = False) -> ModelBundle:
        model_name = model_name or self.cfg.model_name
        df, source = self.load_data(ticker, period=period, refresh=refresh)
        if len(df) < self.cfg.min_bars:
            raise ValueError(
                f"Only {len(df)} bars for {ticker} — need at least {self.cfg.min_bars}. "
                "Try a longer --days window."
            )
        X, y = make_dataset(df)
        bundle = train_model(
            X, y, ticker=ticker, model_name=model_name,
            split=self.cfg.train_test_split, random_state=self.cfg.random_state,
        )
        path = bundle.save(self.cfg.models_dir)
        print(f"[train] {ticker} · {model_name} → saved to {path}")
        print(f"[train] data: {source}")
        return bundle

    def _get_bundle(self, ticker: str, model_name: Optional[str], period: str,
                    refresh: bool) -> ModelBundle:
        model_name = model_name or self.cfg.model_name
        b = ModelBundle.load(self.cfg.models_dir, ticker, model_name)
        if b is not None:
            return b
        print(f"[agent] No saved model for {ticker}/{model_name} — training now.")
        return self.train(ticker, model_name=model_name, period=period, refresh=refresh)

    # ------------------------------------------------------------- backtest
    def backtest(self, ticker: str, model_name: Optional[str] = None,
                 period: str = "5y", refresh: bool = False) -> BacktestResult:
        bundle = self._get_bundle(ticker, model_name, period, refresh)
        df, _ = self.load_data(ticker, period=period, refresh=refresh)
        full = build_features(df)
        mask = full[FEATURES].notna().all(axis=1)
        full = full[mask]
        full["prob_up"] = bundle.predict_up_series(full[FEATURES])
        # Evaluate only on the held-out test window (last 20% of bars).
        cut = int(len(full) * self.cfg.train_test_split)
        return run_backtest(full.iloc[cut:], self.cfg)

    # ------------------------------------------------------------ recommend
    def recommend(self, ticker: str, model_name: Optional[str] = None,
                  period: str = "5y", refresh: bool = False) -> TradeSignal:
        bundle = self._get_bundle(ticker, model_name, period, refresh)
        df, _ = self.load_data(ticker, period=period, refresh=refresh)
        full = build_features(df)
        last = full[full[FEATURES].notna().all(axis=1)].iloc[-1]
        prob_up = bundle.predict_up(last)
        return make_signal(
            ticker=ticker,
            prob_up=prob_up,
            last_close=float(last["Close"]),
            atr_pct=float(last["atr_pct"]),
            recent_vol=max(float(last["vol_20"]), 1e-6),
            equity=self.cfg.initial_capital,
            cfg=self.cfg,
        )

    # -------------------------------------------------------------- analyze
    def analyze(self, ticker: str, model_name: Optional[str] = None,
                period: str = "5y", refresh: bool = False) -> dict:
        bundle = self.train(ticker, model_name=model_name, period=period, refresh=refresh)
        result = self.backtest(ticker, model_name=bundle.model_name, period=period)
        signal = self.recommend(ticker, model_name=bundle.model_name, period=period)
        return {"bundle": bundle, "backtest": result, "signal": signal}

    # ------------------------------------------------------------- printing
    def print_train_report(self, bundle: ModelBundle) -> None:
        m = bundle.metrics
        auc = f"{m['roc_auc']:.3f}" if "roc_auc" in m else "n/a"
        top = ", ".join(f"{n} ({v:.2f})" for n, v in feature_importance(bundle, 5)) or "n/a"
        print()
        print("┌────────────────────────── TRAINING REPORT ──────────────────────────")
        print(f"│ Ticker          : {bundle.ticker}")
        print(f"│ Model           : {bundle.model_name}")
        print(f"│ Trained         : {bundle.trained_at}")
        print(f"│ Train window    : {bundle.train_window['start']} → {bundle.train_window['end']}")
        print(f"│ Bars            : {m['n_train']} train / {m['n_test']} test")
        print("├────────────────────────── Out-of-sample ──────────────────────────")
        print(f"│ Accuracy        : {m['accuracy']:.1%}   (baseline: {m['baseline_accuracy']:.1%})")
        print(f"│ ROC-AUC         : {auc}")
        print(f"│ Precision/Recall: {m['precision']:.3f} / {m['recall']:.3f}   (F1 {m['f1']:.3f})")
        print(f"│ Top features    : {top}")
        print("└──────────────────────────────────────────────────────────────────────")

    def print_backtest_report(self, result: BacktestResult) -> None:
        m = result.metrics
        print()
        print("┌────────────────────────── BACKTEST REPORT ──────────────────────────")
        print(f"│ Window          : {result.window['start']} → {result.window['end']}")
        print(f"│ Strategy return : {m['total_return_pct']:+.2f}%   (buy & hold: {m['buy_hold_return_pct']:+.2f}%)")
        print(f"│ Sharpe ratio    : {m['sharpe']:.2f}")
        print(f"│ Max drawdown    : {m['max_drawdown_pct']:.2f}%")
        print(f"│ Trades          : {m['num_trades']}   (win rate {m['win_rate_pct']:.1f}%, avg {m['avg_trade_pct']:+.2f}%)")
        print(f"│ Final equity    : ${m['final_equity']:,.2f}")
        if result.trades:
            print("│ Last trades     :")
            for t in result.trades[-5:]:
                print(f"│   {t['entry_date']} → {t['exit_date']}  "
                      f"${t['entry']:.2f}→${t['exit']:.2f}  {t['pnl_pct']:+.2f}%  ({t['reason']})")
        print("└──────────────────────────────────────────────────────────────────────")

    def print_recommendation(self, signal: TradeSignal, bundle: ModelBundle) -> None:
        m = bundle.metrics
        auc = f"{m['roc_auc']:.3f}" if "roc_auc" in m else "n/a"
        W = 66

        def line(s: str) -> None:
            print("║ " + s.ljust(W - 2) + "║")

        print()
        print("╔" + "═" * W + "╗")
        line("TRADE RECOMMENDATION")
        line(f"Ticker           : {signal.ticker}")
        line(f"Last close       : ${signal.last_close:,.2f}")
        line(f"Model            : {bundle.model_name} (OOS acc {m['accuracy']:.1%}, AUC {auc})")
        print("╠" + "═" * W + "╣")
        marker = {"BUY": "▲", "SELL": "▼", "HOLD": "■"}[signal.action]
        line(f"ACTION           : {marker} {signal.action:<6}  confidence: {signal.confidence}")
        line(f"P(up next day)   : {signal.prob_up:.1%}   (buy ≥ {self.cfg.buy_threshold:.0%}, sell ≤ {self.cfg.sell_threshold:.0%})")
        line(f"Expected move    : {signal.expected_move_pct:+.2f}%")
        if signal.action == "BUY":
            line(f"Entry (est.)     : ${signal.entry_estimate:,.2f}")
            line(f"Stop loss        : ${signal.stop_loss:,.2f}  (-{self.cfg.atr_stop_mult:.1f}×ATR)")
            line(f"Take profit      : ${signal.take_profit:,.2f}  (+{self.cfg.atr_target_mult:.1f}×ATR)")
            line(f"Position size    : {signal.position_shares:,} shares ≈ ${signal.position_value:,.0f}")
            line(f"Risk if stopped  : ${signal.risk_amount:,.2f} ({self.cfg.risk_per_trade:.1%} of equity)")
        if signal.note:
            line(f"Note             : {signal.note}")
        print("╚" + "═" * W + "╝")
        print("  Disclaimer: educational tool — not financial advice.")