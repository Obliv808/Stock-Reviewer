#!/usr/bin/env python
"""CLI entry point for the Financial Agent.

Examples:
    python main.py analyze AAPL                 # full pipeline (default command)
    python main.py recommend MSFT --days 1000   # just the trade call
    python main.py backtest TSLA                # out-of-sample backtest
    python main.py train NVDA --model random_forest
"""

from __future__ import annotations

import argparse
import sys

from financial_agent.agent import FinancialAgent, days_to_period
from financial_agent.config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="financial-agent",
        description="ML agent that predicts trades for a given stock.",
    )
    sub = p.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("ticker", help="Stock ticker, e.g. AAPL, MSFT, 7203.T")
    common.add_argument("--model", default=None,
                        choices=["gradient_boosting", "random_forest", "logistic"],
                        help="Model to use (default: gradient_boosting)")
    common.add_argument("--days", type=int, default=1500,
                        help="History length in trading days (default 1500 ≈ 6y)")
    common.add_argument("--capital", type=float, default=100_000.0,
                        help="Account equity used for position sizing (default $100k)")
    common.add_argument("--refresh", action="store_true",
                        help="Force re-download of market data")

    sub.add_parser("analyze", parents=[common],
                   help="Train + backtest + recommendation (default)")
    sub.add_parser("recommend", parents=[common], help="Print the trade recommendation")
    sub.add_parser("backtest", parents=[common], help="Run the out-of-sample backtest")
    sub.add_parser("train", parents=[common], help="Train and save a model")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "analyze"

    cfg = Config(model_name=args.model or "gradient_boosting",
                 initial_capital=args.capital)
    agent = FinancialAgent(cfg)
    period = days_to_period(args.days)

    try:
        if command == "train":
            bundle = agent.train(args.ticker, model_name=cfg.model_name,
                                 period=period, refresh=args.refresh)
            agent.print_train_report(bundle)
        elif command == "backtest":
            result = agent.backtest(args.ticker, model_name=cfg.model_name,
                                    period=period, refresh=args.refresh)
            agent.print_backtest_report(result)
        elif command == "recommend":
            bundle = agent._get_bundle(args.ticker, cfg.model_name, period, args.refresh)
            signal = agent.recommend(args.ticker, model_name=cfg.model_name,
                                     period=period, refresh=args.refresh)
            agent.print_recommendation(signal, bundle)
        else:  # analyze
            out = agent.analyze(args.ticker, model_name=cfg.model_name,
                                period=period, refresh=args.refresh)
            agent.print_train_report(out["bundle"])
            agent.print_backtest_report(out["backtest"])
            agent.print_recommendation(out["signal"], out["bundle"])
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())