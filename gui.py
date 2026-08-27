#!/usr/bin/env python
"""Simple GUI launcher for the Financial Agent — no terminal needed.

Run with:      python gui.py
Or double-click "Launch Financial Agent.bat" / FinancialAgent.exe
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

# When frozen into an .exe, point data/model storage at the exe's folder
# so cached prices and trained models live next to it (stable location).
if getattr(sys, "frozen", False):
    _BASE = os.path.dirname(os.path.abspath(sys.executable))
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

import financial_agent.config as _cfg  # noqa: E402
if getattr(sys, "frozen", False):
    import dataclasses
    for _f in dataclasses.fields(_cfg.Config):
        if _f.name == "cache_dir":
            _f.default = os.path.join(_BASE, "data")
        elif _f.name == "models_dir":
            _f.default = os.path.join(_BASE, "models")

import main as cli  # noqa: E402

MODELS = ["gradient_boosting", "random_forest", "logistic"]
COMMANDS = {
    "Analyze (train + backtest + trade call)": "analyze",
    "Recommend only": "recommend",
    "Backtest only": "backtest",
    "Train model only": "train",
}

BG = "#1e1f24"
FG = "#e8e8ea"
ACCENT = "#4da3ff"
GREEN = "#3ddc84"
RED = "#ff5c5c"
AMBER = "#ffc857"


class _Tee:
    """Routes print() output from the worker thread into a queue."""

    def __init__(self, q: queue.Queue) -> None:
        self.q = q

    def write(self, s: str) -> None:
        if s:
            self.q.put(s)

    def flush(self) -> None:
        pass


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.running = False

        root.title("Financial Agent — Trade Predictor")
        root.geometry("880x560")
        root.minsize(720, 440)
        root.configure(bg=BG)

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton",
                  background=[("active", ACCENT), ("!disabled", "#2b4a75")],
                  foreground=[("disabled", "#777788")])

        self._build_controls()
        self._build_output()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll_queue)

    # ------------------------------------------------------------ layout
    def _build_controls(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(12, 6))

        ttk.Label(top, text="Ticker:").grid(row=0, column=0, sticky="w")
        self.ticker_var = tk.StringVar(value="AAPL")
        ent = ttk.Entry(top, textvariable=self.ticker_var, width=12,
                        font=("Consolas", 12))
        ent.grid(row=0, column=1, padx=(4, 14), sticky="w")
        ent.bind("<Return>", lambda e: self._run())

        ttk.Label(top, text="Model:").grid(row=0, column=2, sticky="e")
        self.model_var = tk.StringVar(value=MODELS[0])
        ttk.Combobox(top, textvariable=self.model_var, values=MODELS,
                     width=18, state="readonly").grid(row=0, column=3, padx=(4, 14))

        ttk.Label(top, text="History (days):").grid(row=0, column=4, sticky="e")
        self.days_var = tk.IntVar(value=1500)
        ttk.Spinbox(top, from_=63, to=2520, increment=63,
                    textvariable=self.days_var, width=7).grid(row=0, column=5, padx=(4, 14))

        ttk.Label(top, text="Capital ($):").grid(row=0, column=6, sticky="e")
        self.capital_var = tk.StringVar(value="100000")
        ttk.Entry(top, textvariable=self.capital_var,
                  width=10).grid(row=0, column=7, padx=(4, 14))

        self.refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Refresh data",
                        variable=self.refresh_var).grid(row=0, column=8, sticky="w")

        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", padx=12, pady=(0, 6))

        ttk.Label(row2, text="Action:").pack(side="left")
        self.cmd_var = tk.StringVar(value=list(COMMANDS)[0])
        ttk.Combobox(row2, textvariable=self.cmd_var,
                     values=list(COMMANDS), width=36,
                     state="readonly").pack(side="left", padx=(4, 14))

        self.run_btn = ttk.Button(row2, text="▶  RUN", style="Accent.TButton",
                                  command=self._run)
        self.run_btn.pack(side="left")

        ttk.Button(row2, text="Clear output",
                   command=lambda: self.out.delete("1.0", "end")).pack(
            side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready — type a ticker and press RUN.")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground=AMBER).pack(fill="x", padx=14)

    def _build_output(self) -> None:
        wrap = ttk.Frame(self.root)
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.out = scrolledtext.ScrolledText(
            wrap, wrap="none", state="disabled",
            font=("Consolas", 10), bg="#14151a", fg=FG,
            insertbackground=FG, relief="flat", padx=10, pady=8)
        self.out.pack(fill="both", expand=True)

        self.out.tag_configure("error", foreground=RED)
        self.out.tag_configure("action_buy", foreground=GREEN)
        self.out.tag_configure("action_sell", foreground=RED)
        self.out.tag_configure("action_hold", foreground=AMBER)
        self.out.tag_configure("section", foreground=ACCENT)

    # ------------------------------------------------------------- output
    def _append(self, text: str) -> None:
        if not text:
            return
        for chunk in text.splitlines(keepends=True):
            tags = []
            low = chunk.lower()
            if "traceback" in low or low.startswith("error"):
                tags = ["error"]
            elif "action" in low and "buy" in low:
                tags = ["action_buy"]
            elif "action" in low and "sell" in low:
                tags = ["action_sell"]
            elif "action" in low and "hold" in low:
                tags = ["action_hold"]
            elif chunk.startswith("╔") or "out-of-sample" in low \
                    or "backtest" in low or low.startswith("[train]"):
                tags = ["section"]
            self.out.configure(state="normal")
            self.out.insert("end", chunk, tags)
            self.out.see("end")
            self.out.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:  # worker finished
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.status_var.set("Done.")
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ---------------------------------------------------------------- run
    def _run(self) -> None:
        if self.running:
            return

        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showwarning("Financial Agent",
                                   "Please enter a ticker (e.g. AAPL).")
            return
        try:
            capital = float(self.capital_var.get().replace(",", ""))
        except ValueError:
            messagebox.showwarning("Financial Agent", "Capital must be a number.")
            return

        command = COMMANDS[self.cmd_var.get()]
        argv = [command, ticker,
                "--model", self.model_var.get(),
                "--days", str(self.days_var.get()),
                "--capital", str(capital)]
        if self.refresh_var.get():
            argv.append("--refresh")

        self.running = True
        self.run_btn.configure(state="disabled")
        self.status_var.set(f"Running {command} for {ticker} … (first run may take a minute)")
        self._append(f"\n$ financial-agent {' '.join(argv)}\n{'-' * 60}\n")

        threading.Thread(target=self._worker, args=(argv,), daemon=True).start()

    def _worker(self, argv: list[str]) -> None:
        old_out, old_err = sys.stdout, sys.stderr
        tee = _Tee(self.q)
        sys.stdout = sys.stderr = tee
        try:
            rc = cli.main(argv)
            if rc != 0:
                self.q.put("\n[finished with errors — see above]\n")
        except Exception:  # noqa: BLE001
            import traceback
            self.q.put("\n" + traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self.q.put(None)

    def _on_close(self) -> None:
        if self.running and not messagebox.askokcancel(
                "Financial Agent", "Analysis still running. Quit anyway?"):
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — visible even in a windowed .exe
        import traceback
        tkinter.messagebox.showerror("Financial Agent", traceback.format_exc())
