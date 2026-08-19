#!/usr/bin/env python3
"""Live paper-trading configuration loader.

Reads `live_config.json` (next to this file). Sensitive values can be
overridden through environment variables so the JSON never has to hold a
live token in source control:

    DHAN_CLIENT_ID     overrides config.dhan.client_id
    DHAN_ACCESS_TOKEN  overrides config.dhan.access_token

Multi-strategy format:

    "strategies": [
        { "name": "dema_15m", "fast": "15m", "slow": "1h",
          "capital": 2000000, "max_positions": 20,
          "max_capital_per_stock": 100000, "slippage_pct": 0.0005,
          "sl_pct": 0.0, "tp_pct": 0.0, "dema_period": 3,
          "atr_period": 6, "atr_factor": 1.0, "min_bars": 11 },
        { "name": "dema_5m",  "fast": "5m",  "slow": "15m", ... }
    ]

Every strategy trades the SAME symbol universe (the `stocks` block) but keeps
its own capital, position slots, per-stock cap, SL/TP and trades. For backward
compatibility, a config without a `strategies` array is interpreted as a single
strategy built from the old `strategy` + `paper` blocks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "live_config.json"


class StrategyConfig:
    """One strategy: its DEMA-ATR parameters, risk limits and own capital."""

    def __init__(self, data: dict, defaults: dict | None = None):
        d = defaults or {}
        self.name = data.get("name", d.get("name", "dema_atr"))
        self.fast = data.get("fast", d.get("fast", "15m"))
        self.slow = data.get("slow", d.get("slow", "1h"))
        self.dema_period = data.get("dema_period", d.get("dema_period", 3))
        self.atr_period = data.get("atr_period", d.get("atr_period", 6))
        self.atr_factor = data.get("atr_factor", d.get("atr_factor", 1.0))
        self.min_bars = data.get("min_bars", d.get("min_bars", 11))
        self.sl_pct = data.get("sl_pct", d.get("sl_pct", 0.0))
        self.tp_pct = data.get("tp_pct", d.get("tp_pct", 0.0))
        self.capital = data.get("capital", d.get("capital", 2000000.0))
        self.max_positions = data.get("max_positions", d.get("max_positions", 20))
        self.max_capital_per_stock = data.get(
            "max_capital_per_stock", d.get("max_capital_per_stock", 100000.0))
        self.slippage_pct = data.get("slippage_pct", d.get("slippage_pct", 0.0005))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "fast": self.fast, "slow": self.slow,
            "dema_period": self.dema_period, "atr_period": self.atr_period,
            "atr_factor": self.atr_factor, "min_bars": self.min_bars,
            "sl_pct": self.sl_pct, "tp_pct": self.tp_pct,
            "capital": self.capital, "max_positions": self.max_positions,
            "max_capital_per_stock": self.max_capital_per_stock,
            "slippage_pct": self.slippage_pct,
        }


class LiveConfig:
    def __init__(self, path: Path | str = CONFIG_PATH):
        with open(path) as f:
            self.raw = json.load(f)

        dhan = self.raw.get("dhan", {})
        self.client_id = os.environ.get("DHAN_CLIENT_ID", dhan.get("client_id"))
        self.access_token = os.environ.get("DHAN_ACCESS_TOKEN", dhan.get("access_token"))
        self.exchange_segment = dhan.get("exchange_segment", "NSE_EQ")
        self.instrument_type = dhan.get("instrument_type", "EQUITY")

        stocks = self.raw.get("stocks", {})
        self.symbols = [s for s, c in stocks.items() if c.get("enabled", True)]
        self.stock_capital = {s: c.get("capital", 100000) for s, c in stocks.items()}

        # strategies: [] array, each with own params + capital + risk limits
        defaults = {
            **self.raw.get("strategy", {}),
            **{k: v for k, v in self.raw.get("paper", {}).items()
               if k in ("capital", "max_positions", "max_capital_per_stock",
                        "slippage_pct")},
        }
        self.strategies = [StrategyConfig(s, defaults)
                           for s in self.raw.get("strategies", [])]
        if not self.strategies:  # backward compat: single strategy from old blocks
            self.strategies = [StrategyConfig(defaults, {})]

        # legacy attributes = first strategy (kept for existing code/tests)
        s0 = self.strategies[0]
        self.strategy_name = s0.name
        self.fast, self.slow = s0.fast, s0.slow
        self.dema_period, self.atr_period, self.atr_factor = (
            s0.dema_period, s0.atr_period, s0.atr_factor)
        self.min_bars = s0.min_bars
        self.sl_pct, self.tp_pct = s0.sl_pct, s0.tp_pct
        self.capital = s0.capital
        self.max_positions = s0.max_positions
        self.max_capital_per_stock = s0.max_capital_per_stock
        self.slippage_pct = s0.slippage_pct

        paper = self.raw.get("paper", {})
        # fetch_mode: "tick"  = old per-symbol intraday fetch every poll
        #              "minute" = 200-stock optimized: batch OHLC quote every
        #                         cycle (chunked to quote_chunk_size IDs) +
        #                         intraday fetch once per minute, throttled to
        #                         max_intraday_per_cycle symbols/cycle and
        #                         fetched concurrently (intraday_parallel workers)
        self.fetch_mode = paper.get("fetch_mode", "tick")
        self.cycle_seconds = paper.get("cycle_seconds", 1.0)
        self.max_intraday_per_cycle = paper.get("max_intraday_per_cycle", 20)
        self.intraday_parallel = paper.get("intraday_parallel", 8)
        self.quote_chunk_size = paper.get("quote_chunk_size", 100)
        self.poll_seconds_per_stock = paper.get("poll_seconds_per_stock", 1.0)

        app = self.raw.get("app", {})
        self.db_path = Path(__file__).resolve().parent.parent / app.get("db_path", "live/live_trading.db")
        self.history_days = app.get("history_days", 5)
        self.monitor_port = app.get("monitor_port", 8000)

    def validate(self):
        if not self.client_id or not self.access_token:
            raise RuntimeError(
                "Missing Dhan credentials. Set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN "
                "or fill config/live_config.json."
            )
        if not self.symbols:
            raise ValueError("No enabled stocks in config")
        names = [s.name for s in self.strategies]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate strategy names: {names}")
        for s in self.strategies:
            if s.fast not in ("1m", "3m", "5m", "15m", "30m", "1h"):
                raise ValueError(f"[{s.name}] Unsupported fast TF: {s.fast}")
            if s.slow not in ("1m", "3m", "5m", "15m", "30m", "1h"):
                raise ValueError(f"[{s.name}] Unsupported slow TF: {s.slow}")
            if s.capital <= 0:
                raise ValueError(f"[{s.name}] capital must be > 0")
        return self


def load_config(path: Path | str = CONFIG_PATH) -> LiveConfig:
    return LiveConfig(path).validate()