#!/usr/bin/env python3
"""Streaming signal engine — exact port of `core.bt_strategy.DEMAATRDecisionStrategy.next()`.

State machine per symbol (no backtrader):
    - On a bullish DEMA-ATR crossover (fast above slow) when flat:
      remember the signal candle's HIGH as `pending_high`.
    - On a later bar whose HIGH breaks above `pending_high`: BUY.
    - On a bearish crossover while in a position: SELL.
    - Optional SL/TP % levels are enforced by the paper broker, not here.

`warmup()` replays history silently to initialise pending_high so the first
live bar acts exactly like the backtest would (no historical re-trades).
"""
from __future__ import annotations

import math

import pandas as pd


class DEMAATRSignals:
    def __init__(self, fast_col: str, slow_col: str, min_bars: int = 11):
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.min_bars = min_bars
        self.pending_high: float | None = None
        self.pending_dt = None
        self._last_fast_dt: pd.Timestamp | None = None
        self.warmed_up = False

    @classmethod
    def from_config(cls, cfg, pipeline) -> "DEMAATRSignals":
        return cls(fast_col=pipeline.fast_col, slow_col=pipeline.slow_col,
                   min_bars=cfg.min_bars)

    # ------------------------------------------------------------------
    def _valid(self, fast: pd.Series, slow: pd.Series, i: int) -> bool:
        for series in (fast, slow):
            for k in (i, i - 1):
                if k < 0:
                    return False
                try:
                    if math.isnan(float(series.iloc[k])):
                        return False
                except (IndexError, TypeError, ValueError):
                    return False
        return True

    def _step(self, i: int, fast: pd.Series, slow: pd.Series,
              high: pd.Series, dts: pd.Series, in_position: bool):
        """One bar of the decision machine. Returns an action dict or None."""
        if not self._valid(fast, slow, i):
            return None
        f0, f1 = float(fast.iloc[i]), float(fast.iloc[i - 1])
        s0, s1 = float(slow.iloc[i]), float(slow.iloc[i - 1])
        curr_above = f0 > s0
        prev_above = f1 > s1
        bull_cross = curr_above and not prev_above
        bear_cross = (not curr_above) and prev_above

        if not in_position:
            if bull_cross:
                self.pending_high = float(high.iloc[i])
                self.pending_dt = dts.iloc[i]
            elif self.pending_high is not None and float(high.iloc[i]) > self.pending_high:
                sig = {
                    "type": "BUY",
                    "datetime": dts.iloc[i],
                    "breakout_above": self.pending_high,
                    "bar_high": float(high.iloc[i]),
                    "close": float(high.iloc[i]),
                }
                self.pending_high = None
                self.pending_dt = None
                return sig
        else:
            if bear_cross:
                return {
                    "type": "SELL",
                    "datetime": dts.iloc[i],
                    "reason": "cross",
                    "close": float(high.iloc[i]),
                }
        return None

    # ------------------------------------------------------------------
    def warmup(self, fast_df: pd.DataFrame) -> dict | None:
        """Replay all historical fast-TF bars, updating state only (no actions).

        Returns the last action that would have fired on the FINAL bar, or None
        — lets the engine reconcile a breakout that already fired while the
        engine was offline (a `resume` buy due at the first live price).
        """
        if fast_df is None or len(fast_df) < self.min_bars:
            return None
        fast = fast_df[self.fast_col]
        slow = fast_df[self.slow_col]
        high = fast_df["high"]
        dts = pd.to_datetime(fast_df["datetime"])
        last_action: dict | None = None
        for i in range(self.min_bars - 1, len(fast_df)):
            act = self._step(i, fast, slow, high, dts, in_position=False)
            if act is not None:
                last_action = act
        self._last_fast_dt = dts.iloc[-1]
        self.warmed_up = True
        return last_action

    def next_bar_index(self, fast_df: pd.DataFrame) -> int:
        """Index of the first bar this machine still needs to process (the bar
        after the persisted cursor). Shared by `on_new_bars` and callers that
        drive the machine bar-by-bar (e.g. the live engine tracking position
        state across the batch)."""
        if fast_df is None or fast_df.empty:
            return 0
        dts = pd.to_datetime(fast_df["datetime"])
        start = self.min_bars - 1
        if self._last_fast_dt is not None:
            start = max(start, int(dts.searchsorted(self._last_fast_dt, side="right")))
        elif not self.warmed_up:
            start = len(fast_df) - 1
        return start

    def on_new_bars(self, fast_df: pd.DataFrame, in_position: bool) -> list[dict]:
        """Process fast-TF bars that closed after the last processed bar.

        Returns a list of action dicts ({type: BUY|SELL, ...}) in order.

        `in_position` is the position state at the START of the batch. Callers
        that need per-bar state (a BUY and a later SELL inside the same batch)
        must call `on_new_bars` once per new bar while tracking the flag
        themselves — see `engine.LiveEngine._process_fetched`.
        """
        actions: list[dict] = []
        if fast_df is None or fast_df.empty:
            return actions
        fast = fast_df[self.fast_col]
        slow = fast_df[self.slow_col]
        high = fast_df["high"]
        dts = pd.to_datetime(fast_df["datetime"])

        start = self.next_bar_index(fast_df)
        for i in range(start, len(fast_df)):
            action = self._step(i, fast, slow, high, dts, in_position)
            if action is not None:
                actions.append(action)
        if len(fast_df):
            self._last_fast_dt = dts.iloc[-1]
        return actions

    def reset_pending(self) -> None:
        self.pending_high = None
        self.pending_dt = None
