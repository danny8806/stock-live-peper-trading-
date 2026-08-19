#!/usr/bin/env python3
"""Live DEMA-ATR pipeline — streaming version of the backtest preprocessing.

Reuses the exact indicator implementations from
`build_all_timeframes_enriched` (session-aware resample, Pine-faithful
recursive DEMA-ATR band, non-repainting HTF mapping) so live signals match
the backtest exactly.

Only CLOSED 1-min bars feed the indicators (the forming candle never
repaints the signal). With ~5 trading days in the buffer (<= ~1300 bars)
a full recompute per poll is fast; the incremental path simply calls the
same closed set each time.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from build_all_timeframes_enriched import (
    MINUTES,
    SESSION_MINUTES,
    SESSION_OPEN,
    TZ,
    build_all,
)

COLS = ["datetime", "open", "high", "low", "close", "volume"]


def prepare_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Anchor a 1-min frame to trading sessions (09:15 IST) like the backtest."""
    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(TZ, ambiguous="infer",
                                                       nonexistent="shift_forward")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(TZ)
    df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="last")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.reset_index(drop=True)

    session_date = df["datetime"].dt.date.astype(str)
    session_start = pd.to_datetime(
        session_date + f" {SESSION_OPEN}", format="%Y-%m-%d %H:%M"
    ).dt.tz_localize(TZ)
    minutes_from_open = ((df["datetime"] - session_start).dt.total_seconds() // 60).astype(int)
    df = df[(minutes_from_open >= 0) & (minutes_from_open < SESSION_MINUTES)].copy()
    df["_session_start"] = session_start.loc[df.index]
    df["_minutes"] = minutes_from_open.loc[df.index]
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df.reset_index(drop=True)


class DEMAATRPipeline:
    def __init__(self, fast: str = "15m", slow: str = "1h",
                 dema_period: int = 3, atr_period: int = 6,
                 atr_factor: float = 1.0):
        if MINUTES[fast] >= MINUTES[slow]:
            raise ValueError(f"fast ({fast}) must be smaller than slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.dema_period = dema_period
        self.atr_period = atr_period
        self.atr_factor = atr_factor
        self.fast_col = f"demaatr_{fast}"
        self.slow_col = f"demaatr_{slow}"

    @classmethod
    def from_config(cls, cfg) -> "DEMAATRPipeline":
        return cls(fast=cfg.fast, slow=cfg.slow,
                   dema_period=cfg.dema_period, atr_period=cfg.atr_period,
                   atr_factor=cfg.atr_factor)

    def closed_slice(self, buffer_df: pd.DataFrame) -> pd.DataFrame:
        """Only bars strictly older than the current clock-minute."""
        if buffer_df.empty:
            return pd.DataFrame(columns=COLS)
        now = pd.Timestamp(dt.datetime.now().replace(second=0, microsecond=0))
        return buffer_df[pd.to_datetime(buffer_df["datetime"]) < now].copy()

    def compute(self, closed_1m: pd.DataFrame) -> pd.DataFrame:
        """Resample -> DEMA-ATR -> map slow->fast. Returns the fast-TF frame
        with `demaatr_<fast>` and `demaatr_<slow>` columns.

        Only fully-CLOSED fast-TF buckets are returned: a bucket whose window
        (start + tf_minutes) has not yet ended is dropped, so the signal engine
        never sees an incomplete bar (no repainting, exactly like the backtest).
        """
        if closed_1m.empty or len(closed_1m) < 2:
            return pd.DataFrame()
        prepared = prepare_1m(closed_1m)
        # build_all computes every TF with the module-level DEMA_* constants;
        # patch the constants once so the live params flow through.
        import build_all_timeframes_enriched as m
        old = (m.DEMA_PERIOD, m.ATR_PERIOD, m.ATR_FACTOR)
        m.DEMA_PERIOD, m.ATR_PERIOD, m.ATR_FACTOR = (
            self.dema_period, self.atr_period, self.atr_factor)
        try:
            enriched = build_all(prepared)
        finally:
            m.DEMA_PERIOD, m.ATR_PERIOD, m.ATR_FACTOR = old

        fast_df = enriched[self.fast].copy()
        fast_df = fast_df[["datetime", "open", "high", "low", "close", "volume",
                           self.fast_col, self.slow_col]].reset_index(drop=True)
        # drop buckets that have not closed yet (window end still in the future)
        tf_min = MINUTES[self.fast]
        now = pd.Timestamp(dt.datetime.now().replace(second=0, microsecond=0))
        if len(fast_df):
            closed = pd.to_datetime(fast_df["datetime"]) + pd.Timedelta(minutes=tf_min)
            fast_df = fast_df[closed <= now].reset_index(drop=True)
        return fast_df

    def last_closed_row(self, fast_df: pd.DataFrame) -> dict | None:
        """The latest fully-closed fast-TF bar as a dict, or None."""
        if fast_df is None or fast_df.empty:
            return None
        row = fast_df.iloc[-1]
        # drop leading NaN in slow (mapping) so a fresh slow bar doesn't fire
        if pd.isna(row[self.slow_col]):
            return None
        return row.to_dict()
