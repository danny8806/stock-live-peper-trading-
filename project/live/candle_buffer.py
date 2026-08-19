#!/usr/bin/env python3
"""Per-symbol 1-minute candle store for live trading.

The Dhan intraday endpoint returns closed 1-min bars PLUS the current
forming bar. The buffer keeps the full recent 1-min series for each symbol
and exposes:

  - merge(one_min_df):        upsert new bars, return newly *closed* bars
  - forming_price:            live price of the current forming candle
  - df:                       full sorted 1-min frame (tz-naive IST)

Closed-bar detection: a bar whose minute is strictly older than the current
clock-minute is "closed"; the bar stamped exactly at the current minute is
the forming candle whose close tracks the live price.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

COLS = ["datetime", "open", "high", "low", "close", "volume"]


class CandleBuffer:
    def __init__(self, symbol: str, security_id: str | int):
        self.symbol = symbol
        self.security_id = security_id
        self.df = pd.DataFrame(columns=COLS)

    @property
    def forming_price(self) -> float | None:
        """Live price of the current forming 1-min candle (last row)."""
        if self.df.empty:
            return None
        return float(self.df.iloc[-1]["close"])

    def seed(self, one_min_df: pd.DataFrame) -> None:
        """Replace the buffer with a full history frame."""
        if one_min_df.empty:
            return
        df = one_min_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="last")
        df = df[COLS].reset_index(drop=True)
        self.df = df

    def merge(self, one_min_df: pd.DataFrame) -> pd.DataFrame:
        """Upsert fetched bars into the buffer.

        Returns the rows that are NEW since the last merge (used to detect
        newly closed 1-min bars).
        """
        if one_min_df.empty:
            return pd.DataFrame(columns=COLS)
        new = one_min_df.copy()
        new["datetime"] = pd.to_datetime(new["datetime"])
        new = new[COLS]

        if self.df.empty:
            self.df = new.sort_values("datetime").drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)
            return self.df.copy()

        old_keys = set(pd.to_datetime(self.df["datetime"]))
        incoming = new[~new["datetime"].isin(old_keys)]
        if incoming.empty:
            return incoming

        combined = pd.concat([self.df, new], ignore_index=True)
        combined = combined.sort_values("datetime").drop_duplicates(subset="datetime", keep="last")
        self.df = combined.reset_index(drop=True)
        return incoming.reset_index(drop=True)

    def closed_bars_since(self, last_closed: pd.Timestamp | None) -> pd.DataFrame:
        """All closed bars strictly older than the current clock-minute."""
        if self.df.empty:
            return pd.DataFrame(columns=COLS)
        now = pd.Timestamp(dt.datetime.now().replace(second=0, microsecond=0))
        closed = self.df[pd.to_datetime(self.df["datetime"]) < now]
        if last_closed is not None:
            closed = closed[pd.to_datetime(closed["datetime"]) > last_closed]
        return closed

    def __len__(self) -> int:
        return len(self.df)
