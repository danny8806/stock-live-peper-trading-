#!/usr/bin/env python3
"""Thin wrapper over the DhanHQ API for live paper trading data.

Responsibilities:
  - Authenticate (client_id + access_token).
  - Look up / cache NSE_EQ security IDs for symbols.
  - Fetch 1-minute intraday OHLCV (last 5 trading days).
  - Fetch live OHLC quotes (last price + today's OHLC) for a batch of symbols.

All data is returned as pandas DataFrames/dicts with tz-naive IST wall-clock
timestamps so the rest of the live pipeline (which mirrors the backtest
preprocessing) can consume it directly.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from dhanhq import DhanContext, dhanhq

TZ_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CACHE_FILE = Path(__file__).resolve().parent / "dhan_security_ids.json"


def _epoch_to_ist(ts: float) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.fromtimestamp(ts, tz=TZ_IST)).tz_localize(None)


class DhanClient:
    def __init__(self, client_id: str, access_token: str,
                 exchange_segment: str = "NSE_EQ",
                 instrument_type: str = "EQUITY"):
        self.client_id = client_id
        self.exchange_segment = exchange_segment
        self.instrument_type = instrument_type
        self.last_error: str | None = None
        ctx = DhanContext(client_id, access_token)
        self.client = dhanhq(ctx)

    # ------------------------------------------------------------------
    # Account + market data
    # ------------------------------------------------------------------
    def get_funds(self) -> dict:
        return self.client.get_fund_limits()

    def fetch_ohlc_quotes(self, security_ids: list[int]) -> dict:
        """Live quotes for a batch: {symbol: {"last_price": float, "ohlc": {...}}}"""
        if not security_ids:
            return {}
        r = self.client.ohlc_data({self.exchange_segment: security_ids})
        out = {}
        if r.get("status") == "success":
            data = r.get("data", {})
            # Dhan nests the payload one level deeper: {"data": {"data": {...}}}
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            seg = data.get(self.exchange_segment, {}) if isinstance(data, dict) else {}
            for sid, payload in seg.items():
                out[str(sid)] = payload
        return out

    def fetch_intraday_1m(self, security_id: str | int,
                          from_date: dt.date, to_date: dt.date) -> pd.DataFrame:
        """1-min OHLCV for a symbol between from_date..to_date.

        Returns DataFrame[datetime, open, high, low, close, volume] sorted,
        tz-naive IST. Empty frame on failure.
        """
        r = self.client.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=self.exchange_segment,
            instrument_type=self.instrument_type,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval=1,
        )
        self.last_error = None
        if r.get("status") != "success":
            # transient (rate-limit/network) -> retry briefly
            import time
            for _ in range(2):
                time.sleep(0.5)
                r = self.client.intraday_minute_data(
                    security_id=str(security_id),
                    exchange_segment=self.exchange_segment,
                    instrument_type=self.instrument_type,
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d"),
                    interval=1,
                )
                if r.get("status") == "success":
                    break
            if r.get("status") != "success":
                self.last_error = r.get("remarks")
                return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        data = r.get("data")
        if not isinstance(data, dict) or not data.get("timestamp"):
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame({
            "datetime": [_epoch_to_ist(t) for t in data["timestamp"]],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data.get("volume", [0.0] * len(data["timestamp"])),
        })
        df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
        df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="last")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Security IDs
    # ------------------------------------------------------------------
    def load_security_ids(self, symbols: list[str]) -> dict[str, str]:
        """Return {SYMBOL: security_id}, pulling from a local cache first."""
        cache = self._read_cache()
        missing = [s for s in symbols if s not in cache]
        if missing:
            fresh = self._fetch_security_ids(missing)
            cache.update(fresh)
            self._write_cache(cache)
        return {s: cache[s] for s in symbols if s in cache}

    def _fetch_security_ids(self, symbols: list[str]) -> dict[str, str]:
        try:
            master = self.client.fetch_security_list(
                mode="compact", filename=str(Path.home() / "dhan_security_id_list.csv"))
        except Exception:
            master = None
        out: dict[str, str] = {}
        if master is None or master.empty:
            return out
        master = master.copy()
        master["_SYM"] = master["SEM_TRADING_SYMBOL"].astype(str).str.upper()
        eq = master[master["SEM_EXM_EXCH_ID"].astype(str).str.upper() == "NSE"]
        for sym in symbols:
            hit = eq[eq["_SYM"] == sym]
            if len(hit):
                out[sym] = str(hit.iloc[0]["SEM_SMST_SECURITY_ID"])
        return out

    @staticmethod
    def _read_cache() -> dict[str, str]:
        if CACHE_FILE.is_file():
            try:
                return json.loads(CACHE_FILE.read_text())
            except Exception:
                return {}
        return {}

    @staticmethod
    def _write_cache(data: dict[str, str]) -> None:
        CACHE_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))
