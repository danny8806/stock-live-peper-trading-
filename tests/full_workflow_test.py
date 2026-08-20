#!/usr/bin/env python3
"""Full-workflow integration test: exercises EVERY module in the repo from
start to end of the production pipeline.

Segments (in workflow order):
  S1 Data prep   : utils/resample_to_1min, build_all_timeframes_enriched
                   (preprocess_1min/resample/pine_ema/wilder_atr/dema_atr/
                   map_htf_to_target/build_all), build_liquid_cache
  S2 Backtest    : core/equity_charges, core/vbt_engine (generate_signals,
                   simulate_trades, run_vectorbt), core/bt_feed,
                   core/bt_strategy + core/backtrader_backtest (run_backtest_full),
                   core/report, backtest_all_stocks_vectorbt (import)
  S3 Live        : live/config, live/dhan_client (construct), live/candle_buffer,
                   live/pipeline, live/signals, live/paper_broker,
                   live/strategies, live/state, live/engine (full run() lifecycle
                   + restart resume), live/monitor (all endpoints)
Synthetic 1-min OHLCV data only — no network, no wall-clock dependence
(FixedDatetime shim for pipeline.dt / candle_buffer.dt).
"""
from __future__ import annotations

import datetime
import json
import math
import os
import pathlib
import sys
import tempfile
import types
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT / "project" / "live"))

import numpy as np
import pandas as pd

# ===========================================================================
# Synthetic data + deterministic clock
# ===========================================================================
BASE = pd.Timestamp("2026-08-17 09:15")
SESSION = 450  # 09:15 -> 15:30 IST in minutes


def ts_of(i: int) -> pd.Timestamp:
    return BASE + pd.Timedelta(days=i // SESSION) + pd.Timedelta(minutes=i % SESSION)


def px_series(i: int) -> float:
    """Oscillating series with drift -> DEMA-ATR crosses + breakouts occur."""
    return round(100.0 + i * 0.015 + math.sin(i / 25.0) * 6.0
                 + math.sin(i / 80.0) * 10.0, 2)


def make_bars(n: int = 4 * SESSION, seed=px_series) -> pd.DataFrame:
    rows = []
    for i in range(n):
        p = seed(i)
        rows.append({
            "datetime": ts_of(i),
            "open": round(p * (1 - 0.0005), 2),
            "high": round(p * 1.004, 2),
            "low": round(p * 0.996, 2),
            "close": p,
            "volume": 1000 + (i % 500),
        })
    return pd.DataFrame(rows)


class FixedDatetime(datetime.datetime):
    _now: datetime.datetime | None = None

    @classmethod
    def now(cls, tz=None):
        if cls._now is None:
            return super().now(tz)
        return cls._now


def install_clock(now: pd.Timestamp) -> None:
    FixedDatetime._now = now.to_pydatetime()
    for modname in ("pipeline", "candle_buffer"):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        shim = types.ModuleType(f"{modname}.dt")
        shim.datetime = FixedDatetime
        mod.dt = shim


def restore_clock() -> None:
    FixedDatetime._now = None


class FakeDhan:
    """Minimal DhanClient interface: security IDs + 1-min history + quotes."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.sid = {s: str(50000 + k) for k, s in enumerate(frames)}

    def load_security_ids(self, symbols):
        return {s: self.sid[s] for s in symbols if s in self.sid}

    def fetch_intraday_1m(self, security_id, from_date, to_date):
        sym = next(s for s, i in self.sid.items() if i == str(security_id))
        return self.frames[sym].copy()

    def fetch_ohlc_quotes(self, security_ids):
        return {str(i): {"last_price": 101.0} for i in security_ids}


def temp_live_config(db_path: str, symbols: list[str]) -> dict:
    return {
        "dhan": {"client_id": "1102461741", "access_token": "FAKE_TOKEN_FOR_TEST",
                 "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY"},
        "stocks": {s: {"capital": 100000, "lot_size": 1, "enabled": True}
                   for s in symbols},
        "strategies": [
            {"name": "dema_15m_1h", "fast": "15m", "slow": "1h", "dema_period": 3,
             "atr_period": 6, "atr_factor": 1.0, "min_bars": 11, "sl_pct": 0.0,
             "tp_pct": 0.0, "capital": 200000.0, "max_positions": 5,
             "max_capital_per_stock": 100000.0, "slippage_pct": 0.0005},
            {"name": "dema_5m_15m", "fast": "5m", "slow": "15m", "dema_period": 3,
             "atr_period": 6, "atr_factor": 1.0, "min_bars": 11, "sl_pct": 0.0,
             "tp_pct": 0.0, "capital": 100000.0, "max_positions": 3,
             "max_capital_per_stock": 100000.0, "slippage_pct": 0.0005},
        ],
        "paper": {"fetch_mode": "minute", "cycle_seconds": 0.0,
                  "max_intraday_per_cycle": 20, "intraday_parallel": 4,
                  "quote_chunk_size": 100, "poll_seconds_per_stock": 0.0},
        "app": {"db_path": db_path, "history_days": 5, "monitor_port": 8000},
    }


# ===========================================================================
# Runner
# ===========================================================================
results: list[tuple[str, str, str]] = []


def section(name: str):
    def deco(fn):
        try:
            fn()
            results.append((name, "PASS", ""))
            print(f"[ PASS] {name}")
        except Exception:
            results.append((name, "FAIL", traceback.format_exc().strip().splitlines()[-1]))
            print(f"[ FAIL] {name}\n{traceback.format_exc()}")
        return fn
    return deco


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="workflow_"))
    frames = {"KAYNES": make_bars(), "HFCL": make_bars(n=3 * SESSION + 120)}

    # ---------------------------------------------------------- S1 data prep
    @section("S1.1 utils/resample_to_1min (load_data + resample_to_1min)")
    def s1_1():
        from utils.resample_to_1min import load_data, resample_to_1min
        # 15m synthetic bars -> 1-min
        df15 = make_bars(n=4 * 30)
        df15 = df15.iloc[::15].reset_index(drop=True)
        df15.to_csv(tmp / "src_15m.csv", index=False)
        hf = load_data(tmp / "src_15m.csv")
        assert hf["datetime"].dt.tz is not None
        one = resample_to_1min(hf)
        assert list(one.columns) == ["datetime", "open", "high", "low", "close", "volume"]
        assert one["datetime"].dt.tz is None
        assert len(one) > len(hf)

    @section("S1.2 build_all_timeframes_enriched (preprocess/build_all)")
    def s1_2():
        import build_all_timeframes_enriched as m
        bars = make_bars()
        bars.to_csv(tmp / "KAYNES_1m.csv", index=False)
        df = m.preprocess_1min(tmp / "KAYNES_1m.csv")
        assert len(df) > 100
        enriched = m.build_all(df)
        assert set(m.TIMEFRAMES) <= set(enriched)
        for tf in ("1m", "5m", "15m", "1h"):
            f = enriched[tf]
            assert {"open", "high", "low", "close", "volume"} <= set(f.columns)
            assert f"demaatr_{tf}" in f.columns
            assert not f[f"demaatr_{tf}"].iloc[30:].isna().all()

    @section("S1.3 build_liquid_cache (compute_liquidity)")
    def s1_3():
        from build_liquid_cache import compute_liquidity
        bars = make_bars()
        bars.to_csv(tmp / "HFCL_1m.csv", index=False)
        liq = compute_liquidity(str(tmp / "HFCL_1m.csv"))
        assert liq["days"] == 4 and liq["avg_daily_volume"] > 0
        assert liq["symbol"] == "HFCL_1m"

    # ---------------------------------------------------------- S2 backtest
    @section("S2.1 core/equity_charges (calculate + round trip)")
    def s2_1():
        from core.equity_charges import EquityChargesEngine
        e = EquityChargesEngine()
        c = e.calculate(100.0, 100, is_buy=True)
        assert c.total_charges > 0
        assert c.brokerage > 0 and c.exchange_charges > 0 and c.stamp_duty > 0
        assert c.sebi_fees > 0
        rt = e.calculate_round_trip(100.0, 110.0, 100)
        assert rt["total"]["total_charges"] > 0
        assert rt["buy"]["stt"] == 0 and rt["sell"]["stt"] > 0

    @section("S2.2 core/vbt_engine (signals + simulate_trades + vectorbt)")
    def s2_2():
        import build_all_timeframes_enriched as m
        from core import vbt_engine
        df = m.preprocess_1min(tmp / "KAYNES_1m.csv")
        enriched = m.build_all(df)
        frame = enriched["5m"].copy()
        sig = vbt_engine.generate_signals(frame, "demaatr_5m", "demaatr_15m",
                                          min_bars=11)
        assert {"entries", "exits"} <= set(sig.columns)
        trades, cash = vbt_engine.simulate_trades(frame, "demaatr_5m",
                                                  "demaatr_15m", min_bars=11)
        assert cash <= vbt_engine.CAPITAL * 1.5
        for t in trades:
            assert t["entry_date"] <= t["exit_date"]
            assert t["size"] > 0
        pf = vbt_engine.run_vectorbt(sig)
        assert pf.final_value() > 0

    @section("S2.3 core/backtrader_backtest + bt_strategy + bt_feed + report")
    def s2_3():
        import build_all_timeframes_enriched as m
        from core.backtrader_backtest import run_backtest_full
        df = m.preprocess_1min(tmp / "KAYNES_1m.csv")
        enriched = m.build_all(df)
        stats, strat = run_backtest_full(enriched, "5m", "15m", verbose=False)
        assert stats is not None and strat is not None
        assert strat.trades is not None  # in-memory trade list collected
        # report: patch pyplot.show to a no-op so nothing blocks/opens windows
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
        from core.report import show_report
        show_report(strat, "5m", "15m")

    @section("S2.4 backtest_all_stocks_vectorbt imports (module fixed)")
    def s2_4():
        import importlib
        mod = importlib.import_module("backtest_all_stocks_vectorbt")
        assert mod.CAPITAL is not None and len(mod.COMBOS) == 15
        from core.vbt_engine import simulate_trades
        assert callable(simulate_trades)

    # ---------------------------------------------------------- S3 live
    @section("S3.1 live/config (load + validate + env override)")
    def s3_1():
        from config import LiveConfig, load_config
        cfg_json = tmp / "live_config.json"
        cfg_json.write_text(json.dumps(temp_live_config("live/test.db", ["KAYNES"])),
                            encoding="utf-8")
        cfg = load_config(cfg_json)
        assert cfg.strategies[0].name == "dema_15m_1h"
        assert cfg.fetch_mode == "minute"
        assert cfg.symbols == ["KAYNES"]
        try:
            cfg.validate()
        except Exception:
            raise AssertionError("valid config must validate")
        # env override for token
        os.environ["DHAN_ACCESS_TOKEN"] = "ENV_TOKEN"
        cfg2 = LiveConfig(cfg_json)
        assert cfg2.access_token == "ENV_TOKEN"
        del os.environ["DHAN_ACCESS_TOKEN"]

    @section("S3.2 live/dhan_client constructs + candle_buffer ops")
    def s3_2():
        from dhan_client import DhanClient
        c = DhanClient("1102461741", "FAKE", "NSE_EQ", "EQUITY")
        assert c.client_id == "1102461741"
        from candle_buffer import CandleBuffer
        b = CandleBuffer("KAYNES", 123)
        df = frames["KAYNES"]
        b.seed(df.iloc[:500])
        assert len(b) == 500
        incoming = b.merge(df.iloc[500:600])
        assert len(incoming) == 100 and len(b) == 600
        closed = b.closed_bars_since(None)
        assert len(closed) > 0
        assert b.forming_price is not None

    @section("S3.3 live/pipeline (compute + closed_slice + last_closed_row)")
    def s3_3():
        from pipeline import DEMAATRPipeline
        pipe = DEMAATRPipeline("5m", "15m", 3, 6, 1.0)
        assert pipe.fast_col == "demaatr_5m" and pipe.slow_col == "demaatr_15m"
        install_clock(pd.Timestamp("2026-08-20 15:30:00"))
        try:
            closed = pipe.closed_slice(frames["KAYNES"])
            assert closed["datetime"].max() < pd.Timestamp("2026-08-20 15:30:00")
            fast_df = pipe.compute(closed)
            assert not fast_df.empty and pipe.slow_col in fast_df.columns
            row = pipe.last_closed_row(fast_df)
            assert row is None or isinstance(row, dict)
        finally:
            restore_clock()

    @section("S3.4 live/signals (warmup + per-bar on_new_bars)")
    def s3_4():
        from pipeline import DEMAATRPipeline
        from signals import DEMAATRSignals
        pipe = DEMAATRPipeline("5m", "15m", 3, 6, 1.0)
        install_clock(pd.Timestamp("2026-08-20 15:30:00"))
        try:
            fast_df = pipe.compute(pipe.closed_slice(frames["KAYNES"]))
            eng = DEMAATRSignals(pipe.fast_col, pipe.slow_col, min_bars=11)
            last = eng.warmup(fast_df)
            assert eng.warmed_up and eng.next_bar_index(fast_df) >= len(fast_df)
            seq, in_pos = [], False
            for i in range(eng.next_bar_index(fast_df), len(fast_df)):
                for a in eng.on_new_bars(fast_df.iloc[:i + 1], in_pos):
                    seq.append(a["type"])
                    in_pos = a["type"] == "BUY"
            assert isinstance(seq, list)
        finally:
            restore_clock()

    @section("S3.5 live/paper_broker (buy/sell/SL/TP/equity)")
    def s3_5():
        from paper_broker import PaperBroker
        b = PaperBroker(capital=100000.0, max_positions=5,
                        max_capital_per_stock=50000.0, slippage_pct=0.0005,
                        sl_pct=0.0, tp_pct=0.0, strategy="dema_5m_15m")
        pos = b.buy("KAYNES", 100.0, quote_dt=datetime.datetime(2026, 8, 20, 10, 0))
        assert pos and b.holding("KAYNES") and b.equity > 0
        t = b.sell("KAYNES", 105.0, quote_dt=datetime.datetime(2026, 8, 20, 11, 0),
                   reason="cross")
        assert t and not b.holding("KAYNES") and t["pnl"] > 0
        # SL/TP path
        b2 = PaperBroker(capital=100000.0, max_positions=5,
                         max_capital_per_stock=50000.0, slippage_pct=0.0,
                         sl_pct=5.0, tp_pct=0.0, strategy="t")
        b2.buy("HFCL", 100.0, quote_dt=datetime.datetime(2026, 8, 20, 10, 0))
        stops = b2.mark_to_market({"HFCL": 94.0})
        assert len(stops) == 1 and stops[0]["reason"] == "SL"

    @section("S3.6 live/strategies (registry factories)")
    def s3_6():
        from strategies import create_broker, create_pipeline, create_signals
        from config import StrategyConfig
        sc = StrategyConfig({"name": "dema_5m_15m", "fast": "5m", "slow": "15m",
                             "capital": 100000.0})
        pipe = create_pipeline(sc)
        sig = create_signals(sc, pipe)
        broker = create_broker(sc)
        assert pipe.fast == "5m" and broker.strategy == "dema_5m_15m"
        assert sig is not None

    @section("S3.7 live/state (full CRUD + summary + checkpoint + sync)")
    def s3_7():
        from state import StateStore
        db = tmp / "state.db"
        st = StateStore(db)
        st.upsert_position({"strategy": "dema_5m_15m", "symbol": "KAYNES",
                            "qty": 10, "entry_price": 100.0, "entry_charges": 2.0,
                            "entry_dt": "2026-08-19T10:00:00", "sl_level": None,
                            "tp_level": None, "last_price": 100.0})
        st.log_signal("KAYNES", "BUY", {"breakout_above": 99.0},
                      strategy="dema_5m_15m")
        st.log_equity(100000.0, 99000.0, 1, strategy="dema_5m_15m")
        st.save_checkpoint("dema_5m_15m", "KAYNES", "2026-08-19 10:05:00",
                           "2026-08-19 10:05:00")
        pos = st.load_positions()
        assert len(pos) == 1 and pos[0]["symbol"] == "KAYNES"
        assert st.summary("dema_5m_15m")["open_positions"] == 1
        assert st.portfolio_summary(100000.0, "dema_5m_15m")["equity"] >= 0
        t = {"strategy": "dema_5m_15m", "symbol": "KAYNES", "qty": 10,
             "entry_price": 100.0, "exit_price": 105.0,
             "entry_dt": "2026-08-19T10:00:00", "exit_dt": "2026-08-19T10:10:00",
             "pnl": 50.0, "charges": 5.0, "reason": "cross"}
        assert st.close_position_trade(t, "dema_5m_15m") is True
        assert st.close_position_trade(t, "dema_5m_15m") is False  # idempotent
        assert len(st.load_positions()) == 0
        cps = st.load_checkpoints()
        assert ("dema_5m_15m", "KAYNES") in cps
        st.close()

    @section("S3.8 live/engine full run() lifecycle + restart resume")
    def s3_8():
        from config import load_config
        from engine import LiveEngine
        from state import StateStore
        cfg_json = tmp / "engine_config.json"
        cfg_json.write_text(json.dumps(temp_live_config(
            str(tmp / "engine_live.db"), ["KAYNES", "HFCL"])), encoding="utf-8")
        cfg = load_config(cfg_json)
        install_clock(pd.Timestamp("2026-08-20 15:30:00"))
        try:
            st = StateStore(cfg.db_path)
            dhan = FakeDhan(frames)
            eng = LiveEngine(cfg, dhan, st)
            eng.run(iterations=2)          # run() closes the state store on exit
            assert eng.status["cycle"] == 2
            assert eng.status["running"] is False  # stopped after iterations
            assert set(eng.security_ids) == {"KAYNES", "HFCL"}
            assert len(eng.buffers["KAYNES"]) == len(frames["KAYNES"])
            for strat in eng.strategy_names:
                for sym in eng.security_ids:
                    s = eng.signal_engines[strat][sym]
                    assert s.warmed_up and s._last_fast_dt is not None
            # run() closed the DB -> reopen to inspect persisted state
            st_v = StateStore(cfg.db_path)
            cps = st_v.load_checkpoints()
            assert any(k[0] in eng.strategy_names for k in cps), f"no checkpoints: {cps}"
            eq = st_v.conn.execute("SELECT COUNT(*) n FROM equity").fetchone()["n"]
            assert eq >= 2, f"expected >=2 equity rows, got {eq}"
            # restart resume: a second engine instance from the same DB
            eng2 = LiveEngine(cfg, dhan, st_v)
            eng2.setup()
            eng2.seed_history()
            eng2.warmup_signals()
            eng2._stop = True
            st_v.close()
        finally:
            restore_clock()

    @section("S3.9 live/monitor (all endpoints)")
    def s3_9():
        import config as config_mod
        from state import StateStore
        cfg_json = tmp / "monitor_config.json"
        cfg_json.write_text(json.dumps(temp_live_config(
            str(tmp / "monitor.db"), ["KAYNES"])), encoding="utf-8")
        mcfg = config_mod.load_config(cfg_json)
        config_mod.load_config = lambda *a, **k: mcfg
        import monitor
        monitor.state = StateStore(mcfg.db_path)
        # seed some data for the endpoints
        st = monitor.state
        st.upsert_position({"strategy": "dema_5m_15m", "symbol": "KAYNES",
                            "qty": 10, "entry_price": 100.0, "entry_charges": 2.0,
                            "entry_dt": "2026-08-19T10:00:00", "sl_level": None,
                            "tp_level": None, "last_price": 100.0})
        st.save_trade({"strategy": "dema_5m_15m", "symbol": "KAYNES", "qty": 10,
                       "entry_price": 100.0, "exit_price": 105.0,
                       "entry_dt": "2026-08-19T10:00:00",
                       "exit_dt": "2026-08-19T10:10:00", "pnl": 50.0,
                       "charges": 5.0, "reason": "cross"})
        st.log_signal("KAYNES", "BUY", {"x": 1}, strategy="dema_5m_15m")
        for path in ("/status", "/portfolio", "/trades", "/positions", "/signals"):
            handler = {r.path.lstrip("/") for r in monitor.app.routes}
        assert {"/status", "/portfolio", "/trades", "/positions", "/signals"} <= \
            {r.path for r in monitor.app.routes}
        s = monitor.status()
        assert s["config"]["strategies"][0]["name"] == "dema_15m_1h"
        assert monitor.portfolio()["equity"] >= 0
        assert len(monitor.trades()) == 1
        assert len(monitor.positions()) == 1
        assert len(monitor.signals()) == 1
        assert len(monitor.trades(strategy="dema_5m_15m")) == 1
        monitor.state.close()

    # ---------------------------------------------------------- summary
    npass = sum(1 for _, s, _ in results if s == "PASS")
    print("\n================ FULL-WORKFLOW INTEGRATION TEST ================")
    print(f"{len(results)} segments -> PASS={npass} FAIL={len(results) - npass}\n")
    for name, status, detail in results:
        if status != "PASS":
            print(f"[{status}] {name}\n   {detail}")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
