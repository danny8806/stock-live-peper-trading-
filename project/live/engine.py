#!/usr/bin/env python3
"""Live paper-trading engine — multi-strategy.

Each configured strategy (from `cfg.strategies`) runs on the SAME symbol
universe but with its OWN:
  - DEMA-ATR parameters (fast/slow TF, periods, sl/tp, min_bars)
  - capital, position slots, per-stock cap, slippage
  - positions (keyed symbol+strategy), trades, signals, P&L, equity log

The 1-min candle buffers are shared per symbol (market data is strategy-
agnostic); each strategy resamples to its own timeframes and runs its own
signal engine + paper broker.

Two fetch modes (config `paper.fetch_mode`):
  "tick"   — sequential per-symbol intraday fetch (~1s cadence)
  "minute" — 200-stock optimized: 1 batch OHLC quote per cycle + intraday
             fetch throttled to max_intraday_per_cycle symbols/cycle

Signals NEVER fire on the forming candle — only on fully-closed fast-TF bars,
exactly like the backtest.
"""
from __future__ import annotations

import datetime as dt
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from candle_buffer import CandleBuffer
from config import LiveConfig
from dhan_client import DhanClient
from paper_broker import PaperBroker
from state import StateStore
from strategies import create_broker, create_pipeline, create_signals


class LiveEngine:
    def __init__(self, config: LiveConfig, dhan: DhanClient, state: StateStore):
        self.cfg = config
        self.dhan = dhan
        self.state = state

        self.buffers: dict[str, CandleBuffer] = {}
        self.security_ids: dict[str, str] = {}

        # one pipeline, broker and per-symbol signal engine per strategy
        self.strategy_names: list[str] = [s.name for s in config.strategies]
        self.pipelines: dict[str, object] = {
            s.name: create_pipeline(s) for s in config.strategies}
        self.brokers: dict[str, PaperBroker] = {
            s.name: create_broker(s) for s in config.strategies}
        self.signal_engines: dict[str, dict[str, object]] = {
            s.name: {} for s in config.strategies}

        # backward-compat aliases -> FIRST strategy (single-strategy configs).
        # Properties so they stay live even when setup() rebinds the dicts.
        self._first = self.strategy_names[0]

        self.status = {
            "running": False, "cycle": 0, "started_at": None,
            "last_poll": {},
            "signals_today": {n: 0 for n in self.strategy_names},
            "trades_today": {n: 0 for n in self.strategy_names},
            "fetch_mode": config.fetch_mode,
            "strategies": self.strategy_names,
        }
        self._stop = False
        self._last_minute: dict[str, dt.datetime] = {}
        # (strategy, symbol) pairs whose held position is ALREADY bearish at
        # startup (the exit cross fired while the engine was offline) -> close
        # at the first live price, then resume normal live scanning.
        self._resume_exits: set[tuple[str, str]] = set()
        # (strategy, symbol) pairs flat at startup whose breakout ALREADY fired
        # on the last closed bar while offline -> open at the first live price.
        self._resume_buys: set[tuple[str, str]] = set()
        # last saved processing cursor per (strategy, symbol) -> only write to
        # the checkpoints table when it actually advances.
        self._saved_cp: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    @property
    def pipeline(self) -> object:
        return self.pipelines[self._first]

    @property
    def broker(self) -> PaperBroker:
        return self.brokers[self._first]

    @property
    def signals(self) -> dict[str, object]:
        return self.signal_engines[self._first]

    def setup(self) -> None:
        print(f"[setup] resolving security IDs for {self.cfg.symbols} ...")
        self.security_ids = self.dhan.load_security_ids(self.cfg.symbols)
        missing = [s for s in self.cfg.symbols if s not in self.security_ids]
        if missing:
            print(f"[setup] WARNING: no security ID for {missing}")
        self.security_ids = {k: v for k, v in self.security_ids.items() if k in self.cfg.symbols}
        for sym in self.security_ids:
            self.buffers[sym] = CandleBuffer(sym, self.security_ids[sym])
        for sc in self.cfg.strategies:
            self.signal_engines[sc.name] = {
                sym: create_signals(sc, self.pipelines[sc.name])
                for sym in self.security_ids}
        # restore open positions from the DB so a restart doesn't lose them.
        # Legacy rows written before multi-strategy support carry strategy=""
        # and belong to the first configured strategy.
        for row in self.state.load_positions():
            strat = row["strategy"] or self.strategy_names[0]
            broker = self.brokers.get(strat)
            if broker is None:
                print(f"[setup] WARNING: position for unknown strategy "
                      f"{row['strategy']} skipped ({row['symbol']})")
                continue
            row["strategy"] = strat
            broker.positions[row["symbol"]] = row
            broker.cash -= row["qty"] * row["entry_price"] + row["entry_charges"]
        running = [(s, sym, row) for s, b in self.brokers.items()
                   for sym, row in b.positions.items()]
        for s, sym, row in running:
            print(f"[running][{s}] {sym}: qty={row['qty']} "
                  f"entry={row['entry_price']} {row['entry_dt']} "
                  f"last={row['last_price']}")
        print(f"[setup] {len(self.security_ids)} symbols x "
              f"{len(self.strategy_names)} strategies ready, restored "
              f"{len(running)} running position(s) from DB.")

    # ------------------------------------------------------------------
    def seed_history(self) -> None:
        today = dt.date.today()
        start = today - dt.timedelta(days=self.cfg.history_days)
        print(f"[seed] fetching {self.cfg.history_days}d 1-min history for "
              f"{len(self.security_ids)} symbols (this is slow, one-time) ...")
        for sym, sid in self.security_ids.items():
            df = self.dhan.fetch_intraday_1m(sid, start, today)
            self.buffers[sym].seed(df)
            print(f"  {sym:>10}: {len(df)} bars "
                  f"({df['datetime'].min() if len(df) else 'n/a'} -> "
                  f"{df['datetime'].max() if len(df) else 'n/a'})")
        print("[seed] done.")

    def warmup_signals(self) -> None:
        """Replay history to initialise the per-symbol state machines, then
        reconcile any trade actions that fired while the engine was offline.

        The engine persists a `checkpoints` cursor (last fast-TF candle the
        signal machine consumed) after every poll, so a restart resumes from
        exactly that bar.

          HELD symbols (position in DB, any age):
            resume from the saved cursor, then REPLAY every fast-TF bar that
            closed while offline through the signal machine (in_position=True)
            to catch exits that already fired (e.g. a cross + whipsaw that the
            end-state check alone would miss). Flagged for a `resume` exit at
            the first live price if a due exit is found OR the current DEMA
            state is already bearish.

          FLAT symbols:
            full replay reconstructs pending_high (the buy arm level). If the
            breakout already fired on the last closed bar while offline, the
            BUY is flagged as a due entry (`resume` buy at first live price).
        """
        cps = self.state.load_checkpoints()
        # Resume reconciliation only makes sense when the engine has PRIOR
        # activity (it ran before and was offline for a while). A fresh,
        # first-ever start must begin cleanly: warmup only arms the machines.
        n_pos = len(self.state.load_positions())
        n_trd = self.state.conn.execute(
            "SELECT COUNT(*) n FROM trades").fetchone()["n"]
        prior_activity = bool(cps) or n_pos > 0 or n_trd > 0
        for sc in self.cfg.strategies:
            pipe = self.pipelines[sc.name]
            broker = self.brokers[sc.name]
            for sym in self.security_ids:
                closed = pipe.closed_slice(self.buffers[sym].df)
                fast_df = pipe.compute(closed)
                if fast_df is None or fast_df.empty:
                    continue
                eng = self.signal_engines[sc.name][sym]
                fast = fast_df[pipe.fast_col]
                slow = fast_df[pipe.slow_col]
                high = fast_df["high"]
                dts = pd.to_datetime(fast_df["datetime"])
                last_bar = fast_df.iloc[-1]

                if broker.holding(sym):
                    # resume from the saved cursor; replay bars after it
                    cp = cps.get((sc.name, sym), {}).get("last_fast_dt")
                    cp_ts = pd.Timestamp(cp) if cp else dts.iloc[0]
                    start = max(eng.min_bars - 1,
                                int(dts.searchsorted(cp_ts, side="right")))
                    due = False
                    for i in range(start, len(fast_df)):
                        act = eng._step(i, fast, slow, high, dts, in_position=True)
                        if act is not None and act["type"] == "SELL":
                            due = True
                            break
                    f, s = float(last_bar[pipe.fast_col]), float(last_bar[pipe.slow_col])
                    if not (math.isnan(f) or math.isnan(s)) and f <= s:
                        due = True
                    eng._last_fast_dt = dts.iloc[-1]
                    if due:
                        self._resume_exits.add((sc.name, sym))
                else:
                    # flat: full replay reconstructs the buy-arm level; if the
                    # breakout already fired on the final closed bar while the
                    # engine was offline AND it had prior activity, the entry
                    # is due at resume. A first-ever start begins cleanly.
                    act = eng.warmup(fast_df)
                    if prior_activity and act is not None and act["type"] == "BUY":
                        self._resume_buys.add((sc.name, sym))

                # advance the persisted cursor to the last replayed bar
                self._maybe_save_cp(sc.name, sym, eng._last_fast_dt)
        pend = {n: sum(1 for s in self.signal_engines[n].values()
                       if getattr(s, "pending_high", None) is not None)
                for n in self.strategy_names}
        print(f"[warmup] signal state machines initialised ({pend} pending).")
        if self._resume_exits:
            print(f"[resume] {len(self._resume_exits)} held position(s) due exit "
                  f"(cross fired while offline) -> close at first live price: "
                  f"{sorted(self._resume_exits)}")
        if self._resume_buys:
            print(f"[resume] {len(self._resume_buys)} flat symbol(s) due entry "
                  f"(breakout fired while offline) -> open at first live price: "
                  f"{sorted(self._resume_buys)}")

    def _maybe_save_cp(self, strat: str, sym: str, last_fast_dt,
                       last_1m_dt=None) -> None:
        if last_fast_dt is None and last_1m_dt is None:
            return  # nothing new to persist
        key = (strat, sym)
        cur = self._saved_cp.get(key)
        fv = str(last_fast_dt) if last_fast_dt is not None else None
        mv = str(last_1m_dt) if last_1m_dt is not None else None
        if cur != (fv, mv):
            self.state.save_checkpoint(strat, sym, fv, mv)
            self._saved_cp[key] = (fv, mv)

    # ------------------------------------------------------------------
    def _close_stops(self, strat: str, closed_by_stop: list[dict]) -> None:
        """Persist trades closed by SL/TP for one strategy and clean up."""
        for t in closed_by_stop:
            print(f"[LIVE][{strat}] {t['exit_dt']} SELL {t['symbol']} @ {t['exit_price']} "
                  f"pnl={t['pnl']:.2f} ({t['reason']})")
            self.state.save_trade(t)
            self.state.delete_position(t["symbol"], strat)
            self.status["trades_today"][strat] += 1

    def _poll_symbol(self, sym: str, live_price: float | None = None) -> None:
        """Full live pass for one symbol across ALL strategies: intraday fetch
        -> (per strategy) pipeline -> signals -> execute -> mark-to-market."""
        sid = self.security_ids[sym]
        today = dt.date.today()
        df = self.dhan.fetch_intraday_1m(sid, today, today)
        if df.empty:
            err = f"{dt.datetime.now().strftime('%H:%M:%S')} empty"
            if self.dhan.last_error:
                err += f" (dhan: {self.dhan.last_error})"
            self.status["last_poll"][sym] = err
            return
        self.buffers[sym].merge(df)
        forming = float(df.iloc[-1]["close"])
        last_1m_dt = str(pd.to_datetime(df.iloc[-1]["datetime"]))
        # one consistent price for fills and SL/TP: the batch-quote live price
        # when available (minute mode), else the forming-candle close
        px = live_price if live_price is not None else forming

        for sc in self.cfg.strategies:
            strat = sc.name
            pipe = self.pipelines[strat]
            broker = self.brokers[strat]
            engine = self.signal_engines[strat][sym]

            # newly closed 1-min bars -> recompute pipeline -> signal -> act
            closed = pipe.closed_slice(self.buffers[sym].df)
            fast_df = pipe.compute(closed)
            if fast_df is not None and not fast_df.empty:
                in_pos = broker.holding(sym)
                actions = engine.on_new_bars(fast_df, in_pos)
                for act in actions:
                    if act["type"] == "BUY":
                        self.state.log_signal(sym, "BUY", act, strategy=strat)
                        self.status["signals_today"][strat] += 1
                        pos = broker.buy(sym, px)
                        if pos:
                            print(f"[LIVE][{strat}] {act['datetime']} BUY  {sym} "
                                  f"@ {pos['entry_price']} x {pos['qty']} "
                                  f"(breakout>{act['breakout_above']:.2f})")
                            self.state.upsert_position(pos)
                        else:
                            print(f"[LIVE][{strat}] {act['datetime']} BUY  {sym} "
                                  f"REJECTED (no cash / slot full)")
                    elif act["type"] == "SELL":
                        self.state.log_signal(sym, "SELL", act, strategy=strat)
                        t = broker.sell(sym, px, reason=act.get("reason", "cross"))
                        if t:
                            print(f"[LIVE][{strat}] {act['datetime']} SELL {sym} "
                                  f"@ {t['exit_price']} pnl={t['pnl']:.2f} ({t['reason']})")
                            self.state.save_trade(t)
                            self.state.delete_position(sym, strat)
                            self.status["trades_today"][strat] += 1

            # RESUME reconciliation (one-shot, restart only): trade actions that
            # fired while the engine was offline are executed at the first live
            # price, then normal live scanning continues untouched.
            if (strat, sym) in self._resume_buys:
                self._resume_buys.discard((strat, sym))
                if not broker.holding(sym):
                    pos = broker.buy(sym, px)
                    if pos:
                        print(f"[RESUME][{strat}] BUY  {sym} @ {pos['entry_price']} "
                              f"x {pos['qty']} (entry due while offline)")
                        self.state.upsert_position(pos)
                        self.status["signals_today"][strat] += 1
            if (strat, sym) in self._resume_exits:
                self._resume_exits.discard((strat, sym))
                if broker.holding(sym) and fast_df is not None and not fast_df.empty:
                    last = fast_df.iloc[-1]
                    f, s = float(last[pipe.fast_col]), float(last[pipe.slow_col])
                    if not (math.isnan(f) or math.isnan(s)) and f <= s:
                        t = broker.sell(sym, px, reason="resume")
                        if t:
                            print(f"[RESUME][{strat}] SELL {sym} @ {t['exit_price']} "
                                  f"pnl={t['pnl']:.2f} (exit due while offline)")
                            self.state.save_trade(t)
                            self.state.delete_position(sym, strat)
                            self.status["trades_today"][strat] += 1

            # mark-to-market + SL/TP on every poll using the live price
            self._close_stops(strat, broker.mark_to_market({sym: px}))

            # persist the processing cursor so a restart resumes from here
            self._maybe_save_cp(strat, sym, engine._last_fast_dt, last_1m_dt)

        self.status["last_poll"][sym] = (
            f"{dt.datetime.now().strftime('%H:%M:%S')} close={forming:.2f} "
            f"bars={len(self.buffers[sym])}")

    # ------------------------------------------------------------------
    def _tick(self, sym: str, price: float) -> None:
        """Pure tick: SL/TP mark-to-market for one symbol across ALL strategies."""
        for sc in self.cfg.strategies:
            strat = sc.name
            self._close_stops(strat, self.brokers[strat].mark_to_market({sym: price}))
        self.status["last_poll"][sym] = (
            f"{dt.datetime.now().strftime('%H:%M:%S')} tick px={price:.2f}")

    @staticmethod
    def _quote_price(quotes: dict, symbol_id: str) -> float | None:
        q = quotes.get(symbol_id)
        if q and q.get("last_price") is not None:
            return float(q["last_price"])
        return None

    # ------------------------------------------------------------------
    def _run_sequential(self, iterations: int | None) -> None:
        """Original mode: intraday fetch + full pass per symbol, ~1s each."""
        cycle = 0
        while not self._stop:
            cycle += 1
            self.status["cycle"] = cycle
            for sym in list(self.security_ids):
                if self._stop:
                    break
                t0 = time.perf_counter()
                try:
                    self._poll_symbol(sym)
                except Exception as e:  # keep the loop alive
                    print(f"[ERROR] {sym}: {e}")
                self.status["last_poll"]["_ts"] = dt.datetime.now().strftime("%H:%M:%S")
                for strat in self.strategy_names:
                    self.state.sync_broker_positions(strat, self.brokers[strat])
                sleep = self.cfg.poll_seconds_per_stock - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
                if iterations is not None and cycle >= iterations:
                    self._stop = True
                    break
            self._log_cycle()

    def _run_optimized(self, iterations: int | None) -> None:
        """200-stock mode: one batch quote per cycle + intraday fetch spread
        across cycles (max_intraday_per_cycle per cycle)."""
        self._last_minute = {s: None for s in self.security_ids}
        all_ids = [int(v) for v in self.security_ids.values()]
        cycle = 0
        while not self._stop:
            cycle += 1
            self.status["cycle"] = cycle
            t0 = time.perf_counter()

            # 1) one batch OHLC quote request for ALL symbols
            quotes: dict = {}
            try:
                quotes = self.dhan.fetch_ohlc_quotes(all_ids)
            except Exception as e:
                print(f"[ERROR] batch quotes: {e}")

            now_min = dt.datetime.now().replace(second=0, microsecond=0)

            # 2) symbols whose minute changed since their last intraday fetch
            #    (throttled so 200 symbols are refreshed within ~a minute)
            todo = [s for s in self.security_ids if self._last_minute[s] != now_min]
            todo = todo[: self.cfg.max_intraday_per_cycle]
            for s in todo:
                if self._stop:
                    break
                try:
                    self._poll_symbol(s, live_price=self._quote_price(quotes, str(self.security_ids[s])))
                    self._last_minute[s] = now_min
                except Exception as e:  # keep the loop alive
                    print(f"[ERROR] {s}: {e}")
                for strat in self.strategy_names:
                    self.state.sync_broker_positions(strat, self.brokers[strat])

            # 3) pure SL/TP tick for every other symbol using the quote price
            for s in self.security_ids:
                if s in todo:
                    continue
                px = self._quote_price(quotes, str(self.security_ids[s]))
                if px is None:
                    continue
                try:
                    self._tick(s, px)
                except Exception as e:
                    print(f"[ERROR] tick {s}: {e}")

            self.status["last_poll"]["_ts"] = dt.datetime.now().strftime("%H:%M:%S")
            self._log_cycle()
            if iterations is not None and cycle >= iterations:
                self._stop = True
            sleep = self.cfg.cycle_seconds - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _log_cycle(self) -> None:
        for sc in self.cfg.strategies:
            strat = sc.name
            broker = self.brokers[strat]
            self.state.log_equity(broker.equity, broker.cash,
                                  broker.open_positions(),
                                  unrealized=broker.unrealized_pnl(),
                                  invested=broker.invested_value(),
                                  strategy=strat)
        self.status["equity"] = {n: round(self.brokers[n].equity, 2)
                                 for n in self.strategy_names}
        self.status["portfolio"] = {n: self.brokers[n].portfolio_pnl()
                                    for n in self.strategy_names}

    # ------------------------------------------------------------------
    def run(self, iterations: int | None = None) -> None:
        self.setup()
        self.seed_history()
        self.warmup_signals()
        self.status["running"] = True
        self.status["started_at"] = dt.datetime.now().isoformat(timespec="seconds")

        print(f"\n[run] STARTING live paper trading "
              f"(strategies={self.strategy_names}, "
              f"capital={ {n: round(self.brokers[n].capital, 0) for n in self.strategy_names} }, "
              f"fetch_mode={self.cfg.fetch_mode})")
        for sc in self.cfg.strategies:
            print(f"  - {sc.name}: {sc.fast}/{sc.slow} DEMATR({sc.dema_period},{sc.atr_period},"
                  f"x{sc.atr_factor}) sl={sc.sl_pct}% tp={sc.tp_pct}% "
                  f"cap=Rs{sc.capital:,.0f} max_pos={sc.max_positions}")
        if self.cfg.fetch_mode == "minute":
            print(f"[run] batch quote every {self.cfg.cycle_seconds:.1f}s + intraday "
                  f"fetch throttled to {self.cfg.max_intraday_per_cycle} symbols/cycle "
                  f"(200-stock mode)")
        else:
            print(f"[run] sequential poll: {self.cfg.poll_seconds_per_stock:.1f}s/symbol, "
                  f"{len(self.security_ids)} symbols -> "
                  f"{self.cfg.poll_seconds_per_stock * len(self.security_ids):.0f}s per cycle")
        print("Press Ctrl+C to stop.\n")

        try:
            if self.cfg.fetch_mode == "minute":
                self._run_optimized(iterations)
            else:
                self._run_sequential(iterations)
        except KeyboardInterrupt:
            print("\n[run] interrupted.")
        finally:
            self.status["running"] = False
            for strat in self.strategy_names:
                self.state.sync_broker_positions(strat, self.brokers[strat])
            print(f"\n[run] stopped. "
                  f"{ {n: self.state.summary(n) for n in self.strategy_names} }")
            self.state.close()


def main() -> None:
    from config import load_config
    cfg = load_config()
    dhan = DhanClient(cfg.client_id, cfg.access_token,
                      cfg.exchange_segment, cfg.instrument_type)
    state = StateStore(cfg.db_path)
    engine = LiveEngine(cfg, dhan, state)
    engine.run()


if __name__ == "__main__":
    main()