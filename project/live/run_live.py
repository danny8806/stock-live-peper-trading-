#!/usr/bin/env python3
"""run_live.py — CLI entry point for the live Dhan paper-trading engine.

Commands:
    python run_live.py test        One-shot: fetch data for all symbols and
                                   run ONE full poll cycle through the whole
                                   pipeline (data architecture test).
    python run_live.py live        Continuous sequential paper-trading loop.
    python run_live.py live --iter 5   Run 5 poll cycles then stop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from dhan_client import DhanClient
from engine import LiveEngine
from state import StateStore


def run_test() -> None:
    cfg = load_config()
    print("=" * 70)
    print("LIVE DATA ARCHITECTURE TEST  (Dhan API)")
    print("=" * 70)
    print(f"Symbols     : {cfg.symbols}")
    for sc in cfg.strategies:
        print(f"Strategy    : {sc.name}  {sc.fast}/{sc.slow}  "
              f"DEMA({sc.dema_period})-ATR({sc.atr_period},x{sc.atr_factor})  "
              f"capital=Rs{sc.capital:,.0f} max_pos={sc.max_positions}")

    dhan = DhanClient(cfg.client_id, cfg.access_token,
                      cfg.exchange_segment, cfg.instrument_type)

    print("\n[1] Account check ...")
    funds = dhan.get_funds()
    print("   status:", funds.get("status"))
    if funds.get("status") == "success":
        d = funds["data"]
        print(f"   balance: Rs{d.get('availabelBalance'):,.2f}")

    print("[2] Security ID lookup ...")
    ids = dhan.load_security_ids(cfg.symbols)
    for sym, sid in ids.items():
        print(f"   {sym:>10} -> {sid}")
    missing = [s for s in cfg.symbols if s not in ids]
    if missing:
        print(f"   WARNING missing IDs: {missing}")

    print("[3] Seed 1-min history (last {cfg.history_days}d) ...".format(cfg=cfg))
    state = StateStore(cfg.db_path)
    engine = LiveEngine(cfg, dhan, state)
    engine.setup()
    engine.seed_history()

    print("\n[4] Warmup signal state machines ...")
    engine.warmup_signals()
    for strat in engine.strategy_names:
        for sym in engine.security_ids:
            s = engine.signal_engines[strat][sym]
            pend = f"{s.pending_high:.2f}" if s.pending_high is not None else "None"
            print(f"   [{strat}] {sym:>10}: pending_high={pend}, "
                  f"last_fast_dt={s._last_fast_dt}")

    print("\n[5] ONE full poll cycle (all symbols, all strategies) ...")
    for sym in list(engine.security_ids):
        t0 = __import__("time").perf_counter()
        engine._poll_symbol(sym)
        el = __import__("time").perf_counter() - t0
        print(f"   {sym:>10}: {engine.status['last_poll'].get(sym, 'n/a')} "
              f"[{el*1000:.0f}ms]")

    print("\n[6] Paper portfolio state (per strategy) ...")
    for strat in engine.strategy_names:
        b = engine.brokers[strat]
        print(f"   [{strat}] equity={b.equity:,.2f} cash={b.cash:,.2f} "
              f"open={b.open_positions()} trades={len(b.trades)}")
    print(f"   DB       : {cfg.db_path}")
    print(f"   signals  : { {n: engine.state.summary(n) for n in engine.strategy_names} }")
    for strat in engine.strategy_names:
        engine.state.sync_broker_positions(strat, engine.brokers[strat])
    state.close()
    print("\n[OK] Data architecture works end-to-end.")


def run_live(iterations: int | None) -> None:
    cfg = load_config()
    dhan = DhanClient(cfg.client_id, cfg.access_token,
                      cfg.exchange_segment, cfg.instrument_type)
    state = StateStore(cfg.db_path)
    engine = LiveEngine(cfg, dhan, state)
    engine.run(iterations=iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Dhan paper trading")
    parser.add_argument("command", choices=["test", "live"])
    parser.add_argument("--iter", type=int, default=None,
                        help="live: stop after N poll cycles")
    args = parser.parse_args()
    if args.command == "test":
        run_test()
    else:
        run_live(args.iter)


if __name__ == "__main__":
    main()