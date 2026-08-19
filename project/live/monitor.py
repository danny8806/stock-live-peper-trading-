#!/usr/bin/env python3
"""FastAPI monitor for the live paper-trading engine (multi-strategy).

Reads the SQLite state store (per-strategy rows).

Run separately while `python run_live.py live` is running:
    python monitor.py          # serves on port from live_config (default 8000)
Endpoints (all accept an optional ?strategy=NAME filter):
    GET /status                config + per-strategy summary/portfolio
    GET /portfolio             per-strategy portfolio P&L (or one strategy)
    GET /trades                closed paper trades
    GET /positions             open paper positions (symbol+strategy)
    GET /signals               emitted signals
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import load_config
from state import StateStore

cfg = load_config()
state = StateStore(cfg.db_path)

app = FastAPI(title="DEMA-ATR Live Paper Monitor (multi-strategy)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


def _resolve_strategy(q: str | None) -> str | None:
    if not q:
        return None
    names = [s.name for s in cfg.strategies]
    if q not in names:
        raise ValueError(f"Unknown strategy '{q}'. Available: {names}")
    return q


@app.get("/status")
def status(strategy: str | None = None):
    strat = _resolve_strategy(strategy)
    return {
        "config": {
            "symbols": cfg.symbols,
            "strategies": [s.to_dict() for s in cfg.strategies],
            "fetch_mode": cfg.fetch_mode,
        },
        "summary": state.summary(strat),
        "portfolio": state.portfolio_summary(
            next(s.capital for s in cfg.strategies if s.name == strat)
            if strat else sum(s.capital for s in cfg.strategies), strat),
    }


@app.get("/portfolio")
def portfolio(strategy: str | None = None):
    """Per-strategy portfolio P&L: capital, equity, cash, realized + unrealized,
    total return, win rate, charges, per-symbol breakdown."""
    strat = _resolve_strategy(strategy)
    cap = (next(s.capital for s in cfg.strategies if s.name == strat)
           if strat else sum(s.capital for s in cfg.strategies))
    return state.portfolio_summary(cap, strat)


@app.get("/trades")
def trades(strategy: str | None = None):
    strat = _resolve_strategy(strategy)
    sql = ("SELECT strategy, symbol, qty, entry_price, exit_price, entry_dt, "
           "exit_dt, pnl, charges, reason FROM trades ")
    args: tuple = ()
    if strat:
        sql += "WHERE strategy=? "
        args = (strat,)
    sql += "ORDER BY id DESC LIMIT 200"
    rows = state.conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


@app.get("/positions")
def positions(strategy: str | None = None):
    strat = _resolve_strategy(strategy)
    rows = state.load_positions()
    if strat:
        rows = [r for r in rows if r["strategy"] == strat]
    return rows


@app.get("/signals")
def signals(strategy: str | None = None):
    strat = _resolve_strategy(strategy)
    sql = "SELECT strategy, ts, symbol, type, detail FROM signals "
    args: tuple = ()
    if strat:
        sql += "WHERE strategy=? "
        args = (strat,)
    sql += "ORDER BY id DESC LIMIT 200"
    rows = state.conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=cfg.monitor_port)