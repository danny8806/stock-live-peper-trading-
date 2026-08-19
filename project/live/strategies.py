#!/usr/bin/env python3
"""Strategy registry — makes the live indicator/signal engine pluggable.

Adding a new strategy = register a (pipeline_factory, signals_factory) pair
here and set `strategy.name` in live_config.json. The engine and every other
layer (buffers, broker, DB, monitor) are strategy-agnostic.

Contract for a pipeline factory:  cfg -> object with
    .compute(closed_1m) -> fast-TF DataFrame (must include the signal cols)
    .fast_col / .slow_col  (column names the signal engine consumes)
    .closed_slice(buffer_df) -> closed 1-min bars

Contract for a signals factory: cfg, pipeline -> object with
    .warmup(fast_df)                      replay history, update state only
    .on_new_bars(fast_df, in_position) -> [{type:"BUY"|"SELL", ...}, ...]
"""
from __future__ import annotations

from config import LiveConfig
from pipeline import DEMAATRPipeline
from signals import DEMAATRSignals

_REGISTRY: dict[str, dict] = {}


def register(name: str, pipeline_cls, signals_cls) -> None:
    _REGISTRY[name] = {"pipeline": pipeline_cls, "signals": signals_cls}


def _engine_name(cfg) -> str:
    """Strategy name -> registered engine. Config strategy names are free
    labels (e.g. 'dema_15m_1h'); unless the name IS a registered engine the
    default 'dema_atr' engine is used."""
    name = getattr(cfg, "name", None) or getattr(cfg, "strategy_name", "dema_atr")
    return name if name in _REGISTRY else "dema_atr"


def create_pipeline(cfg):
    entry = _REGISTRY.get(_engine_name(cfg))
    if entry is None:
        raise ValueError(f"Unknown strategy engine '{_engine_name(cfg)}'. "
                         f"Registered: {sorted(_REGISTRY)}")
    return entry["pipeline"].from_config(cfg)


def create_signals(cfg, pipeline):
    entry = _REGISTRY.get(_engine_name(cfg))
    if entry is None:
        raise ValueError(f"Unknown strategy engine '{_engine_name(cfg)}'")
    return entry["signals"].from_config(cfg, pipeline)


def create_broker(cfg):
    """One isolated paper broker per strategy (own capital + risk limits).
    The broker's `strategy` label is the CONFIG strategy name (the trade
    owner), not the engine name."""
    from paper_broker import PaperBroker
    name = getattr(cfg, "name", None) or "dema_atr"
    return PaperBroker(
        capital=cfg.capital,
        max_positions=cfg.max_positions,
        max_capital_per_stock=cfg.max_capital_per_stock,
        slippage_pct=cfg.slippage_pct,
        sl_pct=cfg.sl_pct, tp_pct=cfg.tp_pct,
        strategy=name,
    )


# ---------------------------------------------------------------------------
# Built-in strategy: DEMA-ATR crossover + breakout (the original backtest port)
# ---------------------------------------------------------------------------
register("dema_atr", DEMAATRPipeline, DEMAATRSignals)