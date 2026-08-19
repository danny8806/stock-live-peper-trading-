#!/usr/bin/env python3
"""Paper broker — simulates fills, charges, positions and SL/TP without
sending real orders to Dhan.

Mirrors the backtest execution model:
  - Buy fills at live_price * (1 + slippage), Sell at live_price * (1 - slippage)
  - Full Indian equity charge breakdown per side (EquityChargesEngine)
  - Position sizing = max affordable shares within per-stock capital + cash
  - Margin check: fill cost + buy charges must fit available cash
  - Optional SL / TP (percent of entry) monitored on every live tick
  - Max concurrent positions across the portfolio
"""
from __future__ import annotations

import math

from core.equity_charges import EquityChargesEngine


class PaperBroker:
    def __init__(self, capital: float, max_positions: int = 20,
                 max_capital_per_stock: float = 100000.0,
                 slippage_pct: float = 0.0005,
                 sl_pct: float = 0.0, tp_pct: float = 0.0,
                 strategy: str = ""):
        self.capital = capital
        self.cash = capital
        self.max_positions = max_positions
        self.max_capital_per_stock = max_capital_per_stock
        self.slippage_pct = slippage_pct
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.strategy = strategy
        self.engine = EquityChargesEngine()
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []

    # ------------------------------------------------------------------
    @property
    def equity(self) -> float:
        value = self.cash
        for pos in self.positions.values():
            value += pos["qty"] * pos["last_price"]
        return value

    def open_positions(self) -> int:
        return len(self.positions)

    # ------------------------------------------------------------------
    def _max_size(self, symbol: str, price: float) -> int:
        if price <= 0:
            return 0
        alloc = min(self.max_capital_per_stock, self.cash)
        if alloc <= 0:
            return 0
        qty = int(alloc / price)
        if qty <= 0:
            return 0
        chg = self.engine.calculate(price, qty, is_buy=True).total_charges
        if price * qty + chg > alloc:
            qty = int((alloc - chg) / price)
        return max(qty, 0)

    def buy(self, symbol: str, live_price: float, quote_dt=None) -> dict | None:
        """Place a paper market buy. Returns the position dict or None if rejected."""
        if symbol in self.positions:
            return None
        if len(self.positions) >= self.max_positions:
            return None
        fill = live_price * (1.0 + self.slippage_pct)
        qty = self._max_size(symbol, fill)
        if qty <= 0:
            return None
        buy_chg = self.engine.calculate(fill, qty, is_buy=True).total_charges
        cost = fill * qty + buy_chg
        if cost > self.cash:  # margin check
            return None
        self.cash -= cost
        pos = {
            "strategy": self.strategy,
            "symbol": symbol,
            "qty": qty,
            "entry_price": fill,            # full precision so P&L stays consistent
            "entry_charges": buy_chg,
            "entry_dt": quote_dt,
            "last_price": live_price,
            "sl_level": round(fill * (1.0 - self.sl_pct / 100.0), 2) if self.sl_pct > 0 else None,
            "tp_level": round(fill * (1.0 + self.tp_pct / 100.0), 2) if self.tp_pct > 0 else None,
        }
        self.positions[symbol] = pos
        return pos

    def sell(self, symbol: str, live_price: float, quote_dt=None,
             reason: str = "cross") -> dict | None:
        """Close a paper position at market. Returns the closed trade dict."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        fill = live_price * (1.0 - self.slippage_pct)
        sell_chg = self.engine.calculate(fill, pos["qty"], is_buy=False).total_charges
        gross = (fill - pos["entry_price"]) * pos["qty"]
        pnl = gross - pos["entry_charges"] - sell_chg
        self.cash += fill * pos["qty"] - sell_chg
        trade = {
            "strategy": self.strategy,
            "symbol": symbol,
            "qty": pos["qty"],
            "entry_price": pos["entry_price"],
            "exit_price": round(fill, 2),
            "entry_dt": pos["entry_dt"],
            "exit_dt": quote_dt,
            "pnl": round(pnl, 2),
            "charges": round(pos["entry_charges"] + sell_chg, 2),
            "reason": reason,
        }
        del self.positions[symbol]
        self.trades.append(trade)
        return trade

    def mark_to_market(self, prices: dict[str, float]) -> list[dict]:
        """Update last_price for open positions and close any whose SL/TP hit.
        Returns the list of trades closed by stops this tick."""
        closed = []
        for symbol, price in prices.items():
            pos = self.positions.get(symbol)
            if pos is None:
                continue
            pos["last_price"] = price
            reason = None
            if pos.get("sl_level") is not None and price <= pos["sl_level"]:
                reason = "SL"
            elif pos.get("tp_level") is not None and price >= pos["tp_level"]:
                reason = "TP"
            if reason:
                t = self.sell(symbol, price, reason=reason)
                if t:
                    closed.append(t)
        return closed

    def holding(self, symbol: str) -> bool:
        return symbol in self.positions

    # ------------------------------------------------------------------
    # Portfolio P&L (single capital shared across all stocks)
    # ------------------------------------------------------------------
    def invested_cost(self) -> float:
        """Capital deployed in open positions (entry price*qty + buy charges)."""
        return sum(p["qty"] * p["entry_price"] + p["entry_charges"]
                   for p in self.positions.values())

    def invested_value(self) -> float:
        """Current market value of all open positions."""
        return sum(p["qty"] * p["last_price"] for p in self.positions.values())

    def unrealized_pnl(self) -> float:
        """Mark-to-market PnL of open positions (net of entry charges)."""
        return sum((p["last_price"] - p["entry_price"]) * p["qty"] - p["entry_charges"]
                   for p in self.positions.values())

    def realized_pnl(self) -> float:
        return sum(t["pnl"] for t in self.trades)

    def portfolio_pnl(self) -> dict:
        """Full portfolio P&L snapshot (the live 'trading P&L')."""
        realized = self.realized_pnl()
        unrealized = self.unrealized_pnl()
        total = realized + unrealized
        return {
            "capital": round(self.capital, 2),
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "invested_value": round(self.invested_value(), 2),
            "invested_cost": round(self.invested_cost(), 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(total, 2),
            "total_return_pct": round(total / self.capital * 100.0, 4),
            "open_positions": self.open_positions(),
            "trades_closed": len(self.trades),
        }
