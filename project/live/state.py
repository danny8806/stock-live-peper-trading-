#!/usr/bin/env python3
"""SQLite state store for live paper trading.

Persists positions, closed trades, emitted signals and equity snapshots so
the engine can survive restarts and be audited later. Uses the stdlib
`sqlite3` — no ORM needed for this scale.

Every table is strategy-aware: an open position is keyed by (symbol, strategy)
so multiple strategies can hold the same stock independently, each with its
own capital, qty, entry and P&L.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

POSITION_COLS = ["strategy", "symbol", "qty", "entry_price", "entry_charges",
                 "entry_dt", "sl_level", "tp_level", "last_price"]
TRADE_COLS = ["strategy", "symbol", "qty", "entry_price", "exit_price",
              "entry_dt", "exit_dt", "pnl", "charges", "reason"]


class StateStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL DEFAULT '',
                qty INTEGER, entry_price REAL, entry_charges REAL,
                entry_dt TEXT, sl_level REAL, tp_level REAL, last_price REAL,
                PRIMARY KEY (symbol, strategy)
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL DEFAULT '',
                symbol TEXT, qty INTEGER, entry_price REAL, exit_price REAL,
                entry_dt TEXT, exit_dt TEXT, pnl REAL, charges REAL, reason TEXT
            );
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL DEFAULT '',
                ts TEXT, symbol TEXT, type TEXT, detail TEXT
            );
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL DEFAULT '',
                ts TEXT, equity REAL, cash REAL, positions INTEGER,
                unrealized REAL DEFAULT 0, invested REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                last_fast_dt TEXT,
                last_1m_dt TEXT,
                ts TEXT,
                PRIMARY KEY (strategy, symbol)
            );
        """)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add strategy columns / composite PKs to databases created before
        multi-strategy support."""
        def cols(tbl: str) -> list[str]:
            return [r[1] for r in self.conn.execute(f"PRAGMA table_info({tbl})").fetchall()]

        if "strategy" not in cols("equity"):
            self.conn.execute("ALTER TABLE equity ADD COLUMN strategy TEXT NOT NULL DEFAULT ''")
        if "strategy" not in cols("signals"):
            self.conn.execute("ALTER TABLE signals ADD COLUMN strategy TEXT NOT NULL DEFAULT ''")
        if "strategy" not in cols("trades"):
            self.conn.execute("ALTER TABLE trades ADD COLUMN strategy TEXT NOT NULL DEFAULT ''")
        if "strategy" not in cols("positions"):
            # positions was PK(symbol) only -> rebuild with (symbol, strategy)
            self.conn.execute("ALTER TABLE positions RENAME TO positions_old")
            self.conn.execute("""
                CREATE TABLE positions (
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL DEFAULT '',
                    qty INTEGER, entry_price REAL, entry_charges REAL,
                    entry_dt TEXT, sl_level REAL, tp_level REAL, last_price REAL,
                    PRIMARY KEY (symbol, strategy)
                )""")
            self.conn.execute(
                "INSERT INTO positions (symbol, strategy, qty, entry_price, "
                "entry_charges, entry_dt, sl_level, tp_level, last_price) "
                "SELECT symbol, '', qty, entry_price, entry_charges, entry_dt, "
                "sl_level, tp_level, last_price FROM positions_old")
            self.conn.execute("DROP TABLE positions_old")

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def upsert_position(self, pos: dict) -> None:
        pos = dict(pos)
        pos.setdefault("strategy", "")
        cols = ", ".join(POSITION_COLS)
        ph = ", ".join("?" * len(POSITION_COLS))
        vals = [pos.get(c) for c in POSITION_COLS]
        self.conn.execute(
            f"INSERT INTO positions ({cols}) VALUES ({ph}) "
            f"ON CONFLICT(symbol, strategy) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in POSITION_COLS),
            vals)
        self.conn.commit()

    def delete_position(self, symbol: str, strategy: str = "") -> None:
        self.conn.execute("DELETE FROM positions WHERE symbol=? AND strategy=?",
                          (symbol, strategy))
        self.conn.commit()

    def load_positions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]

    def sync_broker_positions(self, strategy_or_broker, broker=None) -> None:
        """Make the DB positions table match ONE strategy's in-memory positions.

        Call either sync_broker_positions(strategy_name, broker) or the legacy
        sync_broker_positions(broker) (strategy taken from the broker).
        """
        if broker is None:
            broker = strategy_or_broker
            strategy = broker.strategy or ""
        else:
            strategy = strategy_or_broker
        db_rows = self.conn.execute(
            "SELECT symbol FROM positions WHERE strategy=?", (strategy,)).fetchall()
        db_symbols = {r["symbol"] for r in db_rows}
        for symbol, pos in broker.positions.items():
            pos["strategy"] = strategy
            self.upsert_position(pos)
        for symbol in db_symbols - set(broker.positions.keys()):
            self.delete_position(symbol, strategy)

    # ------------------------------------------------------------------
    # Trades / signals / equity
    # ------------------------------------------------------------------
    def save_trade(self, trade: dict) -> None:
        trade = dict(trade)
        trade.setdefault("strategy", "")
        self.conn.execute(
            "INSERT INTO trades (strategy, symbol, qty, entry_price, exit_price, "
            "entry_dt, exit_dt, pnl, charges, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [trade.get(c) for c in TRADE_COLS])
        self.conn.commit()

    def close_position_trade(self, trade: dict, strategy: str = "") -> bool:
        """Atomically persist a closing trade AND remove its open position.

        A single SQLite transaction (BEGIN IMMEDIATE) replaces the old
        save_trade-then-delete_position two-step, which left a crash window
        between the two commits: on restart the stale position was re-sold,
        recording a duplicate exit. Now either both write or neither does.

        Idempotency: if a trade for this exact position (same strategy, symbol,
        entry_dt, entry_price and qty) already exists — e.g. a leftover from the
        old crash window — the stale position row is still removed but no second
        trade is inserted. Returns True when a new trade was recorded.
        """
        trade = dict(trade)
        strat = strategy or trade.get("strategy", "")
        trade.setdefault("strategy", strat)
        inserted = False
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            entry_dt = trade.get("entry_dt")
            if entry_dt:
                exists = self.conn.execute(
                    "SELECT id FROM trades WHERE strategy=? AND symbol=? AND "
                    "entry_dt=? AND entry_price=? AND qty=?",
                    (strat, trade["symbol"], entry_dt,
                     trade.get("entry_price"), trade.get("qty"))).fetchone()
            else:
                exists = None
            if exists is None:
                self.conn.execute(
                    "INSERT INTO trades (strategy, symbol, qty, entry_price, "
                    "exit_price, entry_dt, exit_dt, pnl, charges, reason) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [trade.get(c) for c in TRADE_COLS])
                inserted = True
            self.conn.execute(
                "DELETE FROM positions WHERE symbol=? AND strategy=?",
                (trade["symbol"], strat))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return inserted

    def log_signal(self, symbol: str, sig_type: str, detail: dict | None = None,
                   strategy: str = "") -> None:
        self.conn.execute(
            "INSERT INTO signals (strategy, ts, symbol, type, detail) "
            "VALUES (?, datetime('now','localtime'), ?, ?, ?)",
            (strategy, symbol, sig_type, json.dumps(detail, default=str)))
        self.conn.commit()

    def log_equity(self, equity: float, cash: float, n_positions: int,
                   unrealized: float = 0.0, invested: float = 0.0,
                   strategy: str = "") -> None:
        self.conn.execute(
            "INSERT INTO equity (strategy, ts, equity, cash, positions, "
            "unrealized, invested) VALUES (?, datetime('now','localtime'), ?, ?, ?, ?, ?)",
            (strategy, round(equity, 2), round(cash, 2), n_positions,
             round(unrealized, 2), round(invested, 2)))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Signal-processing checkpoint (restart-resume cursor)
    # ------------------------------------------------------------------
    def save_checkpoint(self, strategy: str, symbol: str,
                        last_fast_dt: str | None = None,
                        last_1m_dt: str | None = None) -> None:
        """Persist the processing cursors (last consumed fast-TF candle and
        last merged 1-min candle) per (strategy, symbol), so a restart resumes
        from exactly that bar. A None value preserves the existing cursor —
        a valid cursor is never overwritten with nothing."""
        fast = str(last_fast_dt) if last_fast_dt is not None else None
        one = str(last_1m_dt) if last_1m_dt is not None else None
        if fast is None and one is None:
            return
        row = self.conn.execute(
            "SELECT last_fast_dt, last_1m_dt FROM checkpoints "
            "WHERE strategy=? AND symbol=?", (strategy, symbol)).fetchone()
        if fast is None and row is not None and row["last_fast_dt"] is not None:
            fast = row["last_fast_dt"]
        if one is None and row is not None and row["last_1m_dt"] is not None:
            one = row["last_1m_dt"]
        self.conn.execute(
            "INSERT INTO checkpoints (strategy, symbol, last_fast_dt, last_1m_dt, ts) "
            "VALUES (?,?,?,?, datetime('now','localtime')) "
            "ON CONFLICT(strategy, symbol) DO UPDATE SET "
            "last_fast_dt=excluded.last_fast_dt, last_1m_dt=excluded.last_1m_dt, "
            "ts=excluded.ts",
            (strategy, symbol, fast, one))
        self.conn.commit()

    def load_checkpoints(self) -> dict[tuple[str, str], dict]:
        """All persisted processing cursors, keyed (strategy, symbol)."""
        rows = self.conn.execute("SELECT * FROM checkpoints").fetchall()
        return {(r["strategy"], r["symbol"]): dict(r) for r in rows}

    # ------------------------------------------------------------------
    def summary(self, strategy: str | None = None) -> dict:
        if strategy:
            where, args = "WHERE strategy=?", (strategy,)
        else:
            where, args = "", ()
        trades = self.conn.execute(
            f"SELECT COUNT(*) n, COALESCE(SUM(pnl),0) pnl FROM trades {where}",
            args).fetchone()
        open_pos = self.conn.execute(
            f"SELECT COUNT(*) n FROM positions {where}", args).fetchone()["n"]
        return {
            "strategy": strategy,
            "trades_closed": trades["n"],
            "realized_pnl": round(trades["pnl"], 2),
            "open_positions": open_pos,
        }

    def portfolio_summary(self, capital: float, strategy: str | None = None) -> dict:
        """Full portfolio P&L computed from the DB for one strategy (or all if
        strategy is None) — the audit view of the live paper portfolio."""
        if strategy:
            where, args = "WHERE strategy=?", (strategy,)
        else:
            where, args = "", ()
        trades = self.conn.execute(
            f"SELECT COUNT(*) n, COALESCE(SUM(pnl),0) pnl, "
            f"COALESCE(SUM(charges),0) chg, "
            f"COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),0) wins "
            f"FROM trades {where}", args).fetchone()
        realized = round(trades["pnl"], 2)

        rows = self.conn.execute(
            f"SELECT symbol, strategy, qty, entry_price, entry_charges, "
            f"last_price FROM positions {where}", args).fetchall()
        unrealized = 0.0
        invested_value = 0.0
        per_symbol: dict[str, dict] = {}
        for p in rows:
            upnl = (p["last_price"] - p["entry_price"]) * p["qty"] - p["entry_charges"]
            unrealized += upnl
            invested_value += p["qty"] * p["last_price"]
            # combined view (strategy=None) prefixes named strategies to
            # disambiguate same-symbol holdings; filtered views use plain symbols
            key = (p["symbol"] if (strategy is not None or not p["strategy"])
                   else f"{p['strategy']}/{p['symbol']}")
            per_symbol[key] = {
                "strategy": p["strategy"],
                "qty": p["qty"],
                "entry_price": round(p["entry_price"], 2),
                "last_price": round(p["last_price"], 2),
                "unrealized_pnl": round(upnl, 2),
            }
        per_sym_trades = self.conn.execute(
            f"SELECT symbol, strategy, COUNT(*) n, COALESCE(SUM(pnl),0) pnl "
            f"FROM trades {where} GROUP BY symbol, strategy", args).fetchall()
        for r in per_sym_trades:
            key = (r["symbol"] if (strategy is not None or not r["strategy"])
                   else f"{r['strategy']}/{r['symbol']}")
            d = per_symbol.setdefault(key, {})
            d["trades_closed"] = r["n"]
            d["realized_pnl"] = round(r["pnl"], 2)

        total = realized + unrealized
        equity = round(capital + total, 2)
        cash = round(equity - invested_value, 2)
        return {
            "strategy": strategy,
            "capital": round(capital, 2),
            "equity": equity,
            "cash": cash,
            "invested_value": round(invested_value, 2),
            "realized_pnl": realized,
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(total, 2),
            "total_return_pct": round(total / capital * 100.0, 4) if capital else 0.0,
            "open_positions": len(rows),
            "trades_closed": trades["n"],
            "win_rate_pct": round(trades["wins"] / trades["n"] * 100.0, 2) if trades["n"] else 0.0,
            "charges_total": round(trades["chg"], 2),
            "per_symbol": per_symbol,
        }

    def close(self) -> None:
        self.conn.close()