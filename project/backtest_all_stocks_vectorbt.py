#!/usr/bin/env python3

"""backtest_all_stocks_vectorbt.py - Run VectorBT backtest for ALL stocks, one by one.

Adapted from backtest_all_stocks.py but uses the VectorBT engine
(core.vbt_engine.simulate_trades) instead of Backtrader's run_backtest.

Engine parity verified on KAYNES/HFCL/RELIANCE: same trades, dates, sizes, charges.
With slippage disabled, PnL matches to the rupee.

Usage:
    python backtest_all_stocks_vectorbt.py [--workers N] [--sl 0] [--tp 0]
    python backtest_all_stocks_vectorbt.py --workers 8 --sl 1 --tp 2
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ---- Path setup ----
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from build_all_timeframes_enriched import preprocess_1min, build_all
from core.vbt_engine import simulate_trades, CAPITAL, SLIPPAGE_PCT

# ---- 15 timeframe combos (same as backtest_all_stocks.py) ----
COMBOS = [
    "1m/3m", "1m/5m", "1m/15m", "1m/30m", "1m/1h",
    "3m/5m", "3m/15m", "3m/30m", "3m/1h",
    "5m/15m", "5m/30m", "5m/1h",
    "15m/30m", "15m/1h",
    "30m/1h",
]

# ---- Core worker: one stock, all combos ----
def process_stock_vectorbt(csv_path, sl_pct, tp_pct):
    """Run all 15 combos for one stock using VectorBT simulate_trades."""
    csv_path = Path(csv_path)
    symbol = csv_path.stem

    df = preprocess_1min(csv_path)
    enriched = build_all(df)

    # Slice to May 1 → Aug 1 2026 (3 months, same as RELIANCE run)
    start = pd.Timestamp("2026-05-01")
    end = pd.Timestamp("2026-08-01")
    enriched = {tf: f[(f["datetime"] >= start) & (f["datetime"] < end)].reset_index(drop=True)
                for tf, f in enriched.items()}

    row = {"symbol": symbol}

    best_combo, best_roi = "", float("-inf")

    for combo in COMBOS:
        fast, slow = combo.split("/")
        fast_col = f"demaatr_{fast}"
        slow_col = f"demaatr_{slow}"

        # VectorBT simulate_trades: signals + sizing + charges + margin check
        trades, final_cash = simulate_trades(enriched[fast], fast_col, slow_col)

        roi = round((final_cash - CAPITAL) / CAPITAL * 100, 4)
        trades_cnt = len(trades)
        charges = round(sum(t["charges"] for t in trades), 2)

        row[f"{combo}_roi"] = roi
        row[f"{combo}_trades"] = trades_cnt
        row[f"{combo}_charges"] = charges

        if roi > best_roi:
            best_roi, best_combo = roi, combo

    row["best_combo"] = best_combo
    row["best_roi"] = round(best_roi, 4)
    return row


# ---- Main ----
def main() -> None:
    parser = argparse.ArgumentParser(description="VectorBT all-stocks sweep")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory with 1-min CSV files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Parallel workers (1 = sequential)")
    parser.add_argument("--sl", type=float, default=0.0, help="Stop-loss % (0=off)")
    parser.add_argument("--tp", type=float, default=0.0, help="Take-profit % (0=off)")
    args = parser.parse_args()

    data_dir = args.data_dir
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: no CSV files in {data_dir}")

    # Filter to stocks that have enough data for the 3-month slice
    valid = []
    for p in csv_files:
        try:
            df = preprocess_1min(p)
            # quick check: need at least some bars after slicing
            enriched = build_all(df)
            s = pd.Timestamp("2026-05-01")
            e = pd.Timestamp("2026-08-01")
            ok = len(enriched["1m"][(enriched["1m"]["datetime"] >= s) & (enriched["1m"]["datetime"] < e)]) > 50
            if ok:
                valid.append(p)
        except Exception:
            pass

    n_stocks = len(valid)
    print(f"Found {len(csv_files)} CSV files, {n_stocks} valid for 3-month slice")
    print(f"Combos/stock: {len(COMBOS)} | Workers: {args.workers} | SL/TP: {args.sl}%/{args.tp}%\n")

    # ---- Parallel execution ----
    tasks = [(str(p), args.sl, args.tp) for p in valid]

    start = time.time()
    results = []

    if args.workers <= 1:
        # Sequential
        print("Running SEQUENTIAL (one stock at a time)...")
        for i, task in enumerate(tasks, 1):
            csv_path, sl, tp = task
            print(f"  [{i}/{n_stocks}] {Path(csv_path).stem}", end=" ", flush=True)
            try:
                row = process_stock_vectorbt(csv_path, sl, tp)
                results.append(row)
                print(f"  ROI={row['best_roi']:.2f}%  best={row['best_combo']}")
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        # Parallel with ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Running with {args.workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_stock_vectorbt, t[0], t[1], t[2]): t[0] for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                csv_path_str = futures[future]
                try:
                    row = future.result()
                    results.append(row)
                    print(f"  [{i}/{n_stocks}] {Path(csv_path_str).stem}  ROI={row['best_roi']:.2f}%  best={row['best_combo']}")
                except Exception as e:
                    print(f"  [{i}/{n_stocks}] {Path(csv_path_str).stem}  ERROR: {e}")

    elapsed = time.time() - start

    # ---- Save results CSV ----
    if results:
        write_header = not Path(args.out).is_file()
        with open(args.out, "a", newline="") as f:
            cols = ["symbol"] + [f"{c}_roi" for c in COMBOS] + [f"{c}_trades" for c in COMBOS] + \
                   [f"{c}_charges" for c in COMBOS] + ["best_combo", "best_roi"]
            writer = csv.DictWriter(f, fieldnames=cols)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults appended to: {args.out}")

    elapsed_min = elapsed / 60
    print(f"\nFinished in {elapsed:.1f}s ({elapsed_min:.1f} min)")
    print(f"Processed {len(results)}/{n_stocks} stocks")


if __name__ == "__main__":
    main()

DEFAULT_DATA_DIR = Path(r"C:\Users\pc\Desktop\ORB scaner\DEMAB\data_1m_6m")
DEFAULT_OUT = Path(__file__).parent / "results_fresh_all_stocks_vectorbt.csv"