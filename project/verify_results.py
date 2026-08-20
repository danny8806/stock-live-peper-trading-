"""Independent verification: run fresh backtests for sample stocks using the
original run_backtest pipeline and compare against results_fresh_all_stocks.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_all_timeframes_enriched import preprocess_1min, build_all
from core.backtrader_backtest import run_backtest
import pandas as pd

DATA_DIR = Path(r"C:\Users\pc\Desktop\ORB scaner\DEMAB\data_1m_6m")
CSV_FILE = Path(__file__).resolve().parent / "results_fresh_all_stocks.csv"

# Stocks to verify — top performers from best_results.py + diverse sample
CHECKS = [
    # Top performers (from best_results output)
    ("HFCL",      "3m/30m"),    # #1 best overall: 138.56%
    ("HFCL",      "5m/15m"),    # #2: 115.73%
    ("TEJASNET",  "15m/1h"),    # #9: 81.10%
    ("CEMPRO",    "5m/1h"),     # #12: 75.33%
    ("BLISSGVS",  "5m/15m"),    # #13: 74.00%
    ("SPARC",     "3m/30m"),    # #15: 71.08%
    ("SUVEN",     "30m/1h"),    # #16: 70.08%
    ("RAJESHEXPO","30m/1h"),    # #22: 62.09%
    ("WABAG",     "30m/1h"),    # #24: 59.66%
    ("GUJALKALI", "5m/15m"),    # #25: 58.59%
    ("LODHA",     "15m/30m"),   # #27: 57.91%
    ("KALYANKJIL","30m/1h"),    # #28: 57.62%
    # Diverse sample (different performance levels)
    ("3MINDIA",   "15m/1h"),    # near-zero: -1.28%
    ("KAYNES",    "5m/30m"),    # mid: 37.45%
    ("RELIANCE",  "30m/1h"),    # negative: -9.01%
    ("TCS",       "5m/30m"),    # negative: -10.63%
    ("ADANIENT",  "15m/1h"),    # positive: 35.69%
    ("ZEEL",      "30m/1h"),    # positive: 43.91%
    # Worst performers
    ("FLFL",      "1m/5m"),     # worst: -100.00%
    ("VEDL",      "3m/1h"),     # worst best_roi: -61.02%
]

df = pd.read_csv(CSV_FILE)
print(f"Verifying {len(CHECKS)} stocks against {CSV_FILE.name}\n")

all_match = True
for symbol, combo in CHECKS:
    fast, slow = combo.split("/")
    csv_row = df[df["symbol"] == symbol].iloc[0]

    # Run fresh backtest
    csv_path = DATA_DIR / f"{symbol}.csv"
    df_1m = preprocess_1min(csv_path)
    enriched = build_all(df_1m)
    r = run_backtest(enriched, fast, slow, verbose=False, sl_pct=0.0, tp_pct=0.0)

    # Compare
    csv_roi = float(csv_row[f"{combo}_roi"])
    csv_trades = int(csv_row[f"{combo}_trades"])
    csv_maxdd = float(csv_row[f"{combo}_maxdd"])
    csv_charges = float(csv_row[f"{combo}_charges"])

    fresh_roi = round(r["roi"], 4)
    fresh_trades = int(r["trades"])
    fresh_maxdd = round(r["max_dd"], 4)
    fresh_charges = round(r["total_charges"], 2)

    roi_ok = abs(csv_roi - fresh_roi) < 0.01
    trd_ok = csv_trades == fresh_trades
    dd_ok  = abs(csv_maxdd - fresh_maxdd) < 0.01
    chg_ok = abs(csv_charges - fresh_charges) < 1.0

    match = roi_ok and trd_ok and dd_ok and chg_ok
    status = "MATCH" if match else "MISMATCH"
    if not match:
        all_match = False

    print(f"{symbol:>12} {combo:>8}  {status}")
    print(f"  {'':>12}  CSV:  roi={csv_roi:>9.4f}  trades={csv_trades:>5}  maxdd={csv_maxdd:>8.4f}  charges={csv_charges:>10.2f}")
    print(f"  {'':>12}  FRESH: roi={fresh_roi:>9.4f}  trades={fresh_trades:>5}  maxdd={fresh_maxdd:>8.4f}  charges={fresh_charges:>10.2f}")
    print()

print("=" * 60)
if all_match:
    print("ALL CHECKS PASSED — CSV results are accurate and correct.")
else:
    print("SOME CHECKS FAILED — see mismatches above.")