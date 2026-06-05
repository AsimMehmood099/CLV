import pandas as pd
import numpy as np
import sys
import os

# ── Add your CLV root to path ──────────────────────────
# Change this to your actual CLV folder path
CLV_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CLV_ROOT)

from train import clean_data, build_clv

# ── Load your test CSV ─────────────────────────────────
CSV_FILE = os.path.join(CLV_ROOT, "test_data.csv")

print("=" * 50)
print("STEP 1: Loading CSV...")
df = pd.read_csv(CSV_FILE, encoding="ISO-8859-1")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(df.head(3))

print("\n" + "=" * 50)
print("STEP 2: Running clean_data()...")
df_clean = clean_data(df)
if df_clean is None:
    print("  ❌ clean_data() returned None — date column not found!")
    sys.exit()
print(f"  ✅ Rows after cleaning: {len(df_clean)}")
print(f"  Avg TotalPrice: ${df_clean['TotalPrice'].mean():.2f}")

print("\n" + "=" * 50)
print("STEP 3: Running build_clv()...")
clv = build_clv(df_clean)
if clv is None or len(clv) == 0:
    print("  ❌ build_clv() returned empty — no repeat customers!")
    sys.exit()
print(f"  ✅ CLV customers: {len(clv)}")
print(clv.head())

print("\n" + "=" * 50)
print("✅ ALL STEPS PASSED — Upload should work!")
print(f"  Total customers ready for prediction: {len(clv)}")