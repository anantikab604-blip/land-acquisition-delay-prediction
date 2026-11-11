
"""
Step 1 (land acquisition project): inspect the generated dataset.

Run:
    python3 step1_inspect_data.py

It looks for land_acquisition_projects.csv in the current folder first,
then in your home folder. Or give the path yourself:
    python3 step1_inspect_data.py /path/to/land_acquisition_projects.csv
"""

import sys
from pathlib import Path

import pandas as pd

NAME = "land_acquisition_projects.csv"

if len(sys.argv) > 1:
    candidates = [Path(sys.argv[1]).expanduser()]
else:
    candidates = [Path.cwd() / NAME, Path.home() / NAME]

path = next((p for p in candidates if p.exists()), None)

if path is None:
    print(f"Could not find {NAME}. Put it in this folder or pass its path.")
    sys.exit(1)

df = pd.read_csv(path)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)

print(f"File: {path}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Duplicate rows: {int(df.duplicated().sum())}")

print("\n--- Every column: type, missing values, distinct values ---")

info = pd.DataFrame({
    "type": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "distinct": df.nunique(),
})

print(info.to_string())

print("\n--- First 3 rows (one column per line so it fits) ---")
print(df.head(3).T.to_string())

num = df.select_dtypes("number")

print("\n--- Numeric columns: summary ---")
print(
    num.describe().T[["mean", "std", "min", "50%", "max"]]
    .round(2)
    .to_string()
)

print("\n--- Text columns: most common values ---")

for col in df.select_dtypes(exclude="number").columns:
    top = df[col].value_counts().head(5)
    print(f"{col}: " + ", ".join(f"{k} ({v})" for k, v in top.items()))

print("\n--- Share of projects flagged delayed (about 25% per stage expected) ---")

for col in [c for c in df.columns if "delay" in c.lower()]:
    try:
        values = df[col].replace({
            "Yes": 1,
            "No": 0,
            "True": 1,
            "False": 0,
        })
        print(
            f"{col}: "
            f"{pd.to_numeric(values, errors='coerce').mean() * 100:.1f}%"
        )
    except Exception as e:
        print(f"{col}: could not compute ({e})")

print("\n--- Quick sanity checks ---")

for col in [c for c in num.columns if "duration" in c.lower()]:
    bad = int((num[col] < 0).sum())
    if bad:
        print(f"WARNING: {col} has {bad} negative values")

print("Done.")
