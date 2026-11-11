"""
Step 2 (land acquisition project): leakage and signal scan.

For every column, measure how strongly it is linked to each stage's "delayed"
flag. This tells us (a) which columns would be cheating (they reveal the
answer) and (b) which columns carry real signal.

Run from the folder that holds land_acquisition_projects.csv:
    python step2_leakage_scan.py

Output: printed report + leakage_scan.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)

NAME = "land_acquisition_projects.csv"
path = next((p for p in [Path.cwd() / NAME, Path.home() / NAME] if p.exists()), None)
if path is None:
    raise SystemExit(f"Could not find {NAME} in this folder or your home folder.")
df = pd.read_csv(path)
print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {path}")

STAGES = ["notification", "approval", "compensation",
          "legal_dispute", "possession", "rehabilitation"]
TARGETS = {s: f"{s}_delayed" for s in STAGES}


def stage_of(col):
    """Which stage a duration/risk column belongs to (None = project attribute)."""
    for i, s in enumerate(STAGES):
        if col in (f"{s}_duration_days", f"{s}_risk_score"):
            return i
    return None


def strength(x, y):
    """How well one column alone separates delayed from not delayed.
    0.50 = no link, 1.00 = perfect. Direction is ignored."""
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return np.nan
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


# ---------- Columns to scan ----------
num_cols = [c for c in df.select_dtypes("number").columns if not c.endswith("_delayed")]
scan = df[num_cols].copy()
scan["possession_status (ordered)"] = df["possession_status"].map(
    {"Not Started": 0, "Partial": 1, "Complete": 2})
for c in ["compensation_disbursed_pct", "stakeholder_responsiveness_score"]:
    scan[f"{c}__is_missing"] = df[c].isna().astype(int)

table = pd.DataFrame({
    s: {col: strength(scan[col], df[TARGETS[s]]) for col in scan.columns}
    for s in STAGES
})

print("\n=== 1. How strongly each column is linked to each stage's delay ===")
print("(0.50 = no link, 1.00 = perfect; direction ignored)")
print(table.round(2).to_string())
table.round(3).to_csv("leakage_scan.csv")

# ---------- Strongest links, with a label ----------
rows = []
for j, s in enumerate(STAGES):
    for col in scan.columns:
        i = stage_of(col)
        if i == j:
            continue  # a stage's own duration/risk is obviously linked, skip
        tag = ("project attribute / snapshot" if i is None
               else "LATER stage" if i > j else "earlier stage")
        rows.append((table.loc[col, s], col, f"{s}_delayed", tag))
top = pd.DataFrame(rows, columns=["strength", "column", "predicts", "type"])
top = top.sort_values("strength", ascending=False).head(25)
print("\n=== 2. Strongest links (not counting a stage's own duration/risk) ===")
print(top.round(2).to_string(index=False))

# ---------- Risk score check ----------
print("\n=== 3. Is each risk_score just a copy of the duration? ===")
for s in STAGES:
    d, r = df[f"{s}_duration_days"], df[f"{s}_risk_score"]
    print(f"{s:<14} correlation with duration = {d.corr(r):.2f} | "
          f"risk_score alone predicts delayed at {strength(r, df[TARGETS[s]]):.2f}")

# ---------- Do stages influence each other? ----------
print("\n=== 4. Correlation between stage durations ===")
dur = df[[f"{s}_duration_days" for s in STAGES]]
dur.columns = STAGES
print(dur.corr().round(2).to_string())

# ---------- Categorical columns ----------
print("\n=== 5. Delay rate (%) by project type ===")
by_type = df.groupby("project_type")[list(TARGETS.values())].mean().mul(100).round(1)
by_type.columns = STAGES
print(by_type.to_string())

print("\n=== 6. Spread of delay rate (%) across states and districts (groups of 30+) ===")
print("(the overall rate is 25%; district groups are small, so some spread is just chance)")
spread = {}
for s in STAGES:
    row = {}
    for col in ["state", "district"]:
        g = df.groupby(col)[TARGETS[s]].agg(["mean", "size"])
        g = g[g["size"] >= 30]["mean"] * 100
        row[f"{col} lowest"] = g.min()
        row[f"{col} highest"] = g.max()
    spread[s] = row
print(pd.DataFrame(spread).T.round(1).to_string())

# ---------- District naming check ----------
print("\n=== 7. District identity check ===")
pairs = df.groupby(["state", "district"])
print(f"Distinct district names: {df['district'].nunique()}")
print(f"Distinct (state, district) pairs: {pairs.ngroups}")
print(f"Distinct district_historical_delay_rate values: "
      f"{df['district_historical_delay_rate'].nunique()}")
print("Most different rates found inside one (state, district) pair: "
      f"{int(pairs['district_historical_delay_rate'].nunique().max())}")

print("\nDone. Copy everything above and paste it into the chat.")
