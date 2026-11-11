"""
Step 6 (land acquisition project): build RESULTS.md from your own output files.

Reads land_acquisition_projects.csv, final_results.csv and importance_by_stage.csv
and writes RESULTS.md with tables you can paste straight into slides or a report.
Every number comes from your files, so nothing has to be copied by hand.

Run from your land_acq_project folder:
    python make_report.py
"""
from pathlib import Path

import pandas as pd

STAGES = ["notification", "approval", "compensation",
          "legal_dispute", "possession", "rehabilitation"]


def read(name):
    path = Path(name)
    return pd.read_csv(path) if path.exists() else None


def md_table(df):
    """Turn a DataFrame into a Markdown table."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage_order(df):
    df = df.copy()
    df["_o"] = df["stage"].map({s: i for i, s in enumerate(STAGES)})
    return df.sort_values("_o").drop(columns="_o")


data = read("land_acquisition_projects.csv")
res = read("final_results.csv")
imp = read("importance_by_stage.csv")
if res is None:
    raise SystemExit("final_results.csv not found. Run Step 4 first.")

out = ["# Results summary", "",
       "_Generated automatically by make_report.py from final_results.csv, "
       "importance_by_stage.csv and the dataset._", ""]

# ---------------- Data ----------------
if data is not None:
    out += ["## Data", "",
            f"- {len(data):,} simulated projects, {data['state'].nunique()} states, "
            f"{data['project_type'].nunique()} project types",
            "- Six stages: " + ", ".join(STAGES),
            "- Each stage is labelled delayed when its duration is in the slowest 25%", ""]

# ---------------- Tuned models ----------------
model_col = next(c for c in res.columns if c.lower() == "model")
auc_col = next((c for c in res.columns if "roc" in c.lower()), None)
tuned = res[res[model_col].astype(str).str.contains("tuned", case=False)]
tuned = stage_order(tuned.drop(columns=model_col))

out += ["## Headline", ""]
if auc_col is not None and len(tuned):
    best = tuned.loc[tuned[auc_col].idxmax()]
    worst = tuned.loc[tuned[auc_col].idxmin()]
    out += [f"- Average ROC-AUC across the six stages (tuned Random Forest, unseen 20% of projects): "
            f"**{tuned[auc_col].mean():.3f}** (0.5 = coin flip, 1.0 = perfect)",
            f"- Easiest stage to predict: **{best['stage']}** ({best[auc_col]:.3f})",
            f"- Hardest stage to predict: **{worst['stage']}** ({worst[auc_col]:.3f})", ""]

out += ["## Tuned model, each stage (unseen 20% of projects)", "", md_table(tuned), ""]

# ---------------- Model comparison ----------------
if auc_col is not None:
    pivot = res.pivot_table(index="stage", columns=model_col, values=auc_col).reset_index()
    out += ["## ROC-AUC by model", "", md_table(stage_order(pivot)), ""]
    cols = [c for c in pivot.columns if c != "stage"]
    lr = next((c for c in cols if "logistic" in c.lower()), None)
    rf = next((c for c in cols if "tuned" in c.lower()), None)
    if lr and rf:
        wins = int((pivot[rf] > pivot[lr]).sum())
        out += [f"The tuned Random Forest beats Logistic Regression in {wins} of "
                f"{len(pivot)} stages.", ""]

# ---------------- Drivers ----------------
if imp is not None and "stage" in imp.columns:
    drop_col = next((c for c in imp.columns if "drop" in c.lower()), None)
    feat_col = next((c for c in imp.columns if c.lower() in
                     ("feature", "column", "group", "variable")), None)
    if drop_col and feat_col:
        out += ["## Top drivers per stage", "",
                "Importance = how much the ROC-AUC falls when that column is shuffled "
                "on unseen projects.", ""]
        for stage in STAGES:
            sub = imp[imp["stage"] == stage].sort_values(drop_col, ascending=False).head(3)
            if len(sub):
                items = ", ".join(f"{r[feat_col]} ({r[drop_col]:.3f})" for _, r in sub.iterrows())
                out.append(f"- **{stage}**: {items}")
        out.append("")

# ---------------- Limitations ----------------
out += ["## How to read these numbers", "",
        "- The data is simulated by generatingData.py, so the results show how well the "
        "method recovers the patterns built into the simulator. They are not a claim "
        "about real-world accuracy.",
        "- \"Delayed\" means the slowest 25% of durations for that stage (a percentile "
        "rule), not a statutory deadline.",
        "- The test set is about 1,000 projects, so ROC-AUC differences of roughly 0.02 "
        "or less between models are within noise.",
        "- Scores come from class-weighted forests. Treat them as relative risk, not "
        "exact probabilities.",
        "- Columns that describe current progress (for example compensation disbursed) "
        "assume the dashboard user knows the project's current status.",
        ""]

Path("RESULTS.md").write_text("\n".join(out), encoding="utf-8")
print("Saved RESULTS.md")
print("\n".join(out[:25]))
