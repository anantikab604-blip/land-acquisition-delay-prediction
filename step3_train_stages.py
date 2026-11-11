"""
Step 3 (land acquisition project): train a model for each of the six stages.

For every stage we predict "will this stage be delayed?" and compare:
    - Logistic Regression (simple baseline)
    - Random Forest
on three growing sets of columns:
    A: project attributes only
    B: A + current-status columns (progress and days so far)
    C: B + the durations of EARLIER stages (not the stage itself, not later ones)

Never used as inputs: project_id, district names, all *_risk_score columns, and
any duration or delayed flag of the same or later stages.

Run from the folder that holds land_acquisition_projects.csv:
    python step3_train_stages.py

Takes a few minutes. Output: printed tables + stage_results.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

RANDOM_STATE = 42
FOLDS = 5

NAME = "land_acquisition_projects.csv"
path = next((p for p in [Path.cwd() / NAME, Path.home() / NAME] if p.exists()), None)
if path is None:
    raise SystemExit(f"Could not find {NAME} in this folder or your home folder.")
df = pd.read_csv(path)

STAGES = ["notification", "approval", "compensation",
          "legal_dispute", "possession", "rehabilitation"]

# ---------- Build the input table ----------
data = pd.concat(
    [df.drop(columns=["project_type", "state"]),
     pd.get_dummies(df[["project_type", "state"]], dtype=int)],
    axis=1,
)
data["possession_status_ord"] = df["possession_status"].map(
    {"Not Started": 0, "Partial": 1, "Complete": 2})

dummy_cols = [c for c in data.columns if c.startswith(("project_type_", "state_"))]
SET_A = ["district_historical_delay_rate", "land_area_hectares",
         "num_affected_families", "num_approvals_pending",
         "stakeholder_responsiveness_score", "legal_dispute_flag"] + dummy_cols
SET_B = SET_A + ["compensation_disbursed_pct", "rehab_progress_pct",
                 "possession_status_ord", "days_since_notification"]


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RANDOM_STATE)),
    }


def precision_in_top_quarter(y, p):
    """Flag the 25% of projects the model rates riskiest.
    What share of them are truly delayed? (Random guessing gives about 25%.)"""
    k = int(len(y) * 0.25)
    top = np.argsort(-p)[:k]
    return float(np.asarray(y)[top].mean())


cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
rows = []
for j, stage in enumerate(STAGES):
    y = df[f"{stage}_delayed"]
    earlier = [f"{s}_duration_days" for s in STAGES[:j]]
    sets = {"A: project attributes": SET_A, "B: + current status": SET_B}
    if earlier:
        sets["C: + earlier stage durations"] = SET_B + earlier
    print(f"Training models for stage {j + 1}/6: {stage} ...", flush=True)
    for set_name, cols in sets.items():
        X = data[cols]
        for model_name, model in make_models().items():
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            rows.append({
                "stage": stage,
                "columns": set_name,
                "model": model_name,
                "ROC-AUC": roc_auc_score(y, p),
                "PR-AUC": average_precision_score(y, p),
                "Precision in top 25%": precision_in_top_quarter(y, p),
            })

res = pd.DataFrame(rows)
res.round(3).to_csv("stage_results.csv", index=False)

print(f"\n=== 1. All results ({FOLDS}-fold cross-validation on all {len(df)} projects) ===")
print("Random guessing: ROC-AUC 0.50, PR-AUC 0.25, precision in top 25% = 0.25")
print(res.round(3).to_string(index=False))

print("\n=== 2. Random Forest ROC-AUC: stages vs column sets ===")
rf = res[res["model"] == "Random Forest"]
print(rf.pivot(index="stage", columns="columns", values="ROC-AUC")
        .reindex(STAGES).round(3).to_string())

print("\n=== 3. Random Forest minus Logistic Regression (ROC-AUC, best column set) ===")
best = res.pivot_table(index="stage", columns="model", values="ROC-AUC", aggfunc="max")
best["RF advantage"] = best["Random Forest"] - best["Logistic Regression"]
print(best.reindex(STAGES).round(3).to_string())

print("\nDone. Saved stage_results.csv. Copy everything above and paste it into the chat.")