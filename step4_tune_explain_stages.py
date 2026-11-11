"""
Step 4: Tune, test, explain, and save models for six land-acquisition stages.
Run from the folder containing land_acquisition_projects.csv and preferably stage_results.csv:
    python step4_tune_explain_stages.py

Outputs:
  final_results.csv, importance_by_stage.csv, stage_importance.png
  models/<stage>_random_forest.joblib and models/meta.json

This is a proof of concept on simulated data. Check feature availability at
the intended prediction time before interpreting results as real-world performance.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
N_ITER = 10  # Change to 5 for a faster run.
STAGES = ["notification", "approval", "compensation", "legal_dispute", "possession", "rehabilitation"]
DATA_FILE = "land_acquisition_projects.csv"
STEP3_FILE = "stage_results.csv"

data_path = Path.cwd() / DATA_FILE
if not data_path.exists():
    data_path = Path.home() / DATA_FILE
if not data_path.exists():
    raise SystemExit(f"Could not find {DATA_FILE}. Run this script from land_acq_project.")

df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {data_path}")

# Encode project type and state. District names and project_id are not used.
data = pd.concat(
    [df.drop(columns=["project_type", "state"], errors="ignore"),
     pd.get_dummies(df[["project_type", "state"]], dtype=int)],
    axis=1,
)
data["possession_status_ord"] = df["possession_status"].map(
    {"Not Started": 0, "Partial": 1, "Complete": 2}
)
dummy_cols = [c for c in data.columns if c.startswith(("project_type_", "state_"))]
SET_A = [
    "district_historical_delay_rate", "land_area_hectares",
    "num_affected_families", "num_approvals_pending",
    "stakeholder_responsiveness_score", "legal_dispute_flag",
] + dummy_cols
SET_B = SET_A + [
    "compensation_disbursed_pct", "rehab_progress_pct",
    "possession_status_ord", "days_since_notification",
]
SET_ORDER = ["A: project attributes", "B: + current status", "C: + earlier stage durations"]

candidate_sets = {}
for j, stage in enumerate(STAGES):
    sets = {"A: project attributes": SET_A, "B: + current status": SET_B}
    earlier = [f"{s}_duration_days" for s in STAGES[:j]]
    if earlier:
        sets["C: + earlier stage durations"] = SET_B + earlier
    candidate_sets[stage] = sets

step3_path = Path.cwd() / STEP3_FILE
if step3_path.exists():
    step3_results = pd.read_csv(step3_path)
else:
    print(f"WARNING: {STEP3_FILE} not found; using Set A for every stage.")
    step3_results = pd.DataFrame()

def choose_feature_set(stage):
    available = candidate_sets[stage]
    if step3_results.empty:
        return SET_ORDER[0], available[SET_ORDER[0]]
    subset = step3_results[
        (step3_results["stage"] == stage)
        & (step3_results["model"] == "Random Forest")
        & (step3_results["columns"].isin(available.keys()))
    ]
    if subset.empty:
        return SET_ORDER[0], available[SET_ORDER[0]]
    scores = dict(zip(subset["columns"], subset["ROC-AUC"]))
    best_score = max(scores.values())
    # Prefer the simplest set within 0.005 of the best score.
    for name in SET_ORDER:
        if name in available and name in scores and scores[name] >= best_score - 0.005:
            return name, available[name]
    return SET_ORDER[0], available[SET_ORDER[0]]

def make_logistic():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    )

def make_forest(params=None):
    settings = {
        "n_estimators": 300, "min_samples_leaf": 5,
        "class_weight": "balanced_subsample", "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    if params:
        settings.update(params)
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(**settings))

def evaluate(y, probabilities):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    k = max(1, int(np.ceil(len(y) * 0.25)))
    top_idx = np.argsort(-probabilities)[:k]
    top_y = y[top_idx]
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities),
        "Precision top 25%": float(top_y.mean()),
        "Catch rate top 25%": float(top_y.sum() / max(1, int(y.sum()))),
        "Precision at threshold 0.5": precision_score(y, predicted, zero_division=0),
    }

param_distributions = {
    "randomforestclassifier__n_estimators": [200, 300, 500],
    "randomforestclassifier__max_depth": [None, 8, 12, 18],
    "randomforestclassifier__min_samples_split": [2, 5, 10, 20],
    "randomforestclassifier__min_samples_leaf": [2, 5, 10, 20],
    "randomforestclassifier__max_features": ["sqrt", 0.7, 1.0],
}
results_rows, importance_rows, final_models = [], [], {}
metadata = {
    "data_file": DATA_FILE,
    "target_definition": "Each stage's *_delayed column from simulated data",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "selected_feature_sets": {},
    "feature_columns": {},
    "model_files": {},
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()
tuning_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for stage_index, stage in enumerate(STAGES):
    set_name, columns = choose_feature_set(stage)
    columns = list(dict.fromkeys(columns))
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise SystemExit(f"Missing feature columns for {stage}: {missing}")

    print(f"\n[{stage_index + 1}/6] {stage}: using column set {set_name}", flush=True)
    X, y = data[columns].copy(), df[f"{stage}_delayed"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logistic = make_logistic().fit(X_train, y_train)
    p_logistic = logistic.predict_proba(X_test)[:, 1]

    default_forest = make_forest().fit(X_train, y_train)
    p_default = default_forest.predict_proba(X_test)[:, 1]

    search = RandomizedSearchCV(
        make_forest(), param_distributions, n_iter=N_ITER, scoring="roc_auc",
        cv=tuning_cv, random_state=RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    p_tuned = tuned.predict_proba(X_test)[:, 1]

    for model_name, probs in [
        ("Logistic Regression", p_logistic),
        ("Random Forest (default)", p_default),
        ("Random Forest (tuned)", p_tuned),
    ]:
        results_rows.append({
            "stage": stage, "columns": set_name, "model": model_name,
            **evaluate(y_test, probs),
        })

    print(f"  Test ROC-AUC — Logistic Regression: {roc_auc_score(y_test, p_logistic):.3f}")
    print(f"  Test ROC-AUC — Random Forest (default): {roc_auc_score(y_test, p_default):.3f}")
    print(f"  Test ROC-AUC — Random Forest (tuned): {roc_auc_score(y_test, p_tuned):.3f}")
    print(f"  Best tuning parameters: {search.best_params_}", flush=True)

    # Permutation importance on held-out test set. Shuffle one-hot groups together.
    state_cols = [c for c in columns if c.startswith("state_")]
    type_cols = [c for c in columns if c.startswith("project_type_")]
    groups = []
    for c in columns:
        if c in state_cols:
            if c == state_cols[0]:
                groups.append(("state (all categories)", state_cols))
        elif c in type_cols:
            if c == type_cols[0]:
                groups.append(("project_type (all categories)", type_cols))
        else:
            groups.append((c, [c]))

    baseline_auc = roc_auc_score(y_test, tuned.predict_proba(X_test)[:, 1])
    rng = np.random.RandomState(RANDOM_STATE)
    group_scores = []
    for group_name, group_cols in groups:
        drops = []
        for _ in range(5):
            permuted = X_test.copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, group_cols] = X_test.iloc[order][group_cols].to_numpy()
            score = roc_auc_score(y_test, tuned.predict_proba(permuted)[:, 1])
            drops.append(baseline_auc - score)
        mean_drop = float(np.mean(drops))
        group_scores.append((group_name, mean_drop))
        importance_rows.append({"stage": stage, "feature_group": group_name, "mean_ROC_AUC_drop": mean_drop})

    group_scores.sort(key=lambda item: item[1], reverse=True)
    top = group_scores[:5]
    print("  Top 5 feature groups by permutation importance:")
    for name, score in top:
        print(f"    {name}: ROC-AUC drop = {score:.4f}")

    axes[stage_index].barh([x[0] for x in top][::-1], [x[1] for x in top][::-1])
    axes[stage_index].set_title(stage.replace("_", " ").title())
    axes[stage_index].set_xlabel("Drop in ROC-AUC after shuffling")
    axes[stage_index].grid(axis="x", alpha=0.25)

    # Re-tune using all rows for the model that will later power the dashboard.
    final_search = RandomizedSearchCV(
        make_forest(), param_distributions, n_iter=N_ITER, scoring="roc_auc",
        cv=tuning_cv, random_state=RANDOM_STATE, n_jobs=1, refit=True,
    )
    final_search.fit(X, y)
    final_models[stage] = final_search.best_estimator_
    metadata["selected_feature_sets"][stage] = set_name
    metadata["feature_columns"][stage] = columns
    metadata["model_files"][stage] = f"{stage}_random_forest.joblib"

fig.suptitle("Top feature groups by permutation importance (held-out test data)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("stage_importance.png", dpi=160, bbox_inches="tight")
plt.close(fig)

results = pd.DataFrame(results_rows)
results.round(4).to_csv("final_results.csv", index=False)
importance = pd.DataFrame(importance_rows).sort_values(
    ["stage", "mean_ROC_AUC_drop"], ascending=[True, False]
)
importance.round(5).to_csv("importance_by_stage.csv", index=False)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
for stage, model in final_models.items():
    model_path = models_dir / f"{stage}_random_forest.joblib"
    dump(model, model_path)
    print(f"Saved model: {model_path}")
with open(models_dir / "meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\n=== Test-set results (untouched 20%) ===")
print(results.round(3).to_string(index=False))
print("\n=== Tuned forest only ===")
print(results[results["model"] == "Random Forest (tuned)"].round(3).to_string(index=False))
print("\nDone. Saved: final_results.csv, importance_by_stage.csv, stage_importance.png, and models/.")
