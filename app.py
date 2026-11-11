"""
Step 5 (land acquisition project): Streamlit dashboard.

Enter a project's details in the sidebar and see the delay risk for each of the
six stages, plus a "why" chart for any stage.

Put this file in your land_acq_project folder (next to the models/ folder and
land_acquisition_projects.csv), then run:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Land Acquisition Delay Risk", layout="wide")

HERE = Path(__file__).resolve().parent
STAGES = ["notification", "approval", "compensation",
          "legal_dispute", "possession", "rehabilitation"]
STAGE_LABELS = {
    "notification": "Notification", "approval": "Approval",
    "compensation": "Compensation", "legal_dispute": "Legal dispute",
    "possession": "Possession", "rehabilitation": "Rehabilitation",
}
POSSESSION_ORDER = {"Not Started": 0, "Partial": 1, "Complete": 2}
FRIENDLY = {
    "district_historical_delay_rate": "District's historical delay rate",
    "land_area_hectares": "Land area",
    "num_affected_families": "Affected families",
    "num_approvals_pending": "Approvals pending",
    "stakeholder_responsiveness_score": "Stakeholder responsiveness",
    "legal_dispute_flag": "Legal dispute filed",
    "compensation_disbursed_pct": "Compensation disbursed %",
    "rehab_progress_pct": "Rehabilitation progress %",
    "possession_status": "Possession status",
    "days_since_notification": "Days since notification",
    "project_type": "Project type",
    "state": "State",
}
# (min, typical, max) used only if land_acquisition_projects.csv is not found
FALLBACK_RANGES = {
    "district_historical_delay_rate": (0.05, 0.30, 0.55),
    "land_area_hectares": (0.5, 15.0, 500.0),
    "num_affected_families": (1, 115, 5000),
    "num_approvals_pending": (0, 2, 8),
    "stakeholder_responsiveness_score": (1.0, 3.0, 5.0),
    "compensation_disbursed_pct": (0.0, 50.0, 100.0),
    "rehab_progress_pct": (0.0, 30.0, 100.0),
    "days_since_notification": (0, 200, 1500),
}


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    models = {}
    for stage in STAGES:
        for name in (f"{stage}_random_forest.joblib", f"{stage}.joblib"):
            path = HERE / "models" / name
            if path.exists():
                obj = joblib.load(path)
                models[stage] = obj["model"] if isinstance(obj, dict) else obj
                break
    return models


@st.cache_data
def load_reference():
    for path in (HERE / "land_acquisition_projects.csv",
                 Path.cwd() / "land_acquisition_projects.csv"):
        if path.exists():
            return pd.read_csv(path)
    return None


@st.cache_data
def load_table(name):
    path = HERE / name
    return pd.read_csv(path) if path.exists() else None


models = load_models()
REF = load_reference()

st.title("Land Acquisition Delay Risk")
st.caption("Prototype trained on simulated data. Scores show relative risk, "
           "not exact probabilities.")

if not models:
    st.error("No models found. Put this file next to your models/ folder "
             "(it should contain files like notification_random_forest.joblib).")
    st.stop()

MODEL_COLS = {s: list(m.feature_names_in_) for s, m in models.items()}
ALL_COLS = sorted({c for cols in MODEL_COLS.values() for c in cols})
PROJECT_TYPES = sorted({c[len("project_type_"):] for c in ALL_COLS if c.startswith("project_type_")})
STATES = sorted({c[len("state_"):] for c in ALL_COLS if c.startswith("state_")})


def col_range(col):
    """(min, typical, max) for a numeric column."""
    if REF is not None and col in REF and REF[col].notna().any():
        s = REF[col]
        return float(s.min()), float(s.median()), float(s.max())
    if col in FALLBACK_RANGES:
        return tuple(float(v) for v in FALLBACK_RANGES[col])
    return 30.0, 380.0, 1700.0  # stage durations


def typical_inputs():
    """A 'typical project', used as the comparison point for the why chart."""
    ref = {c: col_range(c)[1] for c in ALL_COLS
           if not c.startswith(("project_type_", "state_")) and c != "possession_status_ord"}
    ref["legal_dispute_flag"] = 0
    if REF is not None:
        ref["project_type"] = REF["project_type"].mode().iloc[0]
        ref["state"] = REF["state"].mode().iloc[0]
        ref["possession_status"] = REF["possession_status"].mode().iloc[0]
    else:
        ref["project_type"] = PROJECT_TYPES[0] if PROJECT_TYPES else ""
        ref["state"] = STATES[0] if STATES else ""
        ref["possession_status"] = "Partial"
    return ref


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Project details")
inputs = {}

if PROJECT_TYPES:
    inputs["project_type"] = st.sidebar.selectbox("Project type", PROJECT_TYPES)
if STATES:
    inputs["state"] = st.sidebar.selectbox("State", STATES)


def number_box(col, label, step, as_int=False):
    lo, med, hi = col_range(col)
    if as_int:
        return st.sidebar.number_input(label, int(lo), int(hi), int(med), int(step))
    return st.sidebar.number_input(label, lo, hi, med, float(step))


if "land_area_hectares" in ALL_COLS:
    inputs["land_area_hectares"] = number_box("land_area_hectares", "Land area (hectares)", 1.0)
if "num_affected_families" in ALL_COLS:
    inputs["num_affected_families"] = number_box("num_affected_families", "Affected families", 10, True)
if "num_approvals_pending" in ALL_COLS:
    lo, med, hi = col_range("num_approvals_pending")
    inputs["num_approvals_pending"] = st.sidebar.slider(
        "Approvals pending", int(lo), int(hi), int(med))
if "district_historical_delay_rate" in ALL_COLS:
    lo, med, hi = col_range("district_historical_delay_rate")
    inputs["district_historical_delay_rate"] = st.sidebar.slider(
        "District's historical delay rate", round(lo, 2), round(hi, 2), round(med, 2), 0.01)
if "stakeholder_responsiveness_score" in ALL_COLS:
    lo, med, hi = col_range("stakeholder_responsiveness_score")
    unknown = st.sidebar.checkbox("Stakeholder score not available")
    score = st.sidebar.slider("Stakeholder responsiveness (1-5)", lo, hi, med, 0.1,
                              disabled=unknown)
    inputs["stakeholder_responsiveness_score"] = None if unknown else score
if "legal_dispute_flag" in ALL_COLS:
    inputs["legal_dispute_flag"] = int(
        st.sidebar.selectbox("Legal dispute filed?", ["No", "Yes"]) == "Yes")

if any(c in ALL_COLS for c in ("compensation_disbursed_pct", "rehab_progress_pct",
                               "possession_status_ord", "days_since_notification")):
    st.sidebar.subheader("Current status")
if "compensation_disbursed_pct" in ALL_COLS:
    lo, med, hi = col_range("compensation_disbursed_pct")
    unknown = st.sidebar.checkbox("Compensation % not available")
    pct = st.sidebar.slider("Compensation disbursed (%)", 0.0, 100.0, med, 1.0,
                            disabled=unknown)
    inputs["compensation_disbursed_pct"] = None if unknown else pct
if "rehab_progress_pct" in ALL_COLS:
    lo, med, hi = col_range("rehab_progress_pct")
    inputs["rehab_progress_pct"] = st.sidebar.slider(
        "Rehabilitation progress (%)", 0.0, 100.0, med, 1.0)
if "possession_status_ord" in ALL_COLS:
    inputs["possession_status"] = st.sidebar.selectbox(
        "Possession status", list(POSSESSION_ORDER), index=1)
if "days_since_notification" in ALL_COLS:
    inputs["days_since_notification"] = number_box(
        "days_since_notification", "Days since notification", 10, True)

duration_cols = [f"{s}_duration_days" for s in STAGES if f"{s}_duration_days" in ALL_COLS]
if duration_cols:
    with st.sidebar.expander("Finished stages (optional)"):
        st.caption("Tick a stage if it has finished and enter how long it took. "
                   "Unticked stages are treated as a typical project.")
        for col in duration_cols:
            stage = col.replace("_duration_days", "")
            lo, med, hi = col_range(col)
            done = st.checkbox(f"{STAGE_LABELS[stage]} finished", key=f"done_{stage}")
            days = st.number_input(f"{STAGE_LABELS[stage]} duration (days)", int(lo), int(hi),
                                   int(med), 10, key=f"dur_{stage}", disabled=not done)
            inputs[col] = float(days) if done else None

st.sidebar.header("Risk bands")
HIGH = st.sidebar.slider("High risk from", 0.0, 1.0, 0.60, 0.01)
MEDIUM = st.sidebar.slider("Medium risk from", 0.0, 1.0, 0.40, 0.01)


# ---------------------------------------------------------------- predicting
def build_row(values, cols):
    row = {}
    for c in cols:
        if c.startswith("project_type_"):
            row[c] = int(c == f"project_type_{values.get('project_type')}")
        elif c.startswith("state_"):
            row[c] = int(c == f"state_{values.get('state')}")
        elif c == "possession_status_ord":
            row[c] = POSSESSION_ORDER.get(values.get("possession_status"), np.nan)
        else:
            v = values.get(c)
            row[c] = np.nan if v is None else v
    return pd.DataFrame([row], columns=cols)


def risk(stage, values):
    X = build_row(values, MODEL_COLS[stage])
    return float(models[stage].predict_proba(X)[0, 1])


def input_keys(stage):
    keys = set()
    for c in MODEL_COLS[stage]:
        if c.startswith("project_type_"):
            keys.add("project_type")
        elif c.startswith("state_"):
            keys.add("state")
        elif c == "possession_status_ord":
            keys.add("possession_status")
        else:
            keys.add(c)
    return keys


def pretty(key):
    if key in FRIENDLY:
        return FRIENDLY[key]
    if key.endswith("_duration_days"):
        return STAGE_LABELS.get(key.replace("_duration_days", ""), key) + " duration"
    return key.replace("_", " ").capitalize()


def band(p):
    if p >= HIGH:
        return "High", "🔴"
    if p >= MEDIUM:
        return "Medium", "🟡"
    return "Low", "🟢"


scores = {s: risk(s, inputs) for s in STAGES if s in models}

# ---------------------------------------------------------------- results
st.subheader("Delay risk by stage")
cols = st.columns(len(scores))
for col, (stage, p) in zip(cols, scores.items()):
    label, icon = band(p)
    col.metric(STAGE_LABELS[stage], f"{p:.0%}")
    col.write(f"{icon} {label}")

worst = max(scores, key=scores.get)
label, icon = band(scores[worst])
st.info(f"Highest-risk stage: **{STAGE_LABELS[worst]}** ({scores[worst]:.0%}, {label.lower()} risk).")

chart = pd.Series({STAGE_LABELS[s]: p for s, p in scores.items()}, name="Risk score")
st.bar_chart(chart)

# ---------------------------------------------------------------- why
st.subheader("Why this score?")
stage_choice = st.selectbox(
    "Stage to explain", list(scores), index=list(scores).index(worst),
    format_func=lambda s: STAGE_LABELS[s])

typical = typical_inputs()
base = scores[stage_choice]
effects = {}
for key in input_keys(stage_choice):
    alt = dict(inputs)
    alt[key] = typical.get(key)
    effects[pretty(key)] = base - risk(stage_choice, alt)
effects = pd.Series(effects).sort_values()

st.caption("Each bar shows how much this project's risk changes compared with a "
           "typical project on that one detail. Positive = pushes risk up.")
st.bar_chart(effects)

up = effects[effects > 0.005].sort_values(ascending=False).head(3)
down = effects[effects < -0.005].head(3)
if len(up):
    st.write("**Pushing risk up:** " + ", ".join(f"{k} (+{v:.0%})" for k, v in up.items()))
if len(down):
    st.write("**Pulling risk down:** " + ", ".join(f"{k} ({v:.0%})" for k, v in down.items()))

# ---------------------------------------------------------------- extras
imp = load_table("importance_by_stage.csv")
with st.expander("What matters most overall for this stage"):
    if imp is not None:
        drop_col = next((c for c in imp.columns if "drop" in c.lower()), None)
        feat_col = next((c for c in imp.columns if c.lower() in ("feature", "column", "group")), None)
        if drop_col and feat_col and "stage" in imp.columns:
            sub = imp[imp["stage"] == stage_choice].sort_values(drop_col, ascending=False).head(8)
            st.dataframe(sub[[feat_col, drop_col]], hide_index=True)
        else:
            st.dataframe(imp.head(20), hide_index=True)
    else:
        st.write("importance_by_stage.csv not found.")

res = load_table("final_results.csv")
with st.expander("How accurate are the models? (tested on unseen projects)"):
    if res is not None:
        model_col = next((c for c in res.columns if c.lower() == "model"), None)
        shown = res[res[model_col].astype(str).str.contains("tuned", case=False)] if model_col else res
        st.dataframe(shown, hide_index=True)
        st.caption("ROC-AUC: 0.5 = coin flip, 1.0 = perfect. Data is simulated, so "
                   "this shows the method works, not real-world accuracy.")
    else:
        st.write("final_results.csv not found.")
