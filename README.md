# Land Acquisition Project Delay Prediction

**Project-Based Learning (PBL) Project**

Inspired by Smart India Hackathon 2026 Problem Statement 26017.

**Developer:** Anantika Sisodiya  
**Project Type:** Individual Project
A prototype that predicts, for each stage of a land acquisition project, whether that
stage is likely to be delayed, so officials can step in early. It ships with a Streamlit
dashboard where you enter a project's details and see the risk for all six stages.

> The data is simulated. Results show that the method works, not how accurate it would be
> on real records.

## The problem

Land acquisition projects pass through six stages: notification, approval, compensation,
legal dispute, possession and rehabilitation. A delay in any stage holds up the whole
project. Teams usually find out about a delay after it has happened.

## What it does

- Predicts delay risk separately for each of the six stages
- Shows risk as Low / Medium / High, plus a chart of the stage most likely to slip
- Explains each score by showing which details push the risk up or down compared with a
  typical project
- Reports how accurate the models were on projects they had never seen

## Approach

1. **Simulated data:** `generatingData.py` creates 5,000 projects with project details,
   current status, and the duration of each stage.
2. **Target:** a stage counts as delayed when its duration is in the slowest 25%.
3. **Leakage check:** `step2_leakage_scan.py` measures how strongly each column is linked
   to each delay. Risk scores, and the durations of the same or later stages, are
   excluded because they reveal the answer.
4. **Models:** for each stage, Logistic Regression is the baseline and a Random Forest is
   the main model. Three column sets are compared: project attributes, plus current
   status, plus earlier stages' durations. The simplest set that scores nearly as well as
   the best is used.
5. **Testing:** the forest is tuned with cross-validation on 80% of projects and scored
   on the untouched 20%.
6. **Explaining:** permutation importance (shuffle a column, see how much the score
   drops) and a per-project "why" chart in the dashboard.

Results are written to `RESULTS.md` by `make_report.py`.

## How to run

```
pip install -r requirements.txt

python generatingData.py        # creates land_acquisition_projects.csv
python step1_inspect_data.py    # checks the data
python step2_leakage_scan.py    # finds which columns would leak the answer
python step3_train_stages.py    # compares models and column sets
python step4_tune_explain_stages.py   # tunes, tests, saves models/
python make_report.py           # writes RESULTS.md
streamlit run app.py            # opens the dashboard
```

## Files

| File | Purpose |
| --- | --- |
| `generatingData.py` | Creates the simulated dataset |
| `step1_inspect_data.py` | Data inspection |
| `step2_leakage_scan.py` | Leakage and signal scan |
| `step3_train_stages.py` | Baseline vs Random Forest, three column sets |
| `step4_tune_explain_stages.py` | Tuning, testing, importance, saves models |
| `make_report.py` | Builds `RESULTS.md` |
| `app.py` | Streamlit dashboard |
| `models/` | One trained model per stage |

## Limitations

- Simulated data, so accuracy on real projects is unknown.
- "Delayed" is a percentile rule, not a legal deadline.
- The test set is about 1,000 projects, so small differences between models are noise.
- Scores are relative risk, not exact probabilities.
- Current-status inputs assume the user knows how far the project has progressed.

## Next steps

- Replace the simulated data with real project records
- Use the statutory timeline for each stage as the delay definition
- Add per-project explanations with SHAP
- Deploy the dashboard for district officers
