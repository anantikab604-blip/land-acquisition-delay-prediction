# Results summary

_Generated automatically by make_report.py from final_results.csv, importance_by_stage.csv and the dataset._

## Data

- 5,000 simulated projects, 14 states, 6 project types
- Six stages: notification, approval, compensation, legal_dispute, possession, rehabilitation
- Each stage is labelled delayed when its duration is in the slowest 25%

## Headline

- Average ROC-AUC across the six stages (tuned Random Forest, unseen 20% of projects): **0.761** (0.5 = coin flip, 1.0 = perfect)
- Easiest stage to predict: **legal_dispute** (0.919)
- Hardest stage to predict: **notification** (0.696)

## Tuned model, each stage (unseen 20% of projects)

| stage | columns | ROC-AUC | PR-AUC | Precision top 25% | Catch rate top 25% | Precision at threshold 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| notification | A: project attributes | 0.696 | 0.401 | 0.424 | 0.424 | 0.376 |
| approval | A: project attributes | 0.708 | 0.448 | 0.440 | 0.440 | 0.396 |
| compensation | B: + current status | 0.697 | 0.394 | 0.428 | 0.428 | 0.381 |
| legal_dispute | A: project attributes | 0.919 | 0.758 | 0.784 | 0.784 | 0.763 |
| possession | B: + current status | 0.838 | 0.604 | 0.640 | 0.640 | 0.579 |
| rehabilitation | C: + earlier stage durations | 0.711 | 0.406 | 0.428 | 0.428 | 0.416 |

## ROC-AUC by model

| stage | Logistic Regression | Random Forest (default) | Random Forest (tuned) |
| --- | --- | --- | --- |
| notification | 0.711 | 0.678 | 0.696 |
| approval | 0.707 | 0.710 | 0.708 |
| compensation | 0.706 | 0.697 | 0.697 |
| legal_dispute | 0.929 | 0.924 | 0.919 |
| possession | 0.838 | 0.841 | 0.838 |
| rehabilitation | 0.713 | 0.711 | 0.711 |

The tuned Random Forest beats Logistic Regression in 1 of 6 stages.

## How to read these numbers

- The data is simulated by generatingData.py, so the results show how well the method recovers the patterns built into the simulator. They are not a claim about real-world accuracy.
- "Delayed" means the slowest 25% of durations for that stage (a percentile rule), not a statutory deadline.
- The test set is about 1,000 projects, so ROC-AUC differences of roughly 0.02 or less between models are within noise.
- Scores come from class-weighted forests. Treat them as relative risk, not exact probabilities.
- Columns that describe current progress (for example compensation disbursed) assume the dashboard user knows the project's current status.
