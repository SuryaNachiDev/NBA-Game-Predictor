# NBA-Game-Predictor

Predicts a team's wins per 82 games from its season box-score stats, using 300 team-seasons from 2014-15 through 2023-24.

## How it works

`nba_wins_pipeline.py` turns raw team totals into 16 per-possession features (offensive and defensive rating, pace, effective FG%, true shooting, rebound and turnover rates, and others), then centers each feature on its season average so the model compares teams to their league that year.

It trains three models and scores each with 5-fold cross-validation grouped by season, so no season appears in both training and test data:

| Model | MAE (wins) | R² |
|---|---|---|
| Net rating only (linear baseline) | 2.52 | 0.929 |
| Ridge regression | 2.55 | 0.928 |
| Random forest | 2.90 | 0.898 |
| Gradient boosting | 2.85 | 0.905 |

The one-feature baseline wins. Offensive minus defensive rating already captures almost everything the other 14 features add, and the tree models overfit on 300 rows. Permutation importance on the Ridge model agrees: defensive rating, offensive rating and shooting efficiency carry the predictions, and pace, fouls and assist rate contribute close to nothing.

## Run it

```
pip install pandas numpy scikit-learn matplotlib
python nba_wins_pipeline.py nba_team_stats_00_to_23.csv
```

The script writes `predictions.csv`, `model_features.csv` and `results.png` (actual vs. predicted wins and a feature-importance chart).

## Limits

The model uses same-season stats, so it explains a season's win total after the fact. It does not forecast individual games or future seasons. A forecasting version would train on stats from earlier seasons or games only.
