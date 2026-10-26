import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2024)]

features = {
    "off_rtg": "efficiency",
    "def_rtg": "efficiency",
    "pace": "pace",
    "efg": "shooting",
    "ts": "shooting",
    "fg3_pct": "shooting",
    "ft_pct": "shooting",
    "fg3a_rate": "shooting",
    "ft_rate": "shooting",
    "oreb": "rebounding",
    "dreb": "rebounding",
    "tov": "turnovers",
    "ast_rate": "playmaking",
    "stl": "defense",
    "blk": "defense",
    "pf": "fouls",
}
cols = list(features)


def load_data(path):
    df = pd.read_csv(path)
    df = df[df["season"].isin(SEASONS)].copy()

    assert len(df) == 300
    assert not df.duplicated(["Team", "season"]).any()
    assert (df["wins"] + df["losses"] == df["games_played"]).all()
    assert df.notna().all().all()
    return df


def build_features(df):
    fga = df["field_goals_attempted"]
    fta = df["free_throw_attempted"]
    poss = fga - df["offensive_rebounds"] + df["turnovers"] + 0.44 * fta
    opp_pts = df["points"] - df["plus_minus"]
    mins_per_game = df["Min"] / df["games_played"]

    def per100(col):
        return 100 * df[col] / poss

    X = pd.DataFrame({
        "off_rtg": 100 * df["points"] / poss,
        "def_rtg": 100 * opp_pts / poss,
        "pace": 240 * (poss / df["games_played"]) / mins_per_game,
        "efg": (df["field_goals_made"] + 0.5 * df["three_pointers_made"]) / fga,
        "ts": df["points"] / (2 * (fga + 0.44 * fta)),
        "fg3_pct": df["three_point_percentage"] / 100,
        "ft_pct": df["free_throw_percentage"] / 100,
        "fg3a_rate": df["three_pointers_attempted"] / fga,
        "ft_rate": fta / fga,
        "oreb": per100("offensive_rebounds"),
        "dreb": per100("defensive_rebounds"),
        "tov": per100("turnovers"),
        "ast_rate": df["assists"] / df["field_goals_made"],
        "stl": per100("steals"),
        "blk": per100("blocks"),
        "pf": per100("personal_fouls"),
    })

    X = X - X.groupby(df["season"]).transform("mean")

    out = df[["season", "Team", "wins", "games_played"]].join(X)
    out["wins_82"] = df["win_percentage"] * 82
    return out


def make_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                              random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.03,
                                                      max_depth=3, subsample=0.8,
                                                      random_state=42),
    }


def oof_predict(model, X, y, groups):
    preds = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        model.fit(X[train], y[train])
        preds[test] = model.predict(X[test])
    return preds


def report(name, y, preds):
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{name:<18} MAE {mae:.2f}   R2 {r2:.3f}")
    return mae


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "nba_team_stats_00_to_23.csv"
    data = build_features(load_data(path))

    X = data[cols].values
    y = data["wins_82"].values
    seasons = data["season"].values
    print(f"{len(data)} team-seasons, {len(SEASONS)} seasons, {len(cols)} features\n")

    net = (data["off_rtg"] - data["def_rtg"]).values.reshape(-1, 1)
    report("net rating only", y, oof_predict(LinearRegression(), net, y, seasons))

    results = {}
    for name, model in make_models().items():
        preds = oof_predict(model, X, y, seasons)
        results[name] = (report(name, y, preds), preds)

    best = min(results, key=lambda k: results[k][0])
    best_preds = results[best][1]
    print(f"\nbest: {best}\n")

    model = make_models()[best].fit(data[cols], y)
    perm = permutation_importance(model, data[cols], y, n_repeats=50, random_state=42,
                                  scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({
        "feature": cols,
        "group": [features[c] for c in cols],
        "mae_increase": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("mae_increase", ascending=False)

    print(imp.to_string(index=False, float_format="%.3f"))
    print()
    print(imp.groupby("group")["mae_increase"].sum().sort_values(ascending=False)
          .to_string(float_format="%.3f"))

    preds_df = data[["season", "Team", "wins", "games_played", "wins_82"]].copy()
    preds_df["predicted"] = best_preds.round(1)
    preds_df["error"] = (preds_df["predicted"] - preds_df["wins_82"]).round(1)
    preds_df.to_csv("predictions.csv", index=False)
    data[["season", "Team", "wins_82"] + cols].to_csv("model_features.csv", index=False)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.scatter(y, best_preds, s=14, alpha=0.7)
    left.plot([10, 70], [10, 70], "k--", lw=1)
    left.set_xlabel("actual wins (per 82 games)")
    left.set_ylabel("predicted wins")
    left.set_title(f"{best}, out-of-fold")

    top = imp.head(10).iloc[::-1]
    right.barh(top["feature"], top["mae_increase"], xerr=top["std"])
    right.set_xlabel("MAE increase when shuffled")
    right.set_title("feature importance")

    fig.tight_layout()
    fig.savefig("results.png", dpi=150)
    print("\nsaved predictions.csv, model_features.csv, results.png")


if __name__ == "__main__":
    main()
