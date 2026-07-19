# -*- coding: utf-8 -*-
"""
train.py -- 4th Down Decision Tool: offline training pipeline.

Pulls nflverse play-by-play via nfl_data_py, builds every model and lookup
table the app needs, and writes them to ./models and ./data. Run this locally
whenever you want to refresh the data (e.g. after a new season), then commit the
regenerated artifacts. The deployed app only LOADS them -- it never pulls data
or trains. To fold in a season, add it to SEASONS and re-run: python train.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
"""
SEASONS is the one line to edit when folding in a new year. KICKER_ANCHOR is the
league reference max FG distance (yd) the accuracy curve is scaled against.
XGB_PARAMS is the tuned XGBoost used for the win probability model, chosen by a
5-fold CV bake-off. GO_FEATURES drive the conversion model. WP_FEATURES drive
the win probability model, which the app evaluates at each resulting game state.
"""
SEASONS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]
SEASON_TYPE = "REG"
KICKER_ANCHOR = 65
FG_MIN_DIST, FG_MAX_DIST = 18, 70

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

GO_FEATURES = ["ydstogo", "qtr", "half_seconds_remaining",
               "yardline_100", "score_differential"]

WP_FEATURES = ["yardline_100", "score_differential",
               "game_seconds_remaining", "half_seconds_remaining"]

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbosity=0)

KEEP_COLS = [
    "posteam", "posteam_type", "defteam", "side_of_field", "yardline_100",
    "half_seconds_remaining", "game_seconds_remaining", "qtr", "down",
    "ydstogo", "ydsnet", "yards_gained", "epa", "ep", "wp", "def_wp", "wpa",
    "vegas_wpa", "pass_attempt", "season", "cp", "cpoe", "goal_to_go",
    "air_yards", "field_goal_attempt", "field_goal_result", "kick_distance",
    "score_differential", "no_score_prob", "opp_fg_prob", "opp_td_prob",
    "fg_prob", "td_prob", "punt_blocked", "punt_inside_twenty", "touchback",
    "punt_attempt", "return_yards", "fourth_down_converted", "touchdown",
    "season_type",
]


#------------------------------------------------------------------------------
# Data loading
#------------------------------------------------------------------------------
"""
Load regular-season play-by-play and split it into the working frames. `firsts`
is every fresh first-and-10 (and goal-to-go) snap -- the baseline for a team
that has just taken over a set of downs. `decisiondata` is all 4th-down and FG
plays, further split into go / punt / field-goal attempts.
"""
def load_pbp(seasons):
    import nfl_data_py as nfl
    print(f"Loading play-by-play for seasons {list(seasons)} ...")
    return nfl.import_pbp_data(list(seasons), downcast=True)


def build_datasets(data):
    data = data[data["season_type"] == SEASON_TYPE].copy()

    firsts = data[
        ((data["down"] == 1.0) & (data["ydstogo"] == 10))
        | ((data["down"] == 1.0) & (data["goal_to_go"] == 1.0) & (data["ydstogo"] <= 10))
    ]

    keep = [c for c in KEEP_COLS if c in data.columns]
    decisiondata = data[(data["down"] == 4.0) | (data["field_goal_attempt"] == 1.0)][keep]

    dd4 = decisiondata[decisiondata["down"] == 4]
    go_attempts = dd4[(dd4["field_goal_attempt"] != 1.0) & (dd4["punt_attempt"] != 1.0)]
    punt_attempts = dd4[dd4["punt_attempt"] == 1.0]
    fg_attempts = decisiondata[decisiondata["field_goal_attempt"] == 1.0]

    print(f"  firsts={len(firsts)}  decisiondata(4th)={len(dd4)}  "
          f"go={len(go_attempts)}  punt={len(punt_attempts)}  fg={len(fg_attempts)}")
    return firsts, decisiondata, dd4, go_attempts, punt_attempts, fg_attempts


#------------------------------------------------------------------------------
# Punt model
#------------------------------------------------------------------------------
"""
Empirical punt outcomes averaged by the punting team's field position
(yardline_100): EPA, WPA, opponent next-score odds, touchback and inside-20
rates, plus the average resulting opponent field position (opp_start_yl100).
That last column lets the app map a punter's expected end-yardline back to an
equivalent punt value, so punter range shifts the punt's EPA/WPA, not just the
display. opp_start_yl100 is the opponent's distance to our end zone after the
punt: 100 - yardline_100 + net for returns/downs, 80 for touchbacks.
"""
def build_punt_model(punt_attempts):
    p = punt_attempts.copy()
    net = p["kick_distance"] - p["return_yards"].fillna(0)
    downed_start = 100 - p["yardline_100"] + net
    p["opp_start_yl100"] = np.where(p["touchback"] == 1, 80, downed_start).clip(1, 99)

    summary = (
        p.groupby("yardline_100")
        .agg(
            punt_epa=("epa", "mean"),
            punt_wpa=("wpa", "mean"),
            opp_td_prob=("opp_td_prob", "mean"),
            opp_fg_prob=("opp_fg_prob", "mean"),
            opp_no_score_prob=("no_score_prob", "mean"),
            touchback_prob=("touchback", "mean"),
            inside_twenty_prob=("punt_inside_twenty", "mean"),
            opp_start_yl100=("opp_start_yl100", "mean"),
            n=("epa", "size"),
        )
        .reset_index()
        .rename(columns={"yardline_100": "field_position"})
    )
    summary.to_csv(DATA_DIR / "punt_summary.csv", index=False)
    print(f"  punt_summary.csv  ({len(summary)} field positions)")
    return summary


#------------------------------------------------------------------------------
# Field goal model
#------------------------------------------------------------------------------
"""
A logistic league make-probability curve over kick distance (the app scales it
horizontally by kicker range against KICKER_ANCHOR), plus the same curve dumped
as a lookup table for transparency, plus empirical EPA/WPA by field position
split into makes and misses. The app uses a literal +3 for a made FG's EPA but
draws WPA and all miss values from these empirical summaries.
"""
def build_fg_model(fg_attempts):
    fg = fg_attempts.dropna(subset=["kick_distance", "field_goal_result"]).copy()
    fg["made"] = (fg["field_goal_result"] == "made").astype(int)

    make_model = LogisticRegression(max_iter=10000)
    make_model.fit(fg[["kick_distance"]], fg["made"])
    joblib.dump(make_model, MODELS_DIR / "fg_make_model.pkl")

    dist = np.arange(FG_MIN_DIST, FG_MAX_DIST + 1)
    curve = pd.DataFrame({
        "kick_distance": dist,
        "league_make_prob": make_model.predict_proba(
            pd.DataFrame({"kick_distance": dist}))[:, 1],
    })
    curve.to_csv(DATA_DIR / "fg_makeprob_by_distance.csv", index=False)

    fg["yl"] = fg["yardline_100"].round().astype(int)
    made = (fg[fg["made"] == 1].groupby("yl")
            .agg(make_epa=("epa", "mean"), make_wpa=("wpa", "mean"),
                 make_n=("epa", "size")))
    miss = (fg[fg["made"] == 0].groupby("yl")
            .agg(miss_epa=("epa", "mean"), miss_wpa=("wpa", "mean"),
                 miss_n=("epa", "size")))
    fg_summary = made.join(miss, how="outer").reset_index().rename(columns={"yl": "yardline_100"})
    fg_summary.to_csv(DATA_DIR / "fg_summary.csv", index=False)

    print(f"  fg_make_model.pkl  |  fg_summary.csv ({len(fg_summary)} yardlines)  "
          f"|  made mean epa={fg[fg['made']==1].epa.mean():+.2f}, "
          f"miss mean epa={fg[fg['made']==0].epa.mean():+.2f}")
    return make_model, fg_summary


#------------------------------------------------------------------------------
# Next-score probability tables
#------------------------------------------------------------------------------
"""
Two "who scores next in the half" tables, each also carrying mean expected
points (ep) for the field-position grounding of the go EPA. scoreprobability.csv
is keyed by (yardline_100, ydstogo) from the current 4th-down situation and
serves as the delta baseline and the EP-before. opponentscoreprobability.csv is
keyed by yardline_100 from fresh first downs; its ep is the value of a first-down
possession at that spot, used for both us after a conversion and the opponent
after a turnover or punt (perspective flips by reading it at the relevant spot).
"""
def _next_score_agg(df, keys):
    return (df.groupby(keys)
            .agg(td_prob=("td_prob", "mean"), fg_prob=("fg_prob", "mean"),
                 opp_td_prob=("opp_td_prob", "mean"),
                 opp_fg_prob=("opp_fg_prob", "mean"),
                 no_score_prob=("no_score_prob", "mean"),
                 ep=("ep", "mean"),
                 n=("td_prob", "size"))
            .reset_index())


def build_score_prob_tables(decisiondata, firsts):
    current = _next_score_agg(decisiondata, ["yardline_100", "ydstogo"])
    current.to_csv(DATA_DIR / "scoreprobability.csv", index=False)

    firstdown = _next_score_agg(firsts, ["yardline_100"])
    firstdown.to_csv(DATA_DIR / "opponentscoreprobability.csv", index=False)

    print(f"  scoreprobability.csv ({len(current)} rows)  |  "
          f"opponentscoreprobability.csv ({len(firstdown)} yardlines)")
    return current, firstdown


#------------------------------------------------------------------------------
# Go-for-it conversion model
#------------------------------------------------------------------------------
"""
Logistic model for the chance of converting a fourth down from the situation.
Logistic is used over a tree model on purpose. A decision tool gets asked about
any situation a user types in, including ones that are rare in the data (a tied
game, fourth and 10 at midfield in the first half). Trees extrapolate badly
there and return silly numbers; the smooth logistic curve stays believable
across the whole input range and does not change which choice wins. The value of
going is not learned from the go-attempt sample; the app builds it from the
expected-points tables and the win-probability model, read at the field position
the play would leave you in.
"""
def build_conversion_model(go_attempts):
    cleaned = go_attempts.dropna(subset=GO_FEATURES + ["fourth_down_converted"])
    X, y = cleaned[GO_FEATURES], cleaned["fourth_down_converted"]
    conv = LogisticRegression(fit_intercept=True, max_iter=100000, solver="liblinear")
    conv.fit(X, y)
    joblib.dump(conv, MODELS_DIR / "go_conversion_model.pkl")
    print(f"  go_conversion_model.pkl  (convert rate={y.mean():.3f}, n={len(cleaned)})")
    return conv


#------------------------------------------------------------------------------
# Win probability model
#------------------------------------------------------------------------------
"""
Win probability fit to nflverse wp over every regular-season play, from field
position, score margin, and time left. The app reads our win probability at the
game state each choice would produce (we keep the ball, we score, or the
opponent takes over), so every option is valued the same way. Trained on all
plays, not just fourth downs, so it is not biased by fourth-down decisions.
"""
def build_wp_model(data):
    d = data[data["season_type"] == SEASON_TYPE].dropna(subset=WP_FEATURES + ["wp"])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2 = cross_val_score(XGBRegressor(**XGB_PARAMS), d[WP_FEATURES], d["wp"],
                         cv=kf, scoring="r2").mean()
    model = XGBRegressor(**XGB_PARAMS).fit(d[WP_FEATURES], d["wp"])
    joblib.dump(model, MODELS_DIR / "wp_model.pkl", compress=3)
    print(f"  wp_model.pkl  (n={len(d)}, CV R2={r2:+.3f})")
    return model


#------------------------------------------------------------------------------
# Orchestration
#------------------------------------------------------------------------------
"""
Run the full pipeline: load, split, build each branch, write meta.json, and
print every artifact's size as a free-tier sanity check.
"""
def main(seasons=SEASONS):
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    data = load_pbp(seasons)
    firsts, decisiondata, dd4, go_attempts, punt_attempts, fg_attempts = build_datasets(data)

    print("Building punt model ...")
    build_punt_model(punt_attempts)
    print("Building field goal model ...")
    build_fg_model(fg_attempts)
    print("Building next-score tables ...")
    build_score_prob_tables(decisiondata, firsts)
    print("Building conversion model ...")
    build_conversion_model(go_attempts)
    print("Building win probability model ...")
    build_wp_model(data)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": list(seasons),
        "season_type": SEASON_TYPE,
        "kicker_anchor_yd": KICKER_ANCHOR,
        "go_features": GO_FEATURES,
        "wp_features": WP_FEATURES,
        "counts": {"go": int(len(go_attempts)), "punt": int(len(punt_attempts)),
                   "fg": int(len(fg_attempts)), "firsts": int(len(firsts))},
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nArtifact sizes:")
    for d in (MODELS_DIR, DATA_DIR):
        for p in sorted(d.iterdir()):
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1024:.0f} KB")
    print(f"\nDone in {time.time()-t0:.1f}s. Seasons: {list(seasons)}")


if __name__ == "__main__":
    main()
