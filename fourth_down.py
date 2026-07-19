# -*- coding: utf-8 -*-
"""
fourth_down.py -- 4th Down Decision Tool: runtime decision logic.

Loads the artifacts built by train.py once at import, then exposes decide(),
which takes a live situation and returns the go / field-goal / punt options with
their EPA (points), WPA (fraction), next-score breakdown, delta vs the current
situation, and resulting field position. The Dash app calls decide(). Keeping
the math here means it can be tested on its own.

Every option is valued the same way: by the game state it leaves you in. For
points that is expected points at the resulting field position. For win chance
that is the win probability model read at the resulting state. This keeps the
three choices on one footing and matches how public fourth down models work.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

#------------------------------------------------------------------------------
# Artifact loading
#------------------------------------------------------------------------------
"""
Load every model and table once. KICKER_ANCHOR, GO_FEATURES and WP_FEATURES come
from meta.json so this module always matches whatever train.py produced.
"""
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

_meta = json.loads((DATA_DIR / "meta.json").read_text())
KICKER_ANCHOR = _meta["kicker_anchor_yd"]
GO_FEATURES = _meta["go_features"]
WP_FEATURES = _meta["wp_features"]

_fg_make = joblib.load(MODELS_DIR / "fg_make_model.pkl")
_go_conv = joblib.load(MODELS_DIR / "go_conversion_model.pkl")
_wp_model = joblib.load(MODELS_DIR / "wp_model.pkl")

_punt = pd.read_csv(DATA_DIR / "punt_summary.csv").sort_values("field_position")
_score_cur = pd.read_csv(DATA_DIR / "scoreprobability.csv")
_first = pd.read_csv(DATA_DIR / "opponentscoreprobability.csv").sort_values("yardline_100")

TD_EP = 7.0
FG_EP = 3.0
KICKOFF_YL = 75  # opponent starts about their own 25 after a kickoff


#------------------------------------------------------------------------------
# Small helpers
#------------------------------------------------------------------------------
"""
Coordinate and clock conversions plus table lookups. coach_to_yl100 turns "the
35, our side" into yardline_100. clock_to_seconds turns a quarter and mm:ss into
the half and game seconds the models use. The lookup helpers read the by-yardline
tables and the win probability model.
"""
def coach_to_yl100(yard_line, side):
    if not 1 <= yard_line <= 50:
        raise ValueError("yard_line must be 1-50")
    return 100 - yard_line if side.lower() == "own" else yard_line


def clock_to_seconds(quarter, minutes, seconds):
    rem = minutes * 60 + seconds
    if quarter >= 5:
        return rem, rem
    half_seconds = rem + (15 * 60 if quarter in (1, 3) else 0)
    game_seconds = rem + (4 - quarter) * 15 * 60
    return half_seconds, game_seconds


def _interp(x, xs, ys):
    order = np.argsort(xs)
    return float(np.interp(x, np.asarray(xs)[order], np.asarray(ys)[order]))


def _first_down(yl100):
    """Next-score odds on a fresh first down at yl100 (offense perspective)."""
    yl = float(np.clip(yl100, 1, 99))
    return {c: _interp(yl, _first["yardline_100"], _first[c])
            for c in ("td_prob", "fg_prob", "opp_td_prob", "opp_fg_prob")}


def _firstdown_ep(yl100):
    """Expected points of a fresh first-down possession at yl100."""
    return _interp(np.clip(yl100, 1, 99), _first["yardline_100"], _first["ep"])


def _wp(yl100, score_diff, game_sec, half_sec):
    """Win probability for the team with the ball at this state."""
    X = pd.DataFrame([{"yardline_100": np.clip(yl100, 1, 99),
                       "score_differential": score_diff,
                       "game_seconds_remaining": game_sec,
                       "half_seconds_remaining": half_sec}])[WP_FEATURES]
    return float(np.clip(_wp_model.predict(X)[0], 0.0, 1.0))


def _cur_row(yl100, ydstogo):
    d = _score_cur.copy()
    d["dist"] = (d["yardline_100"] - yl100).abs() + (d["ydstogo"] - ydstogo).abs()
    return d.nsmallest(1, "dist").iloc[0]


def _current_our_score(yl100, ydstogo):
    row = _cur_row(yl100, ydstogo)
    return float(row["td_prob"] + row["fg_prob"])


def _blank():
    return dict(our_td=0.0, our_fg=0.0, opp_td=0.0, opp_fg=0.0)


def _finalize(shares, ep_after, wp_after, base, ball, extra):
    our = shares["our_td"] + shares["our_fg"]
    opp = shares["opp_td"] + shares["opp_fg"]
    out = dict(shares)
    out.update(our_score=our, opp_score=opp, no_score=max(0.0, 1.0 - our - opp),
               epa=float(ep_after - base["ep"]), wpa=float(wp_after - base["wp"]),
               result_yl100=float(np.clip(ball, 0, 100)))
    out.update(extra)
    return out


#------------------------------------------------------------------------------
# Go for it
#------------------------------------------------------------------------------
"""
Conversion chance from the logistic model. On a convert the ball sits at the
line to gain plus one yard, or a touchdown if it reaches the end zone, and we
read our value there. On a fail the opponent takes over at the line of scrimmage
and we read their value from our side. The points and win values are the
convert-weighted blend of those two outcomes.
"""
def _go(st):
    yl100, ydstogo = st["yl100"], st["ydstogo"]
    sd, gsec, hsec = st["sd"], st["gsec"], st["hsec"]
    X = pd.DataFrame([st["feat"]])[GO_FEATURES]
    p = float(_go_conv.predict_proba(X)[0][1])

    spot = yl100 - (ydstogo + 1)
    if spot <= 0:
        conv = dict(our_td=1.0, our_fg=0.0, opp_td=0.0, opp_fg=0.0)
        ep_c = TD_EP
        wp_c = 1 - _wp(KICKOFF_YL, -(sd + 7), gsec, hsec)
    else:
        fd = _first_down(spot)
        conv = dict(our_td=fd["td_prob"], our_fg=fd["fg_prob"],
                    opp_td=fd["opp_td_prob"], opp_fg=fd["opp_fg_prob"])
        ep_c = _firstdown_ep(spot)
        wp_c = _wp(spot, sd, gsec, hsec)

    fdf = _first_down(100 - yl100)
    fail = dict(our_td=fdf["opp_td_prob"], our_fg=fdf["opp_fg_prob"],
                opp_td=fdf["td_prob"], opp_fg=fdf["fg_prob"])
    ep_f = -_firstdown_ep(100 - yl100)
    wp_f = 1 - _wp(100 - yl100, -sd, gsec, hsec)

    s = {k: p * conv[k] + (1 - p) * fail[k] for k in _blank()}
    ep_after = p * ep_c + (1 - p) * ep_f
    wp_after = p * wp_c + (1 - p) * wp_f
    if p >= 0.5:
        poss, ball = ("score", 100) if spot <= 0 else ("us", 100 - max(0, spot))
    else:
        poss, ball = "opp", 100 - yl100
    return _finalize(s, ep_after, wp_after, st["base"], ball,
                     dict(prob=p, prob_label="convert", ball_pos=ball, possession=poss))


#------------------------------------------------------------------------------
# Field goal
#------------------------------------------------------------------------------
"""
Kick distance is yardline_100 plus 17. The league make curve is scaled by kicker
range against KICKER_ANCHOR, so a kicker is treated as equally accurate at his
own limit as the league is at 65. A make is worth 3 points and then a kickoff. A
miss hands the opponent the ball at the spot of the kick, or their 20 if inside.
"""
def _fg(st):
    yl100, sd, gsec, hsec = st["yl100"], st["sd"], st["gsec"], st["hsec"]
    kicker_range = max(1, st["kicker_range"])
    kick_distance = yl100 + 17
    eff = kick_distance * KICKER_ANCHOR / kicker_range
    make_p = float(np.clip(_fg_make.predict_proba(
        pd.DataFrame({"kick_distance": [eff]}))[0][1], 0, 1))

    miss_yl = float(np.clip(min(92 - yl100, 80), 1, 99))
    fdm = _first_down(miss_yl)
    s = dict(our_td=(1 - make_p) * fdm["opp_td_prob"],
             our_fg=make_p * 1.0 + (1 - make_p) * fdm["opp_fg_prob"],
             opp_td=(1 - make_p) * fdm["td_prob"],
             opp_fg=(1 - make_p) * fdm["fg_prob"])

    ep_after = make_p * FG_EP + (1 - make_p) * (-_firstdown_ep(miss_yl))
    wp_make = 1 - _wp(KICKOFF_YL, -(sd + 3), gsec, hsec)
    wp_miss = 1 - _wp(miss_yl, -sd, gsec, hsec)
    wp_after = make_p * wp_make + (1 - make_p) * wp_miss

    poss, ball = ("score", 100 - yl100) if make_p >= 0.5 else ("opp", miss_yl)
    return _finalize(s, ep_after, wp_after, st["base"], ball,
                     dict(prob=make_p, prob_label="make", kick_distance=kick_distance,
                          ball_pos=ball, possession=poss))


#------------------------------------------------------------------------------
# Punt
#------------------------------------------------------------------------------
"""
Assume the punt is downed with no return. Blend by the touchback chance at the
punting spot between the opponent starting at their own 20 and at the downed spot
(100 minus yardline_100 plus punter range). The opponent then has the ball there,
so we read our value from the other side. A stronger punter pins them deeper,
which raises the value, the same way kicker range works for a field goal.
"""
def _punt_option(st):
    yl100, sd, gsec, hsec = st["yl100"], st["sd"], st["gsec"], st["hsec"]
    punter_range = max(1, st["punter_range"])
    tb = float(np.clip(_interp(yl100, _punt["field_position"], _punt["touchback_prob"]), 0, 1))
    downed = float(np.clip(100 - yl100 + punter_range, 1, 99))
    opp_yl = tb * 80 + (1 - tb) * downed

    fd = _first_down(opp_yl)
    s = dict(our_td=fd["opp_td_prob"], our_fg=fd["opp_fg_prob"],
             opp_td=fd["td_prob"], opp_fg=fd["fg_prob"])
    ep_after = -_firstdown_ep(opp_yl)
    wp_after = 1 - _wp(opp_yl, -sd, gsec, hsec)
    return _finalize(s, ep_after, wp_after, st["base"], opp_yl,
                     dict(prob=None, prob_label=None, ball_pos=opp_yl, possession="opp"))


#------------------------------------------------------------------------------
# Public entry point
#------------------------------------------------------------------------------
"""
decide() builds the situation and the current-state baseline, runs the three
branches, tags each option's our-score delta against the current situation, and
picks the points-best and win-best options. WPA is a fraction; the app shows it
as a percent. EPA is in points.
"""
def decide(yard_line, side, quarter, minutes, seconds,
           our_score, their_score, ydstogo, kicker_range, punter_range):
    yl100 = coach_to_yl100(yard_line, side)
    half_sec, game_sec = clock_to_seconds(quarter, minutes, seconds)
    sd = our_score - their_score
    feat = {"ydstogo": ydstogo, "qtr": quarter, "half_seconds_remaining": half_sec,
            "yardline_100": yl100, "score_differential": sd}
    base = {"ep": float(_cur_row(yl100, ydstogo)["ep"]),
            "wp": _wp(yl100, sd, game_sec, half_sec)}
    st = {"yl100": yl100, "ydstogo": ydstogo, "sd": sd, "gsec": game_sec,
          "hsec": half_sec, "feat": feat, "base": base,
          "kicker_range": kicker_range, "punter_range": punter_range}

    options = {"go": _go(st), "fg": _fg(st), "punt": _punt_option(st)}

    baseline = _current_our_score(yl100, ydstogo)
    for o in options.values():
        o["our_score_delta"] = o["our_score"] - baseline

    return {
        "options": options,
        "baseline_our_score": baseline,
        "rec_epa": max(options, key=lambda k: options[k]["epa"]),
        "rec_wpa": max(options, key=lambda k: options[k]["wpa"]),
        "situation": {"yardline_100": yl100, "ydstogo": ydstogo,
                      "half_seconds_remaining": half_sec,
                      "game_seconds_remaining": game_sec, "score_differential": sd},
    }


if __name__ == "__main__":
    r = decide(40, "opponent", 4, 8, 0, 17, 21, 3, 55, 60)
    print(f"EPA pick: {r['rec_epa']}   WPA pick: {r['rec_wpa']}")
    for name, o in r["options"].items():
        pl = f"{o['prob']:.0%} {o['prob_label']}" if o["prob"] is not None else "-"
        print(f"  {name:5s} EPA={o['epa']:+.2f}  WPA={o['wpa']*100:+.1f}%  "
              f"our={o['our_score']:.0%} opp={o['opp_score']:.0%}  ({pl})")
