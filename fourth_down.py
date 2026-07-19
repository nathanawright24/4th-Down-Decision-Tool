# -*- coding: utf-8 -*-
"""
fourth_down.py -- 4th Down Decision Tool: runtime decision logic.

Loads the artifacts built by train.py once at import, then exposes decide(),
which takes a live situation and returns the go / field-goal / punt options with
their EPA (points), WPA (fraction), next-score breakdown, delta vs the current
situation, and resulting field position. The Dash app calls decide() and renders
the result; keeping the math here means it can be unit-tested on its own.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression

#------------------------------------------------------------------------------
# Artifact loading
#------------------------------------------------------------------------------
"""
Load every model and table once. KICKER_ANCHOR and GO_FEATURES come from
meta.json so this module always matches whatever train.py produced. Punt values
are exposed two ways: by punting position (field_position) and by resulting
opponent field position (opp_start_yl100), the latter powering the punter-range
adjustment.
"""
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

_meta = json.loads((DATA_DIR / "meta.json").read_text())
KICKER_ANCHOR = _meta["kicker_anchor_yd"]
GO_FEATURES = _meta["go_features"]

_fg_make = joblib.load(MODELS_DIR / "fg_make_model.pkl")
_go_conv = joblib.load(MODELS_DIR / "go_conversion_model.pkl")
_reg = {f"wpa_{lab}": joblib.load(MODELS_DIR / f"wpa_{lab}_model.pkl")
        for lab in ("success", "fail")}

_punt = pd.read_csv(DATA_DIR / "punt_summary.csv")
_fg_sum = pd.read_csv(DATA_DIR / "fg_summary.csv")
_score_cur = pd.read_csv(DATA_DIR / "scoreprobability.csv")
_first = pd.read_csv(DATA_DIR / "opponentscoreprobability.csv").sort_values("yardline_100")

_punt_by_pos = _punt.sort_values("field_position")
_punt_by_opp = _punt[_punt["n"] >= 5].sort_values("opp_start_yl100")

# Punt value rises monotonically as the opponent starts deeper in their own end
# (higher opp_start_yl100). The raw empirical WPA is tiny and noisy, so fit an
# increasing isotonic curve -- this keeps punter range moving the value the
# right direction without the interpolation bouncing around.
_iso_epa = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
    _punt_by_opp["opp_start_yl100"], _punt_by_opp["punt_epa"], sample_weight=_punt_by_opp["n"])
_iso_wpa = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
    _punt_by_opp["opp_start_yl100"], _punt_by_opp["punt_wpa"], sample_weight=_punt_by_opp["n"])


#------------------------------------------------------------------------------
# Small helpers
#------------------------------------------------------------------------------
"""
Coordinate and clock conversions plus table interpolation. coach_to_yl100 turns
a coach's "the 35, our side" into yardline_100. clock_to_seconds derives the
half/game seconds the models expect from a quarter and mm:ss. _interp is a
clamped linear lookup; _first_down / _fg_val read the by-yardline tables.
"""
def coach_to_yl100(yard_line, side):
    if not 1 <= yard_line <= 50:
        raise ValueError("yard_line must be 1-50")
    return 100 - yard_line if side.lower() == "own" else yard_line


def clock_to_seconds(quarter, minutes, seconds):
    rem = minutes * 60 + seconds
    if quarter >= 5:  # overtime
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


def _fg_val(yl100, col):
    d = _fg_sum.dropna(subset=[col])
    return _interp(np.clip(yl100, 1, 99), d["yardline_100"], d[col])


def _firstdown_ep(yl100):
    """Expected points of a fresh first-down possession at yl100."""
    return _interp(np.clip(yl100, 1, 99), _first["yardline_100"], _first["ep"])


def _cur_row(yl100, ydstogo):
    d = _score_cur.copy()
    d["dist"] = (d["yardline_100"] - yl100).abs() + (d["ydstogo"] - ydstogo).abs()
    return d.nsmallest(1, "dist").iloc[0]


def _current_our_score(yl100, ydstogo):
    """Our (td+fg) next-score odds in the current 4th-down situation."""
    row = _cur_row(yl100, ydstogo)
    return float(row["td_prob"] + row["fg_prob"])


def _blank():
    return dict(our_td=0.0, our_fg=0.0, opp_td=0.0, opp_fg=0.0)


def _finalize(shares, epa, wpa, result_yl100, extra):
    our = shares["our_td"] + shares["our_fg"]
    opp = shares["opp_td"] + shares["opp_fg"]
    out = dict(shares)
    out.update(our_score=our, opp_score=opp,
               no_score=max(0.0, 1.0 - our - opp),
               epa=float(epa), wpa=float(wpa),
               result_yl100=float(np.clip(result_yl100, 0, 100)))
    out.update(extra)
    return out


#------------------------------------------------------------------------------
# Go for it
#------------------------------------------------------------------------------
"""
Conversion probability from the logistic model. On success the ball is spotted
at the line to gain + 1 (a TD if it reaches the end zone) and we read our fresh
first-down odds there; on failure the opponent takes over at the line of
scrimmage and their first-down odds become the opponent's scoring here. EPA is
grounded in expected points by field position -- P(convert) weighting the EP of
the resulting spot against the EP of handing the opponent the ball, minus the EP
of the current 4th down -- which avoids the go-attempt selection bias. WPA stays
the conversion-weighted blend of the success and failure regressors.
"""
TD_EP = 7.0


def _go(feat, yl100, ydstogo):
    X = pd.DataFrame([feat])[GO_FEATURES]
    p = float(_go_conv.predict_proba(X)[0][1])

    s = _blank()
    spot = yl100 - (ydstogo + 1)
    if spot <= 0:  # conversion reaches the end zone
        conv = dict(our_td=1.0, our_fg=0.0, opp_td=0.0, opp_fg=0.0)
        ep_convert = TD_EP
    else:
        fd = _first_down(spot)
        conv = dict(our_td=fd["td_prob"], our_fg=fd["fg_prob"],
                    opp_td=fd["opp_td_prob"], opp_fg=fd["opp_fg_prob"])
        ep_convert = _firstdown_ep(spot)
    fdf = _first_down(100 - yl100)  # opponent takes over at the LOS
    fail = dict(our_td=fdf["opp_td_prob"], our_fg=fdf["opp_fg_prob"],
                opp_td=fdf["td_prob"], opp_fg=fdf["fg_prob"])
    for k in s:
        s[k] = p * conv[k] + (1 - p) * fail[k]

    ep_before = float(_cur_row(yl100, ydstogo)["ep"])
    ep_fail = -_firstdown_ep(100 - yl100)
    epa = p * ep_convert + (1 - p) * ep_fail - ep_before
    wpa = p * _reg["wpa_success"].predict(X)[0] + (1 - p) * _reg["wpa_fail"].predict(X)[0]
    if p >= 0.5:
        poss, ball = ("score", 100) if spot <= 0 else ("us", 100 - max(0, spot))
    else:
        poss, ball = "opp", 100 - yl100
    return _finalize(s, epa, wpa, ball, dict(prob=p, prob_label="convert",
                                             ball_pos=ball, possession=poss))


#------------------------------------------------------------------------------
# Field goal
#------------------------------------------------------------------------------
"""
Kick distance is yardline_100 + 17. The league make-probability curve is scaled
horizontally by kicker range against KICKER_ANCHOR, so a kicker is treated as
equally accurate at his own max as the league is at 65. A make banks our field
goal (EPA = +3 by design; WPA empirical); a miss hands the opponent the ball at
the spot of the kick (or their 20 if inside), with empirical EPA/WPA.
"""
def _fg(yl100, kicker_range):
    kicker_range = max(1, kicker_range)
    kick_distance = yl100 + 17
    eff = kick_distance * KICKER_ANCHOR / kicker_range
    make_p = float(np.clip(_fg_make.predict_proba(
        pd.DataFrame({"kick_distance": [eff]}))[0][1], 0, 1))

    miss_yl = float(np.clip(min(92 - yl100, 80), 1, 99))  # opp takes over here
    fdm = _first_down(miss_yl)
    s = dict(
        our_td=(1 - make_p) * fdm["opp_td_prob"],
        our_fg=make_p * 1.0 + (1 - make_p) * fdm["opp_fg_prob"],
        opp_td=(1 - make_p) * fdm["td_prob"],
        opp_fg=(1 - make_p) * fdm["fg_prob"],
    )
    epa = make_p * 3.0 + (1 - make_p) * _fg_val(yl100, "miss_epa")
    wpa = make_p * _fg_val(yl100, "make_wpa") + (1 - make_p) * _fg_val(yl100, "miss_wpa")
    poss, ball = ("score", 100 - yl100) if make_p >= 0.5 else ("opp", miss_yl)
    return _finalize(s, epa, wpa, ball, dict(prob=make_p, prob_label="make",
                                             kick_distance=kick_distance,
                                             ball_pos=ball, possession=poss))


#------------------------------------------------------------------------------
# Punt
#------------------------------------------------------------------------------
"""
Assume the punt is downed with no return. Blend by the empirical touchback
probability at the punting spot between the opponent starting at their own 20
and at the downed spot (100 - yardline_100 + punter_range). Opponent next-score
comes from the first-down table at that expected end-yardline. EPA/WPA are read
at the same end-yardline via the opp_start_yl100 mapping, so a stronger punter
improves the punt's value, consistent with how kicker range works.
"""
def _punt_option(yl100, punter_range):
    punter_range = max(1, punter_range)
    tb = _interp(yl100, _punt_by_pos["field_position"], _punt_by_pos["touchback_prob"])
    tb = float(np.clip(tb, 0, 1))
    downed = float(np.clip(100 - yl100 + punter_range, 1, 99))
    opp_yl = tb * 80 + (1 - tb) * downed

    fd = _first_down(opp_yl)
    s = dict(our_td=fd["opp_td_prob"], our_fg=fd["opp_fg_prob"],
             opp_td=fd["td_prob"], opp_fg=fd["fg_prob"])
    epa = float(_iso_epa.predict([opp_yl])[0])
    wpa = float(_iso_wpa.predict([opp_yl])[0])
    return _finalize(s, epa, wpa, opp_yl,
                     dict(prob=None, prob_label=None,
                          ball_pos=opp_yl, possession="opp"))


#------------------------------------------------------------------------------
# Public entry point
#------------------------------------------------------------------------------
"""
decide() assembles the situation, runs the three branches, tags the Δ of each
option's our-score against the current situation, and picks the EPA-best and
WPA-best options. Returns everything the cards need; WPA is a fraction (the app
multiplies by 100 for display) and EPA is in points.
"""
def decide(yard_line, side, quarter, minutes, seconds,
           our_score, their_score, ydstogo, kicker_range, punter_range):
    yl100 = coach_to_yl100(yard_line, side)
    half_sec, game_sec = clock_to_seconds(quarter, minutes, seconds)
    feat = {"ydstogo": ydstogo, "qtr": quarter, "half_seconds_remaining": half_sec,
            "yardline_100": yl100, "score_differential": our_score - their_score}

    options = {"go": _go(feat, yl100, ydstogo),
               "fg": _fg(yl100, kicker_range),
               "punt": _punt_option(yl100, punter_range)}

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
                      "game_seconds_remaining": game_sec,
                      "score_differential": our_score - their_score},
    }


if __name__ == "__main__":
    r = decide(yard_line=40, side="opponent", quarter=4, minutes=8, seconds=0,
               our_score=17, their_score=21, ydstogo=3,
               kicker_range=55, punter_range=60)
    print(f"EPA pick: {r['rec_epa']}   WPA pick: {r['rec_wpa']}")
    for name, o in r["options"].items():
        pl = f"{o['prob']:.0%} {o['prob_label']}" if o["prob"] is not None else "-"
        print(f"  {name:5s} EPA={o['epa']:+.2f}  WPA={o['wpa']*100:+.1f}%  "
              f"our={o['our_score']:.0%} opp={o['opp_score']:.0%}  ({pl})")
