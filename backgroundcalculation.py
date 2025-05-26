# -*- coding: utf-8 -*-
"""
Created on Thu May 22 13:56:58 2025

@author: natha
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib as mpl
import math
from scipy.interpolate import interp1d

seasons = [2021,2022,2023,2024]
data = nfl.import_pbp_data(seasons)
filtered = data[(data['down'] == 4.0) | (data['field_goal_attempt'] == 1.0)]
columns_to_keep = ['posteam','posteam_type','defteam','side_of_field',
                   'yardline_100','half_seconds_remaining',
                   'game_seconds_remaining','qtr','down','ydstogo','ydsnet',
                   'yards_gained','epa','wp','def_wp','wpa','vegas_wpa','pass_attempt',
                   'season','cp','cpoe','goal_to_go','air_yards','field_goal_attempt',
                   'field_goal_result','kick_distance','score_differential','no_score_prob','opp_fg_prob',
                   'opp_td_prob', 'fg_prob','td_prob','punt_blocked','punt_inside_twenty',
                   'touchback','punt_attempt','fourth_down_converted','touchdown']

decisiondata = filtered[columns_to_keep]
print(decisiondata.info())

decisiondata_4th = decisiondata[decisiondata['down'] == 4]
go_attempts = decisiondata_4th[
    (decisiondata_4th['field_goal_attempt'] != 1.0) & 
    (decisiondata_4th['punt_attempt'] != 1.0)
]
fg_attempts = decisiondata[decisiondata['field_goal_attempt'] == 1.0]
punt_attempts = decisiondata_4th[decisiondata_4th['punt_attempt'] == 1.0]

"weightpointsadded = (epa * successprob) - (epaf *failprob)"
#-------------------------------------
# Punt modeling
punt_summary = (
    punt_attempts.groupby('yardline_100')
    .agg({
        'epa': 'mean',
        'wpa': 'mean',
        'opp_td_prob': 'mean',
        'opp_fg_prob': 'mean',
        'no_score_prob': 'mean',
        'touchback': 'mean',
        'punt_inside_twenty':'mean'
    })
    .reset_index()
)
punt_summary['weighted_points'] = punt_summary['epa']  # For punts, EPA is effectively the weighted value
punt_summary.columns = ['field_position', 'punt_epa', 'punt_wpa', 'opp_td_prob', 'opp_fg_prob', 'opp_no_score_prob','touchback_prob','inside_twenty_prob', 'punt_weighted_points']
print(punt_summary.info())
#---------------------------------
from scipy.interpolate import interp1d

def convert_coach_yardline_to_yardline_100(yardline: int, team_side: str) -> int:
    if not (1 <= yardline <= 50):
        raise ValueError("Yardline must be between 1 and 50")
    if team_side.lower() == 'own':
        return 100 - yardline
    elif team_side.lower() == 'opponent':
        return yardline
    else:
        raise ValueError("team_side must be 'own' or 'opponent'")

f_punt_epa = interp1d(
    punt_summary['field_position'],
    punt_summary['punt_epa'],
    kind='linear',
    fill_value='extrapolate'
)
f_punt_wpa = interp1d(
    punt_summary['field_position'],
    punt_summary['punt_wpa'],
    kind='linear',
    fill_value='extrapolate'
)
f_touchback = interp1d(
    punt_summary['field_position'],
    punt_summary['touchback_prob'],
    kind='linear',
    fill_value='extrapolate'
)
f_opp_td_prob = interp1d(
    punt_summary['field_position'],
    punt_summary['opp_td_prob'],
    kind='linear',
    fill_value='extrapolate'
)
f_opp_fg_prob = interp1d(
    punt_summary['field_position'],
    punt_summary['opp_fg_prob'],
    kind='linear',
    fill_value='extrapolate'
)
f_opp_no_score_prob = interp1d(
    punt_summary['field_position'],
    punt_summary['opp_no_score_prob'],
    kind='linear',
    fill_value='extrapolate'
)

def adjusted_punt_field_pos(yardline_100, gross_punt_yards):
    tb_prob = float(f_touchback(yardline_100))
    raw_landing_yl_100 = yardline_100 + gross_punt_yards
    pos_if_tb = 80
    pos_if_no_tb = 100 - raw_landing_yl_100
    return tb_prob * pos_if_tb + (1 - tb_prob) * pos_if_no_tb

def epa_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_punt_epa(adj_fp))

def wpa_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_punt_wpa(adj_fp))

def opp_td_prob_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_opp_td_prob(adj_fp))

def opp_fg_prob_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_opp_fg_prob(adj_fp))

def opp_no_score_prob_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_opp_no_score_prob(adj_fp))

def weighted_points_added_punt(yardline_100, gross_punt_yards):
    tb_prob = float(f_touchback(yardline_100))
    raw_landing_yl_100 = yardline_100 + gross_punt_yards
    epa_no_tb = float(f_punt_epa(100 - raw_landing_yl_100))
    epa_tb = float(f_punt_epa(80))
    return (epa_no_tb * (1 - tb_prob)) - (epa_tb * tb_prob)

def punt_decision_metrics(coach_yardline, team_side, gross_punt_yards):
    yardline_100 = convert_coach_yardline_to_yardline_100(coach_yardline, team_side)
    epa = epa_if_punt(yardline_100, gross_punt_yards)
    wpa = wpa_if_punt(yardline_100, gross_punt_yards)
    opp_td = opp_td_prob_if_punt(yardline_100, gross_punt_yards)
    opp_fg = opp_fg_prob_if_punt(yardline_100, gross_punt_yards)
    opp_no_score = opp_no_score_prob_if_punt(yardline_100, gross_punt_yards)
    weighted_points = weighted_points_added_punt(yardline_100, gross_punt_yards)
    return {
        "epa": epa,
        "wpa": wpa,
        "opp_td_prob": opp_td,
        "opp_fg_prob": opp_fg,
        "opp_no_score_prob": opp_no_score,
        "weighted_points_added": weighted_points
    }

# Example usage
def test_punt_metrics():
    coach_yardline = 35
    team_side = 'own'        # 'own' or 'opponent'
    gross_punt_yards = 45

    results = punt_decision_metrics(coach_yardline, team_side, gross_punt_yards)

    print(f"Punt metrics for yardline {coach_yardline} ({team_side} side) with gross punt yards {gross_punt_yards}:")
    print(f"EPA: {results['epa']:.4f}")
    print(f"WPA: {results['wpa']:.4f}")
    print(f"Opponent TD Probability: {results['opp_td_prob']:.4f}")
    print(f"Opponent FG Probability: {results['opp_fg_prob']:.4f}")
    print(f"Opponent No Score Probability: {results['opp_no_score_prob']:.4f}")
    print(f"Weighted Points Added: {results['weighted_points_added']:.4f}")

test_punt_metrics()

punt_summary.to_csv("C:/Users/natha/Documents/BGA/4thDownTool/punt_summary.csv", index = False)
def initialize_punt_model():
    global f_punt_epa, f_punt_wpa, f_touchback
    global f_opp_td_prob, f_opp_fg_prob, f_opp_no_score_prob

    punt_summary = pd.read_csv("C:/Users/natha/Documents/BGA/4thDownTool/punt_summary.csv")

    f_punt_epa = interp1d(punt_summary['field_position'], punt_summary['punt_epa'], kind='linear', fill_value='extrapolate')
    f_punt_wpa = interp1d(punt_summary['field_position'], punt_summary['punt_wpa'], kind='linear', fill_value='extrapolate')
    f_touchback = interp1d(punt_summary['field_position'], punt_summary['touchback_prob'], kind='linear', fill_value='extrapolate')
    f_opp_td_prob = interp1d(punt_summary['field_position'], punt_summary['opp_td_prob'], kind='linear', fill_value='extrapolate')
    f_opp_fg_prob = interp1d(punt_summary['field_position'], punt_summary['opp_fg_prob'], kind='linear', fill_value='extrapolate')
    f_opp_no_score_prob = interp1d(punt_summary['field_position'], punt_summary['opp_no_score_prob'], kind='linear', fill_value='extrapolate')

#------------------------------------------------------------------------------------
#--------------------------------- Go Modeling --------------------------------------
#------------------------------------------------------------------------------------
scoreprobability = (
    decisiondata.groupby(['yardline_100','ydstogo'])
    .agg({
        'td_prob': 'mean',
        'fg_prob': 'mean',
        'opp_td_prob': 'mean',
        'opp_fg_prob': 'mean',
        'no_score_prob': 'mean'
    })
    .reset_index()
)
scoreprobability.to_csv("C:/Users/natha/Documents/BGA/4thDownTool/scoreprobability.csv", index = False)

print(go_attempts.info())
go_attempts['side_of_field'] = go_attempts[go_attempts['side_of_field']] if 'side_of_field' == 'posteam'

Xgo = ['ydstogo','qtr','side_of_field','yardline_100','wp','cp','td_prob','fg_prob','opp_fg_prob','opp_td_prob','no_score_prob']
Ygo = ['fourth_down_converted']
go_attempts['conversionprobability'] = 
go_summary = (
    go_attempts.groupby('ydstogo')
    .agg({
        'epa': 'mean',
        'wpa': 'mean',
        'opp_td_prob': 'mean',
        'opp_fg_prob': 'mean',
        'no_score_prob': 'mean',
        'touchback': 'mean',
        'punt_inside_twenty':'mean'
    })
    .reset_index()
)
punt_summary['weighted_points'] = punt_summary['epa']