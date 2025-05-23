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
attempts = decisiondata_4th[
    (decisiondata_4th['field_goal_attempt'] != 1.0) & 
    (decisiondata_4th['punt_attempt'] != 1.0)
]
fg_attempts = decisiondata[decisiondata['field_goal_attempt'] == 1.0]
punt_attempts = decisiondata_4th[decisiondata_4th['punt_attempt'] == 1.0]

weightpointsadded = epa * successprob
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
        'touchback': 'mean'
    })
    .reset_index()
)
punt_summary['weighted_points'] = punt_summary['epa']  # For punts, EPA is effectively the weighted value
punt_summary.columns = ['field_position', 'punt_epa', 'punt_wpa', 'opp_td_prob', 'opp_fg_prob', 'opp_no_score_prob','touchback_prob', 'punt_weighted_points']
# Build interpolation functions
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

# Adjusted opponent field position given punt with touchback logic
def adjusted_punt_field_pos(yardline_100, gross_punt_yards):
    # Lookup touchback probability for this field position
    tb_prob = float(f_touchback(yardline_100))
    
    raw_landing_yl_100 = yardline_100 + gross_punt_yards
    # Touchback → opponent at their 20 => yardline_100 = 80
    pos_if_tb = 80
    pos_if_no_tb = 100 - raw_landing_yl_100

    return tb_prob * pos_if_tb + (1 - tb_prob) * pos_if_no_tb

# EPA and WPA if punted, adjusting for punter distance & touchback
def epa_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_punt_epa(adj_fp))

def wpa_if_punt(yardline_100, gross_punt_yards):
    adj_fp = adjusted_punt_field_pos(yardline_100, gross_punt_yards)
    return float(f_punt_wpa(adj_fp))

def opp_td_prob_if_punt(yardline_100, gross_punt_yards, touchback_prob=0.3):
    adj_yl_100 = adjusted_punt_field_pos(yardline_100, gross_punt_yards, touchback_prob)
    return f_opp_td_prob(adj_yl_100)

# Repeat for opp_fg_prob, opp_no_score_prob if you plan to display them

# Example usage
print("EPA if punt from own 40 with 45 yard gross:", epa_if_punt(60, 45))
print("WPA if punt from own 40 with 45 yard gross:", wpa_if_punt(60, 45))
"""epa_if_punt[LOS] = epa_model[100 - (LOS + 47.5)]
scaled_net_yards = 28
opponent_pos = 100 - (LOS + scaled_net_yards)
epa_if_punt = epa_model[opponent_pos]

scale_fg = lambda d: min(1, fg_prob_nfl[d] * (coach_max_range / nfl_avg_range))
net_dist_coach = net_dist_nfl[yardline] * (coach_net_avg / nfl_net_avg)
"""

