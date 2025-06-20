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
firsts = data[(data['down'] == 1.0) & (data['ydstogo']==10) | (data['down']==1.0) & (data['goal_to_go'] == 1.0) & (data['ydstogo'] <= 10)]
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
    team_side = 'own'
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
#------------------------------- Field Goal Modeling --------------------------------
#------------------------------------------------------------------------------------
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
fg_attempts = decisiondata[decisiondata['field_goal_attempt'] == 1.0]

bins = list(range(19, 71)) + [float('inf')]
labels = [str(i) for i in range(19, 71)] 

fg_attempts['kick_bin'] = pd.cut(fg_attempts['kick_distance'], bins=bins, labels=labels, right=False)

distance_freq = fg_attempts['kick_bin'].value_counts().sort_index()

distance_freq_df = distance_freq.reset_index()
distance_freq_df.columns = ['Kick Distance (yards)', 'FG Attempts']

display(distance_freq_df)
#-----------
bins = list(range(19, 67)) + [float('inf')]
labels = [str(i) for i in range(19, 67)] 

made_fgs = fg_attempts[fg_attempts['field_goal_result'] == 'made']

made_fgs['kick_bin'] = pd.cut(made_fgs['kick_distance'], bins=bins, labels=labels, right=False)

made_distance_freq = made_fgs['kick_bin'].value_counts().sort_index()

made_distance_freq_df = made_distance_freq.reset_index()
made_distance_freq_df.columns = ['Kick Distance (yards)', 'FG Makes']

display(made_distance_freq_df)
#---------------
bins = list(range(19, 72))
labels = [str(i) for i in range(19, 71)]

fg_attempts['kick_bin'] = pd.cut(fg_attempts['kick_distance'], bins=bins, labels=labels, right=False)

fg_attempts_filtered = fg_attempts.dropna(subset=['kick_bin'])

attempts = fg_attempts_filtered['kick_bin'].value_counts().sort_index()
made = fg_attempts_filtered[fg_attempts_filtered['field_goal_result'] == 'made']['kick_bin'].value_counts().sort_index()

df = pd.DataFrame({
    'Made': made,
    'Attempts': attempts
}).fillna(0)

df['Make_Prob'] = df['Made'] / df['Attempts']
df['Miss_Prob'] = 1 - df['Make_Prob']

df = df.reset_index().rename(columns={'kick_bin': 'Kick Distance (yards)'})

display(df)
#---------------------
df_model = fg_attempts_filtered.dropna(subset=['kick_distance', 'field_goal_result']).copy()

df_model['made_binary'] = (df_model['field_goal_result'] == 'made').astype(int)

X = df_model['kick_distance'].values.reshape(-1, 1)
y = df_model['made_binary'].values

model = LogisticRegression()
model.fit(X, y)

distances = np.arange(19, 71).reshape(-1, 1)
predicted_probs = model.predict_proba(distances)[:, 1]

attempts = fg_attempts_filtered['kick_distance'].value_counts().reindex(range(19, 71), fill_value=0).sort_index()
made = fg_attempts_filtered[fg_attempts_filtered['field_goal_result'] == 'made']['kick_distance'].value_counts().reindex(range(19, 71), fill_value=0).sort_index()

summary_df = pd.DataFrame({
    'Kick Distance (yards)': range(19, 71),
    'Attempts': attempts.values,
    'Made': made.values,
    'Make_Prob': made.values / attempts.replace(0, np.nan).values, 
    'Model_Predicted_Prob': predicted_probs,
    'Model_Missed_Predicted_Prob' : 1 - predicted_probs
})

summary_df['Make_Prob'] = summary_df['Make_Prob'].fillna(0)

display(summary_df)
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

oppprobability = (
    firsts.groupby(['yardline_100','ydstogo'])
    .agg({
        'td_prob': 'mean',
        'fg_prob': 'mean',
        'opp_td_prob': 'mean',
        'opp_fg_prob': 'mean',
        'no_score_prob': 'mean'
    })
    .reset_index()
)
oppprobability.to_csv("C:/Users/natha/Documents/BGA/FourthDownTool/opponentscoreprobability.csv", index = False)

print(go_attempts.info())
go_attempts.to_csv("C:/Users/natha/Documents/BGA/Graduate Projects/FourthDownTool/go_attempts.csv")
#--------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

#--------------------------------
features = ['ydstogo', 'qtr', 'half_seconds_remaining', 'yardline_100', 'score_differential']
go_data_cleaned = go_attempts.dropna(subset=features + ['fourth_down_converted', 'epa', 'wpa'])
X = go_data_cleaned[features]
y = go_data_cleaned['fourth_down_converted']

#--------------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
models = {
    'LogReg': LogisticRegression(max_iter=10000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(),
    'LinReg': LogisticRegression(fit_intercept=True, max_iter=10000, solver='liblinear'),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'NaiveBayes': GaussianNB()
}

def display_feature_importance(model, model_name, feature_names):
    print(f"\n--- Feature Importances for {model_name} ---")
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        for feat, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
            print(f"{feat}: {coef:.4f}")
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"{feat}: {imp:.4f}")
    else:
        print("Feature importances not supported for this model.")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    model.fit(X, y)
    display_feature_importance(model, name, X.columns)

#--------------------------------
top_features = ['ydstogo', 'qtr', 'score_differential', 'yardline_100']
X_top = go_data_cleaned[top_features]
y = go_data_cleaned['fourth_down_converted']
kf = KFold(n_splits=25, shuffle=True, random_state=42)
linreg_top = LogisticRegression(fit_intercept=True, max_iter=100000, solver='liblinear')
linreg_top.fit(X_top, y)
scores = cross_val_score(linreg_top, X_top, y, cv=kf, scoring='accuracy')
print(f"LinReg (Top Features) CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
joblib.dump(linreg_top, "C:/Users/natha/Documents/BGA/Graduate Projects/FourthDownTool/go_for_it_model.pkl")
gosuccessmodel = joblib.load("C:/Users/natha/Documents/BGA/4thDownTool/go_for_it_model.pkl")
input_dict = {'ydstogo': 3, 'qtr': 2, 'score_differential': -3, 'yardline_100': 45}

#--------------------------------
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def select_best_regressor(X, y, label='epa/wpa'):
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, eval_metric='rmse', random_state=42)
    }
    best_model = None
    best_score = -np.inf
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_score = scores.mean()
        print(f'{label} - {name} R2 CV Score: {mean_score:.4f}')
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
    best_model.fit(X, y)
    return best_model

#--------------------------------
success_df = go_data_cleaned[go_data_cleaned['fourth_down_converted'] == 1]
X_success = success_df[features]
epa_success = success_df['epa']
wpa_success = success_df['wpa']

fail_df = go_data_cleaned[go_data_cleaned['fourth_down_converted'] == 0]
X_fail = fail_df[features]
epa_fail = fail_df['epa']
wpa_fail = fail_df['wpa']

epa_model_success = select_best_regressor(X_success, epa_success, label='EPA Success')
wpa_model_success = select_best_regressor(X_success, wpa_success, label='WPA Success')
epa_model_fail = select_best_regressor(X_fail, epa_fail, label='EPA Fail')
wpa_model_fail = select_best_regressor(X_fail, wpa_fail, label='WPA Fail')

joblib.dump(epa_model_success, 'epa_model_success.pkl')
joblib.dump(wpa_model_success, 'wpa_model_success.pkl')

fail_averages = (
    fail_df
    .groupby(['yardline_100', 'ydstogo'])[['epa', 'wpa']]
    .mean()
    .reset_index()
)
fail_averages.to_csv("fail_epa_wpa_averages.csv", index=False)

def get_fail_averages(yardline_100, ydstogo, fail_avg_df):
    df = fail_avg_df.copy()
    row = df[
        (df['yardline_100'] == round(yardline_100)) &
        (df['ydstogo'] == round(ydstogo))
    ]
    if not row.empty:
        return float(row['epa'].values[0]), float(row['wpa'].values[0])
    else:
        return None, None

#--------------------------------
def predict_conversion_prob(input_dict, best_model):
    df = pd.DataFrame([input_dict])
    return best_model.predict_proba(df)[0][1]

def predict_epa_wpa(input_dict):
    df = pd.DataFrame([input_dict])
    epa_success_pred = epa_model_success.predict(df)[0]
    epa_fail_pred = epa_model_fail.predict(df)[0]
    wpa_success_pred = wpa_model_success.predict(df)[0]
    wpa_fail_pred = wpa_model_fail.predict(df)[0]
    return epa_success_pred, epa_fail_pred, wpa_success_pred, wpa_fail_pred

def expected_gain(input_dict, best_model):
    success_prob = predict_conversion_prob(input_dict, best_model)
    epa_succ, epa_fail, wpa_succ, wpa_fail = predict_epa_wpa(input_dict)
    expected_epa = (success_prob * epa_succ) + ((1 - success_prob) * epa_fail)
    expected_wpa = (success_prob * wpa_succ) + ((1 - success_prob) * wpa_fail)
    return {
        'conversion_prob': success_prob,
        'expected_epa': expected_epa,
        'expected_wpa': expected_wpa,
        'epa_success': epa_succ,
        'epa_fail': epa_fail,
        'wpa_success': wpa_succ,
        'wpa_fail': wpa_fail
    }

#--------------------------------
example_input = {
    'ydstogo': 3,
    'qtr': 4,
    'game_seconds_remaining': 120,
    'yardline_100': 40,
    'score_differential': -3
}
results = expected_gain(example_input, gosuccessmodel)
print(f"Conversion Probability: {results['conversion_prob']:.2f}")
print(f"EPA (Success): {results['epa_success']:.2f} | EPA (Fail): {results['epa_fail']:.2f}")
print(f"WPA (Success): {results['wpa_success']:.2f} | WPA (Fail): {results['wpa_fail']:.2f}")
print(f"Expected EPA: {results['expected_epa']:.2f}")
print(f"Expected WPA: {results['expected_wpa']:.2f}")
