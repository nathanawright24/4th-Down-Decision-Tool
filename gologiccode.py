# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 19:51:47 2025

@author: natha
"""

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
