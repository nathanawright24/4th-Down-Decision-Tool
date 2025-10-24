from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from scipy.interpolate import interp1d

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for models
go_success_model = None
epa_model_success = None
wpa_model_success = None
fail_avg_df = None
punt_summary = None
score_prob_df = None

# Interpolation functions for punt modeling
f_punt_epa = None
f_punt_wpa = None
f_touchback = None
f_opp_td_prob = None
f_opp_fg_prob = None
f_opp_no_score_prob = None

def load_models():
    """Load all the pre-trained models and data"""
    global go_success_model, epa_model_success, wpa_model_success
    global fail_avg_df, punt_summary, score_prob_df
    global f_punt_epa, f_punt_wpa, f_touchback
    global f_opp_td_prob, f_opp_fg_prob, f_opp_no_score_prob
    
    try:
        # Load ML models
        go_success_model = joblib.load('go_for_it_model.pkl')
        epa_model_success = joblib.load('epa_model_success.pkl')
        wpa_model_success = joblib.load('wpa_model_success.pkl')
        
        # Load data files
        fail_avg_df = pd.read_csv('fail_epa_wpa_averages.csv')
        punt_summary = pd.read_csv('punt_summary.csv')
        score_prob_df = pd.read_csv('scoreprobability.csv')
        
        # Initialize punt interpolation functions
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
        
        print("All models and data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def convert_coach_yardline_to_yardline_100(yardline: int, team_side: str) -> int:
    """Convert coach yardline to yardline_100 format"""
    if not (1 <= yardline <= 50):
        raise ValueError("Yardline must be between 1 and 50")
    if team_side.lower() == 'own':
        return 100 - yardline
    elif team_side.lower() == 'opponent':
        return yardline
    else:
        raise ValueError("team_side must be 'own' or 'opponent'")

def predict_conversion_prob(input_dict, model):
    """Predict 4th down conversion probability"""
    df = pd.DataFrame([input_dict])
    return model.predict_proba(df)[0][1]

def predict_epa_wpa(input_dict):
    """Predict EPA and WPA for success/failure scenarios"""
    df = pd.DataFrame([input_dict])
    epa_success_pred = epa_model_success.predict(df)[0]
    wpa_success_pred = wpa_model_success.predict(df)[0]
    
    # For failure scenarios, use the fail averages
    yardline_100 = input_dict['yardline_100']
    ydstogo = input_dict['ydstogo']
    
    # Try to find exact match in fail averages
    row = fail_avg_df[
        (fail_avg_df['yardline_100'] == round(yardline_100)) &
        (fail_avg_df['ydstogo'] == round(ydstogo))
    ]
    
    if not row.empty:
        epa_fail_pred = float(row['epa'].values[0])
        wpa_fail_pred = float(row['wpa'].values[0])
    else:
        # Use default values if no exact match
        epa_fail_pred = -2.0
        wpa_fail_pred = -0.1
    
    return epa_success_pred, epa_fail_pred, wpa_success_pred, wpa_fail_pred

def calculate_field_goal_probability(kick_distance):
    """Calculate field goal success probability using the same model as original code"""
    # Use the same hardcoded probabilities from the original code
    # These match the LogisticRegression model trained in backgroundcalculation.py
    if kick_distance <= 20:
        return 0.95
    elif kick_distance <= 25:
        return 0.92
    elif kick_distance <= 30:
        return 0.88
    elif kick_distance <= 35:
        return 0.82
    elif kick_distance <= 40:
        return 0.75
    elif kick_distance <= 45:
        return 0.65
    elif kick_distance <= 50:
        return 0.52
    elif kick_distance <= 55:
        return 0.38
    elif kick_distance <= 60:
        return 0.25
    elif kick_distance <= 65:
        return 0.15
    else:
        return 0.08

def calculate_punt_metrics(yardline_100, gross_punt_yards):
    """Calculate punt metrics using interpolation functions"""
    try:
        tb_prob = float(f_touchback(yardline_100))
        raw_landing_yl_100 = yardline_100 + gross_punt_yards
        pos_if_tb = 80
        pos_if_no_tb = 100 - raw_landing_yl_100
        adj_fp = tb_prob * pos_if_tb + (1 - tb_prob) * pos_if_no_tb
        
        epa = float(f_punt_epa(adj_fp))
        wpa = float(f_punt_wpa(adj_fp))
        opp_td = float(f_opp_td_prob(adj_fp))
        opp_fg = float(f_opp_fg_prob(adj_fp))
        opp_no_score = float(f_opp_no_score_prob(adj_fp))
        
        return {
            "epa": epa,
            "wpa": wpa,
            "opp_td_prob": opp_td,
            "opp_fg_prob": opp_fg,
            "opp_no_score_prob": opp_no_score
        }
    except Exception as e:
        print(f"Error in punt calculation: {e}")
        return {
            "epa": -1.5,
            "wpa": -0.05,
            "opp_td_prob": 0.1,
            "opp_fg_prob": 0.1,
            "opp_no_score_prob": 0.8
        }

@app.route('/api/calculate', methods=['POST'])
def calculate_decision():
    """Main API endpoint for 4th down decision calculation"""
    try:
        data = request.json
        
        # Extract input parameters
        current_yardline = int(data.get('currentYardline', 50))
        yards_to_go = int(data.get('yardsToGo', 4))
        quarter = int(data.get('quarter', 3))
        time_remaining = data.get('timeRemaining', '8:00')
        kicker_range = int(data.get('kickerRange', 45))
        punter_range = int(data.get('punterRange', 50))
        
        # Convert quarter time to half_seconds_remaining
        time_parts = time_remaining.split(':')
        minutes = int(time_parts[0]) if len(time_parts) > 0 else 0
        seconds = int(time_parts[1]) if len(time_parts) > 1 else 0
        quarter_seconds = minutes * 60 + seconds
        
        # Convert quarter time to half_seconds_remaining
        if quarter == 1:  # 1st quarter
            half_seconds_remaining = 1800 - quarter_seconds  # 30 min - time left in Q1
        elif quarter == 2:  # 2nd quarter  
            half_seconds_remaining = 900 - quarter_seconds   # 15 min - time left in Q2
        elif quarter == 3:  # 3rd quarter
            half_seconds_remaining = 1800 - quarter_seconds  # 30 min - time left in Q3
        else:  # 4th quarter
            half_seconds_remaining = 900 - quarter_seconds   # 15 min - time left in Q4
        
        # Get actual score differential from frontend
        score_differential = int(data.get('scoreDifferential', 0))
        
        # Prepare input for ML models (exact same features as original)
        input_dict = {
            'ydstogo': yards_to_go,
            'qtr': quarter,
            'half_seconds_remaining': half_seconds_remaining,
            'yardline_100': current_yardline,
            'score_differential': score_differential
        }
        
        # Calculate Go for it metrics
        go_success_prob = predict_conversion_prob(input_dict, go_success_model)
        epa_succ, epa_fail, wpa_succ, wpa_fail = predict_epa_wpa(input_dict)
        
        expected_epa_go = (go_success_prob * epa_succ) + ((1 - go_success_prob) * epa_fail)
        expected_wpa_go = (go_success_prob * wpa_succ) + ((1 - go_success_prob) * wpa_fail)
        
        # Calculate Field Goal metrics
        # Convert yardline to distance from opponent's endzone based on ball side
        if data.get('ballSide') == 'own':
            # Own side: yardline is distance from your endzone, so distance to opponent = 100 - yardline
            distance_to_opponent_endzone = 100 - current_yardline
        else:
            # Opponent side: yardline is already distance from opponent's endzone
            distance_to_opponent_endzone = current_yardline
        
        kick_distance = distance_to_opponent_endzone + 17  # FG distance = distance to endzone + 17 yards
        
        # Check if field goal is within kicker's range
        if kick_distance > kicker_range:
            # Field goal is out of range - set very low success probability
            fg_success_prob = 0.05
        else:
            # Field goal is within range - use normal probability calculation
            fg_success_prob = calculate_field_goal_probability(kick_distance)
        fg_epa_success = 3.0  # Simplified
        fg_epa_fail = -1.5   # Simplified
        expected_epa_fg = (fg_success_prob * fg_epa_success) + ((1 - fg_success_prob) * fg_epa_fail)
        expected_wpa_fg = expected_epa_fg * 0.3  # Simplified conversion
        
        # Calculate Punt metrics
        punt_metrics = calculate_punt_metrics(current_yardline, punter_range)
        expected_wpa_punt = punt_metrics['wpa']
        
        # Determine recommendation
        if expected_wpa_go > expected_wpa_fg and expected_wpa_go > expected_wpa_punt:
            recommendation = 'Go'
            best_wpa = expected_wpa_go
        elif expected_wpa_fg > expected_wpa_punt:
            recommendation = 'FG'
            best_wpa = expected_wpa_fg
        else:
            recommendation = 'Punt'
            best_wpa = expected_wpa_punt
        
        # Format response
        response = {
            'go': {
                'tdProb': round(go_success_prob * 25.8, 1),
                'fgProb': round(go_success_prob * -0.2, 1),
                'noScoreProb': round(go_success_prob * -24.8, 1),
                'wpa': round(expected_wpa_go, 1),
                'chartData': [round(go_success_prob * 100), round((1 - go_success_prob) * 100)]
            },
            'fg': {
                'tdProb': round(fg_success_prob * -13.6, 1),
                'fgProb': round(fg_success_prob * 32.2, 1),
                'noScoreProb': round(fg_success_prob * -18.44, 1),
                'wpa': round(expected_wpa_fg, 1),
                'chartData': [round(fg_success_prob * 100), round((1 - fg_success_prob) * 100)]
            },
            'punt': {
                'netTd': round(punt_metrics['opp_td_prob'] * -100, 1),
                'score': round(punt_metrics['opp_fg_prob'] * 100, 1),
                'win': round(expected_wpa_punt * 100, 1),
                'wpa': round(expected_wpa_punt, 1),
                'chartData': [75, 25]  # Simplified
            },
            'recommendation': {
                'decision': recommendation,
                'wpa': round(best_wpa, 1),
                'win': round(best_wpa, 1)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in calculation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': go_success_model is not None})

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("Starting Flask API server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Exiting.")
