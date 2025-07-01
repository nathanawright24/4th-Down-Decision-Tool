from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Import your prediction function(s)
from gologiccode import expected_gain

# Load your trained model(s)
gosuccessmodel = joblib.load("go_for_it_model.pkl")  # Update path if needed

app = Flask(__name__)
CORS(app)  # In production, restrict origins

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Call your prediction function
        results = expected_gain(data, gosuccessmodel)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
