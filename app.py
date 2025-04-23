from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load your trained model
model = joblib.load('carbon_rf_model.pkl')

# Define expected features
FEATURES = ["prev1", "prev2", "prev3", "prev_avg", "prev_diff"]

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # Extract features from request
        values = [
            data["prev1"],
            data["prev2"],
            data["prev3"],
            (data["prev1"] + data["prev2"] + data["prev3"]) / 3,  # avg
            data["prev1"] - data["prev3"]                          # diff
        ]

        # Convert to DataFrame
        input_df = pd.DataFrame([values], columns=FEATURES)

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({"emission_prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
