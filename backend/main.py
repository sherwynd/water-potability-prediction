from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib  # Make sure to install joblib with pip if not already installed

app = Flask(__name__)
CORS(app)
import os
print("Current working directory:", os.getcwd())
model = joblib.load('mlp_model.joblib')

@app.route('/')
def first():
    return jsonify({"name": "Adam"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print(data)
    try:
        # Convert data to 2D array as model expects
        features = np.array([[data['ph'], data['hardness'], data['solids'], data['chloramines'], data['sulfate'], data['conductivity'], data['organicCarbon'], data['trihalomethanes'], data['turbidity']]])
        prediction = model.predict(features)
        # You might want to return the prediction as a plain number or boolean
        # Depending on your model's output, adjust accordingly
        is_portable = int(prediction[0])
        return jsonify({'isPortable': is_portable}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
