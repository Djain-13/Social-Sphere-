from app import app
from flask import request, jsonify
import joblib
import pandas as pd
import os

# --- 1. Load Model and Training Columns ---
# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ad_performance_model.joblib')
model = joblib.load(MODEL_PATH)

# Load the columns from the training data (we'll need this to ensure the order is correct)
# NOTE: We will create this file in a moment.
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAINING_COLUMNS_PATH = os.path.join(DATA_DIR, 'training_columns.txt')
with open(TRAINING_COLUMNS_PATH, 'r') as f:
    training_columns = f.read().splitlines()


# --- 2. Define the Home Route ---
@app.route('/')
def index():
    return "Your Ad Performance Prediction API is running!"


# --- 3. Define the Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the incoming JSON data to a pandas DataFrame
    input_df = pd.DataFrame([data])
    
    # One-hot encode the new data using the same columns as the training data
    # This ensures we have the same features for prediction
    input_encoded = pd.get_dummies(input_df)
    
    # Realign columns to match the model's training data
    # This adds any missing columns from the one-hot encoding and fills them with 0
    input_realigned = input_encoded.reindex(columns=training_columns, fill_value=0)

    # Make a prediction
    prediction = model.predict(input_realigned)
    
    # Return the prediction as a JSON response
    # We convert the numpy float to a standard Python float for JSON compatibility
    return jsonify({'predicted_purchases': float(prediction[0])})