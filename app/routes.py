# --- Imports ---
from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, current_user, logout_user, login_required
from app import app, db
from app.models import User
import bcrypt
import pandas as pd
import os
import joblib

# --- Load All Models and the Column List ---
# This code runs once when the application starts.
MODELS = {}
targets_to_predict = ['purchase', 'click', 'share']
for target in targets_to_predict:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', f'{target}_model.joblib')
    MODELS[target] = joblib.load(MODEL_PATH)

COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '..', 'training_columns.joblib')
training_columns = joblib.load(COLUMNS_PATH)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# --- Public Routes ---

@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles new user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles existing user login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs the current user out."""
    logout_user()
    return redirect(url_for('home'))


# --- Protected Routes (Require Login) ---

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the main dashboard with data."""
    data_path = os.path.join(DATA_DIR, 'master_dataset.csv')
    df = pd.read_csv(data_path)
    
    total_budget = df['total_budget'].sum()
    total_impressions = df['Impression'].sum()
    total_clicks = df['Click'].sum()
    total_purchases = df['Purchase'].sum()
    
    campaigns = df[['name', 'total_budget', 'duration_days', 'Click', 'Purchase']].to_dict(orient='records')
    
    return render_template('dashboard.html', 
                           total_budget=int(total_budget), 
                           total_impressions=total_impressions, 
                           total_clicks=total_clicks, 
                           total_purchases=total_purchases,
                           campaigns=campaigns)

@app.route('/predictor')
@login_required
def predictor():
    """Renders the prediction tool page."""
    return render_template('predictor.html')

@app.route('/audience')
@login_required
def audience():
    """Renders the audience insights page with charts."""
    data_path = os.path.join(DATA_DIR, 'master_dataset.csv')
    df = pd.read_csv(data_path)
    
    age_performance = df.groupby('target_age_group')['Purchase'].sum().sort_values(ascending=False)
    age_labels = age_performance.index.tolist()
    age_data = age_performance.values.tolist()

    gender_performance = df.groupby('target_gender')['Purchase'].sum()
    gender_labels = gender_performance.index.tolist()
    gender_data = gender_performance.values.tolist()
    
    return render_template('audience.html', 
                           age_labels=age_labels, 
                           age_data=age_data,
                           gender_labels=gender_labels,
                           gender_data=gender_data)


# --- API Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """Receives data and returns AI predictions for multiple metrics."""
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        
        # Data preparation
        input_df['total_budget'] = pd.to_numeric(input_df['total_budget'])
        input_df['duration_days'] = pd.to_numeric(input_df['duration_days'])
        input_encoded = pd.get_dummies(input_df)
        input_realigned = input_encoded.reindex(columns=training_columns, fill_value=0)
        
        # --- Make a prediction with each model ---
        predictions = {}
        for target, model in MODELS.items():
            prediction_value = model.predict(input_realigned)
            predictions[f'predicted_{target}'] = float(prediction_value[0])
            
        return jsonify(predictions)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500