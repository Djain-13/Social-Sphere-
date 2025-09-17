import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# --- Step 1 to 4: Loading and Merging ---
print("Loading and preparing data...")
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
campaigns_df = pd.read_csv(os.path.join(DATA_DIR, 'campaigns.csv'))
ads_df = pd.read_csv(os.path.join(DATA_DIR, 'ads.csv'))
events_df = pd.read_csv(os.path.join(DATA_DIR, 'ad_events.csv'))
merged_df = pd.merge(ads_df, campaigns_df, on='campaign_id', how='left')
ad_performance_df = events_df.groupby('ad_id')['event_type'].value_counts().unstack(fill_value=0).reset_index()
master_df = pd.merge(merged_df, ad_performance_df, on='ad_id', how='left')
print("Master dataset created.")

# --- Step 5: Feature Engineering ---
print("Performing feature engineering...")
categorical_features = ['ad_platform', 'ad_type', 'target_gender', 'target_age_group', 'target_interests']
engineered_df = pd.get_dummies(master_df, columns=categorical_features, drop_first=True)
print("Feature engineering complete.")

# --- Step 6: Define Target and Features ---
print("Defining target and features for the model...")
# Our goal is to predict the number of 'Purchases'
y = engineered_df['Purchase']

# Our features are all the columns EXCEPT the performance metrics and unnecessary identifiers
X = engineered_df.drop(columns=[
    'ad_id', 'campaign_id', 'name', 'start_date', 'end_date', # Identifiers and dates
    'Click', 'Comment', 'Impression', 'Like', 'Purchase', 'Share' # All possible outcomes
])

# --- Step 7: Split Data into Training and Testing Sets ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 8: Train the Linear Regression Model ---
print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# --- Step 9: Save the Trained Model ---
print("Saving the trained model...")
# We save the model in the root directory of the project for easy access
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ad_performance_model.joblib')
joblib.dump(model, MODEL_PATH)
print(f"Model saved successfully to: {MODEL_PATH}")



