import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# --- Step 1 to 5: Data Preparation (Same as before) ---
print("Loading and preparing data...")
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
campaigns_df = pd.read_csv(os.path.join(DATA_DIR, 'campaigns.csv'))
ads_df = pd.read_csv(os.path.join(DATA_DIR, 'ads.csv'))
events_df = pd.read_csv(os.path.join(DATA_DIR, 'ad_events.csv'))
merged_df = pd.merge(ads_df, campaigns_df, on='campaign_id', how='left')
ad_performance_df = events_df.groupby('ad_id')['event_type'].value_counts().unstack(fill_value=0).reset_index()
master_df = pd.merge(merged_df, ad_performance_df, on='ad_id', how='left')
categorical_features = ['ad_platform', 'ad_type', 'target_gender', 'target_age_group', 'target_interests']
engineered_df = pd.get_dummies(master_df, columns=categorical_features, drop_first=True)
print("Data preparation complete.")

# --- Define Features (X) and All Possible Targets ---
all_outcomes = ['Click', 'Comment', 'Impression', 'Like', 'Purchase', 'Share']
X = engineered_df.drop(columns=['ad_id', 'campaign_id', 'name', 'start_date', 'end_date'] + all_outcomes)
X = X.fillna(0)
y_all = engineered_df[all_outcomes].fillna(0)

# --- Train and Save a Model for Each Target ---
targets_to_predict = ['Purchase', 'Click', 'Share']
for target in targets_to_predict:
    print(f"--- Training model for {target} ---")
    
    y = y_all[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model to a specific file, e.g., 'purchase_model.joblib'
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', f'{target.lower()}_model.joblib')
    joblib.dump(model, MODEL_PATH)
    print(f"Model for {target} saved successfully to: {MODEL_PATH}")

# --- Finally, save the column list once for all models ---
print("\n--- Saving column list ---")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '..', 'training_columns.joblib')
joblib.dump(X.columns.tolist(), COLUMNS_PATH)
print(f"Column list saved successfully to: {COLUMNS_PATH}")