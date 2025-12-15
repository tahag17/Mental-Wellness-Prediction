# predict_service.py
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import json
import numpy as np

# Load trained model, scaler, and metadata
model = load_model("mental_wellness_model.keras")
scaler = joblib.load("scaler.joblib")

with open("columns.json", "r") as f:
    info = json.load(f)

X_columns = info["X_columns"]
num_cols = info["num_cols"]

def predict_new_user(new_user_dict):
    """
    Returns Mental Wellness Index for a new user (0-100)
    """
    new_user_df = pd.DataFrame([new_user_dict])
    cat_cols = ['gender', 'occupation', 'work_mode']
    new_user_df = pd.get_dummies(new_user_df, columns=cat_cols, drop_first=True)

    # Feature engineering
    # new_user_df['work_leisure_screen_ratio'] = new_user_df['work_screen_hours'] / (new_user_df['leisure_screen_hours'] + 1e-5)
    # new_user_df['total_screen_hours'] = new_user_df['work_screen_hours'] + new_user_df['leisure_screen_hours']

    # Add missing columns
    for col in X_columns:
        if col not in new_user_df.columns:
            new_user_df[col] = 0

    new_user_df = new_user_df[X_columns]

    # Scale numerical features
    # scaler_cols = num_cols + ['work_leisure_screen_ratio', 'total_screen_hours']
    # new_user_df[scaler_cols] = scaler.transform(new_user_df[scaler_cols])
    new_user_df[num_cols] = scaler.transform(new_user_df[num_cols])

    # Predict and clamp
    prediction = model.predict(new_user_df)[0][0]
    return float(np.clip(prediction, 0, 100))
