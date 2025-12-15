# # -*- coding: utf-8 -*-
# """final_ann_verbose.py"""

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.optimizers import Adam
# import joblib

# # -----------------------------
# # 0️⃣ Load Data
# # -----------------------------
# print("📥 Loading data from 'pyapi/all_data.xlsx'...")
# df = pd.read_excel("all_data.xlsx")
# print("✅ Data loaded")
# print("Shape of original data:", df.shape)
# print(df.head())

# # -----------------------------
# # 1️⃣ Remove Non-binary / Other genders
# # -----------------------------
# print("\n🔹 Removing 'non-binary/other' gender rows...")
# initial_rows = df.shape[0]
# df = df[~df['gender'].str.lower().isin(['non-binary/other'])]
# removed_rows = initial_rows - df.shape[0]
# print(f"✅ Removed {removed_rows} rows")
# df = df.reset_index(drop=True)
# print("Shape after removal:", df.shape)

# # -----------------------------
# # 2️⃣ Clean gender column to lowercase
# # -----------------------------
# df['gender'] = df['gender'].astype(str).str.lower()
# print("\n🔹 Gender column after cleaning:")
# print(df['gender'].value_counts())

# # -----------------------------
# # 3️⃣ Count missing values
# # -----------------------------
# print("\n🔹 Checking missing values per column:")
# print(df.isnull().sum())

# # -----------------------------
# # 4️⃣ Fill missing values
# # -----------------------------
# num_cols = [
#     'age', 'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
#     'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
#     'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week'
# ]
# target_col = 'mental_wellness_index_0_100'
# cat_cols = ['gender', 'occupation', 'work_mode']

# print("\n🔹 Filling missing numerical values with median...")
# df[num_cols + [target_col]] = df[num_cols + [target_col]].fillna(df[num_cols + [target_col]].median())

# print("🔹 Filling missing categorical values with 'Unknown'...")
# df[cat_cols] = df[cat_cols].fillna('Unknown')

# print("✅ Missing values handled")
# print(df.isnull().sum())

# # -----------------------------
# # 5️⃣ One-hot encode categorical columns
# # -----------------------------
# print("\n🔹 One-hot encoding categorical columns:", cat_cols)
# df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
# print("Shape after encoding:", df_encoded.shape)
# print("Columns preview:", df_encoded.columns.tolist()[:20])  # first 20 columns

# # -----------------------------
# # 6️⃣ Clip outliers for numerical columns
# # -----------------------------
# print("\n🔹 Clipping outliers in numerical columns...")
# for col in num_cols:
#     Q1 = df_encoded[col].quantile(0.25)
#     Q3 = df_encoded[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5*IQR
#     upper = Q3 + 1.5*IQR
#     df_encoded[col] = df_encoded[col].clip(lower, upper)
# print("✅ Outliers clipped")
# print(df_encoded[num_cols].describe())

# # -----------------------------
# # 7️⃣ Split features and target
# # -----------------------------
# print("\n🔹 Splitting features and target")
# X = df_encoded.drop([target_col, 'user_id'], axis=1)
# y = df_encoded[target_col]
# print("Features shape:", X.shape)
# print("Target shape:", y.shape)

# # -----------------------------
# # 8️⃣ Scale numerical features
# # -----------------------------
# print("\n🔹 Scaling numerical features:", num_cols)
# scaler = StandardScaler()
# X[num_cols] = scaler.fit_transform(X[num_cols])
# print("✅ Scaling complete")
# print(X[num_cols].head())

# # Save scaler
# joblib.dump(scaler, "scaler.joblib")
# print("✅ Scaler saved as 'scaler.joblib'")

# # -----------------------------
# # 9️⃣ Train / validation / test split
# # -----------------------------
# print("\n🔹 Splitting data into train/val/test sets...")
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# # -----------------------------
# # 🔟 Build ANN model
# # -----------------------------
# print("\n🔹 Building ANN model...")
# model = Sequential([
#     Input(shape=(X_train.shape[1],)),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='linear')
# ])
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
# print("✅ Model compiled")
# model.summary()

# # -----------------------------
# # 1️⃣1️⃣ Train the model
# # -----------------------------
# print("\n🔹 Training model...")
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=100,
#     batch_size=16,
#     verbose=1
# )

# # -----------------------------
# # 1️⃣2️⃣ Evaluate model
# # -----------------------------
# print("\n🔹 Evaluating on test set...")
# loss, mae = model.evaluate(X_test, y_test, verbose=0)
# y_pred = model.predict(X_test, verbose=0)
# r2 = r2_score(y_test, y_pred)
# print(f"Test MAE: {mae:.3f}, MSE: {loss:.3f}, R²: {r2:.4f}")

# # -----------------------------
# # 1️⃣3️⃣ Validation R²
# # -----------------------------
# y_val_pred = model.predict(X_val, verbose=0)
# r2_val = r2_score(y_val, y_val_pred)
# print(f"Validation R²: {r2_val:.4f}")

# # -----------------------------
# # 1️⃣4️⃣ Save model
# # -----------------------------
# model.save("mental_wellness_model.keras")
# print("✅ Model saved as 'mental_wellness_model.keras'")

# # -----------------------------
# # 1️⃣5️⃣ Print X_columns and num_cols for predict_service
# # -----------------------------
# X_columns = X.columns.tolist()
# print("\n🔹 X_columns (all model features):", X_columns)
# print("🔹 num_cols (numerical features):", num_cols)

# # Save columns.json
# columns_info = {"X_columns": X_columns, "num_cols": num_cols}
# with open("columns.json", "w") as f:
#     import json
#     json.dump(columns_info, f, indent=2)
# print("✅ columns.json saved in pyapi/")

# # -----------------------------
# # 1️⃣6️⃣ Plot training history
# # -----------------------------
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(history.history['loss'], label='Train Loss')
# axes[0].plot(history.history['val_loss'], label='Validation Loss')
# axes[0].set_title('MSE Loss Over Epochs')
# axes[0].legend()
# axes[0].grid(True)

# axes[1].plot(history.history['mae'], label='Train MAE')
# axes[1].plot(history.history['val_mae'], label='Validation MAE')
# axes[1].set_title('MAE Over Epochs')
# axes[1].legend()
# axes[1].grid(True)
# plt.show()

# # -----------------------------
# # 1️⃣7️⃣ Prediction vs Actual
# # -----------------------------
# plt.figure(figsize=(8, 8))
# plt.scatter(y_test, y_pred, alpha=0.5, s=20)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Predicted vs Actual')
# plt.grid(True)
# plt.show()

# # -----------------------------
# # 1️⃣8️⃣ Correlation heatmap
# # -----------------------------
# plt.figure(figsize=(8,6))
# sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="Blues")
# plt.title("Correlation Heatmap")
# plt.show()
