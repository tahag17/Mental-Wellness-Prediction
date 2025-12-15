# # ann_model.py
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
# import joblib

# # -----------------------------
# # 0️⃣ Load and preprocess dataset
# # -----------------------------
# df = pd.read_csv("pyapi\ScreenTime vs MentalWellness.csv")
# df = df.drop_duplicates()
# if 'Unnamed: 15' in df.columns:
#     df = df.drop(columns=['Unnamed: 15'])

# num_cols = [
#     'age', 'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
#     'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
#     'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week'
# ]
# target_col = 'mental_wellness_index_0_100'
# cat_cols = ['gender', 'occupation', 'work_mode']

# # Fill missing values
# df[num_cols + [target_col]] = df[num_cols + [target_col]].fillna(df[num_cols + [target_col]].median())
# df[cat_cols] = df[cat_cols].fillna('Unknown')

# # One-Hot Encoding
# df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# # Feature engineering
# df_encoded['work_leisure_screen_ratio'] = df_encoded['work_screen_hours'] / (df_encoded['leisure_screen_hours'] + 1e-5)
# df_encoded['total_screen_hours'] = df_encoded['work_screen_hours'] + df_encoded['leisure_screen_hours']

# new_num_cols = num_cols + ['work_leisure_screen_ratio', 'total_screen_hours']

# # Clip outliers
# for col in new_num_cols:
#     Q1 = df_encoded[col].quantile(0.25)
#     Q3 = df_encoded[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5*IQR
#     upper = Q3 + 1.5*IQR
#     df_encoded[col] = df_encoded[col].clip(lower, upper)

# # Split features and target
# X = df_encoded.drop([target_col, 'user_id'], axis=1)
# y = df_encoded[target_col]

# # Scaling numerical features
# scaler = StandardScaler()
# X[new_num_cols] = scaler.fit_transform(X[new_num_cols])

# # Save scaler for later
# joblib.dump(scaler, "scaler.joblib")

# # Train / val / test split
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # -----------------------------
# # 1️⃣ Build ANN model
# # -----------------------------
# model = Sequential([
#     Input(shape=(X_train.shape[1],)),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='linear')  # Regression
# ])

# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
# print("✅ Model compiled")
# model.summary()

# # -----------------------------
# # 2️⃣ Train the model
# # -----------------------------
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=100,
#     batch_size=16,
#     verbose=1
# )

# # -----------------------------
# # 3️⃣ Evaluate on test set
# # -----------------------------
# loss, mae = model.evaluate(X_test, y_test, verbose=0)
# print(f"🎯 MAE: {mae:.3f}, 💥 MSE: {loss:.3f}")

# # -----------------------------
# # 4️⃣ Save the model
# # -----------------------------
# model.save("mental_wellness_model.keras")  # ✅ Use .keras extension

# # -----------------------------
# # 5️⃣ Optional: plot training history
# # -----------------------------
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.xlabel('Epochs')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.show()
