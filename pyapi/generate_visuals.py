import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

# ====================================================
# LOAD DATASET
# ====================================================
df = pd.read_csv("synthetic_mental_wellness_5000.csv")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DESCRIPTIVE STATS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# ====================================================
# VISUALIZATIONS
# ====================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Histograms
df[numeric_cols].hist(figsize=(18, 18), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=18)
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["mental_wellness_index_0_100"], kde=True)
plt.title("Mental Wellness Index Distribution")
plt.show()

# Scatterplots
for col in ["stress_level_0_10", "sleep_hours", "exercise_minutes_per_week"]:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=col, y="mental_wellness_index_0_100")
    plt.title(f"{col} vs Mental Wellness Index")
    plt.show()


# ====================================================
# PREPROCESSING
# (must match the preprocessing used during model training)
# ====================================================

df_enc = df.copy()

categorical_cols = ["gender", "occupation", "work_mode"]
df_enc = pd.get_dummies(df_enc, columns=categorical_cols, drop_first=True)

X = df_enc.drop(["user_id", "mental_wellness_index_0_100"], axis=1)
y = df_enc["mental_wellness_index_0_100"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================================
# LOAD YOUR TRAINED MODEL (.keras)
# ====================================================

model = keras.models.load_model("mental_wellness_model.keras")


# ====================================================
# MODEL EVALUATION
# ====================================================

y_pred = model.predict(X_test).flatten()

mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MSE  : {mse:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# ====================================================
# OPTIONAL: Compare real vs predicted
# ====================================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 100], [0, 100], color='red')
plt.title("Real vs Predicted Mental Wellness Index")
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.show()
