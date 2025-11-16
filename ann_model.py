# ---------------------------------------------
# ANN POUR LA PRÉDICTION DU MENTAL WELLNESS INDEX
# ---------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -----------------------------
# 0️⃣ Charger dataset pré-traité
# -----------------------------
df = pd.read_csv("ScreenTime vs MentalWellness.csv")
df = df.drop_duplicates()
if 'Unnamed: 15' in df.columns:
    df = df.drop(columns=['Unnamed: 15'])

# Colonnes numériques et cible
num_cols = [
    'age', 'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
    'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
    'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week'
]
target_col = 'mental_wellness_index_0_100'
cat_cols = ['gender', 'occupation', 'work_mode']

# Valeurs manquantes
df[num_cols + [target_col]] = df[num_cols + [target_col]].fillna(df[num_cols + [target_col]].median())
df[cat_cols] = df[cat_cols].fillna('Unknown')

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Feature Engineering
df_encoded['work_leisure_screen_ratio'] = df_encoded['work_screen_hours'] / (df_encoded['leisure_screen_hours'] + 1e-5)
df_encoded['total_screen_hours'] = df_encoded['work_screen_hours'] + df_encoded['leisure_screen_hours']

new_num_cols = num_cols + ['work_leisure_screen_ratio', 'total_screen_hours']

# Outliers : clipping
for col in new_num_cols:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df_encoded[col] = df_encoded[col].clip(lower, upper)

# Scaling
X = df_encoded.drop([target_col, 'user_id'], axis=1)
y = df_encoded[target_col]
scaler = StandardScaler()
X[new_num_cols] = scaler.fit_transform(X[new_num_cols])

# Split train / val / test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -----------------------------
# 1️⃣ Construction du modèle ANN
# -----------------------------
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # régression
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("✅ Modèle ANN compilé :")
model.summary()

# -----------------------------
# 2️⃣ Entraînement
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=1
)

# -----------------------------
# 3️⃣ Évaluation sur le test set
# -----------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"\n🎯 MAE sur le test set : {mae:.3f}")
print(f"💥 MSE sur le test set : {loss:.3f}")

# -----------------------------
# 4️⃣ Courbes d'apprentissage
# -----------------------------
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss au cours des epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# MAE
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE au cours des epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()
# -----------------------------
# 5️⃣ Fonction de prédiction pour un nouvel utilisateur
# -----------------------------
def predict_new_user(model, scaler, X_columns, new_user_dict, num_cols):
    """
    model         : modèle ANN entraîné
    scaler        : StandardScaler utilisé sur les features numériques
    X_columns     : colonnes du dataset d'origine après preprocessing
    new_user_dict : dictionnaire avec les données du nouvel utilisateur (valeurs brutes, pas encodées)
    num_cols      : liste des colonnes numériques à scaler
    """
    new_user_df = pd.DataFrame([new_user_dict])

    # One-Hot Encoding des colonnes catégorielles
    cat_cols = ['gender', 'occupation', 'work_mode']
    new_user_df = pd.get_dummies(new_user_df, columns=cat_cols, drop_first=True)

    # Création des nouvelles features AVANT réordonnage
    new_user_df['work_leisure_screen_ratio'] = new_user_df['work_screen_hours'] / (new_user_df['leisure_screen_hours'] + 1e-5)
    new_user_df['total_screen_hours'] = new_user_df['work_screen_hours'] + new_user_df['leisure_screen_hours']

    # Ajouter les colonnes manquantes avec 0
    for col in X_columns:
        if col not in new_user_df.columns:
            new_user_df[col] = 0

    # Réordonner les colonnes pour correspondre exactement à X_columns
    new_user_df = new_user_df[X_columns]

    # Scaler uniquement les colonnes numériques et les nouvelles features
    scaler_cols = num_cols + ['work_leisure_screen_ratio', 'total_screen_hours']
    new_user_df[scaler_cols] = scaler.transform(new_user_df[scaler_cols])

    # Prédiction
    prediction = model.predict(new_user_df)
    return prediction[0][0]


new_user = {
    'age': 30,
    'screen_time_hours': 5,
    'work_screen_hours': 2,
    'leisure_screen_hours': 3,
    'sleep_hours': 8,
    'sleep_quality_1_5': 5,
    'stress_level_0_10': 1,
    'productivity_0_100': 95,
    'exercise_minutes_per_week': 120,
    'social_hours_per_week': 10,
    'gender': 'Female',
    'occupation': 'Professional',
    'work_mode': 'Hybrid'
}

prediction = predict_new_user(model, scaler, X.columns, new_user, num_cols)

# Clamp entre 0 et 100
prediction = max(0, min(100, prediction))

print(f"\nPredicted Mental Wellness Index: {prediction:.2f}")


# Nouvel utilisateur avec mental wellness faible
low_user = {
    'age': 30,
    'screen_time_hours': 12,        # beaucoup d'écran
    'work_screen_hours': 8,
    'leisure_screen_hours': 4,
    'sleep_hours': 4,               # peu de sommeil
    'sleep_quality_1_5': 1,         # très mauvaise qualité
    'stress_level_0_10': 9,         # stress élevé
    'productivity_0_100': 40,       # faible productivité
    'exercise_minutes_per_week': 20,# peu d’exercice
    'social_hours_per_week': 2,     # peu de social
    'gender': 'Male',
    'occupation': 'Student',
    'work_mode': 'Remote'
}

# Prédiction
low_prediction = predict_new_user(model, scaler, X.columns, low_user, num_cols)

# Clamp entre 0 et 100
low_prediction = max(0, min(100, low_prediction))

print(f"\nPredicted Mental Wellness Index (low user): {low_prediction:.2f}")

