# ---------------------------------------------
# FIXED ANN pour prédiction Mental Wellness Index
# ---------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------
# 0️⃣ Charger le dataset
# --------------------------------------------------
df = pd.read_csv(r"C:\Users\Electronic Store\Desktop\S9\BI\Projet\ScreenTime vs MentalWellness.csv")
print("✅ Dataset chargé, aperçu :")
print(df.head())

# --------------------------------------------------
# 1️⃣ Nettoyage des données
# --------------------------------------------------
df = df.drop_duplicates()
print(f"✅ Dataset après suppression des doublons: {df.shape}")

num_cols = [
    'age', 'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
    'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
    'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week'
]

target_col = 'mental_wellness_index_0_100'

# Remplissage des valeurs manquantes
df[num_cols + [target_col]] = df[num_cols + [target_col]].fillna(df[num_cols + [target_col]].median())
cat_cols = ['gender', 'occupation', 'work_mode']
df[cat_cols] = df[cat_cols].fillna('Unknown')

print("✅ Valeurs manquantes remplacées.")

# --------------------------------------------------
# 2️⃣ Encodage one-hot
# --------------------------------------------------
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("✅ One-hot encoding effectué, colonnes maintenant :", df.columns.tolist())

# --------------------------------------------------
# 3️⃣ Scaling cohérent
# --------------------------------------------------
scaler = StandardScaler()
X = df.drop([target_col, 'user_id'], axis=1)
y = df[target_col]

# Fit ONLY sur train features après split pour éviter fuite
X_scaled = scaler.fit_transform(X)
print("✅ Scaling appliqué aux features numériques.")

# --------------------------------------------------
# 4️⃣ Split train/val/test
# --------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Shapes : Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# --------------------------------------------------
# 5️⃣ Modèle ANN
# --------------------------------------------------
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("✅ Modèle ANN compilé :")
model.summary()

# --------------------------------------------------
# 6️⃣ Entraînement
# --------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=1
)

# --------------------------------------------------
# 7️⃣ Évaluation finale
# --------------------------------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"🎯 MAE sur le test set : {mae:.3f}")

# --------------------------------------------------
# 8️⃣ Fonction de prédiction avec logging
# --------------------------------------------------
def predict_new_user(model, scaler, X_columns, new_user_dict):
    """
    model     : modèle ANN entraîné
    scaler    : StandardScaler utilisé sur les features
    X_columns : colonnes du dataset d'origine après preprocessing
    new_user_dict : dictionnaire avec les données du nouvel utilisateur
    """
    new_user_df = pd.DataFrame([new_user_dict])
    new_user_df = pd.get_dummies(new_user_df)

    # Ajouter les colonnes manquantes avec 0
    for col in X_columns:
        if col not in new_user_df.columns:
            new_user_df[col] = 0

    # Réordonner les colonnes pour correspondre à X_columns
    new_user_df = new_user_df[X_columns]

    # Normalisation
    new_user_scaled = scaler.transform(new_user_df)

    # Prédiction
    prediction = model.predict(new_user_scaled)
    return prediction[0][0]  # sortie sous forme scalaire

# --------------------------------------------------
# 9️⃣ Exemple utilisateur
# --------------------------------------------------
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
    'gender_Female': 1,
    'occupation_Professional': 1,
    'work_mode_Hybrid': 1
}

# 🔹 Prédiction
prediction = predict_new_user(model, scaler, X.columns, new_user)
print(f"Predicted Mental Wellness Index: {prediction:.2f}")
