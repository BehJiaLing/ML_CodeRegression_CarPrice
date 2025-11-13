import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
import os

# =====================================
# 1. Load preprocessed dataset
# =====================================
df = pd.read_csv("datasets/train-processed-data.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(f"Dataset loaded: {df.shape}")

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# =====================================
# 2. Define function to create baseline model
# =====================================
def create_baseline_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =====================================
# 3. Train Baseline Model
# =====================================
print("\n🏗️ Training baseline DNN model...")
baseline_model = create_baseline_model(X_train.shape[1])

history_base = baseline_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

y_pred_base = baseline_model.predict(X_test).flatten()

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print("\n📊 Baseline Model Evaluation:")
print(f"RMSE: {rmse_base:.4f}")
print(f"MAE:  {mae_base:.4f}")
print(f"R²:   {r2_base:.4f}")

# =====================================
# 4. Define function for Hyperparameter Tuning
# =====================================
def build_tuned_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
    model.add(layers.Dense(1))
    
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

# =====================================
# 5. Hyperparameter Tuning with KerasTuner
# =====================================
print("\n🔍 Starting hyperparameter tuning (KerasTuner)...")

# Use Random Search (try random combinations)
tuner = kt.RandomSearch(
    build_tuned_model,
    # Optimize for validation MAE
    objective='val_mae',
    # Test up to 10 different configurations
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_logs',
    project_name='car_price_dnn'
)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\n✅ Best Hyperparameters:")
for key, value in best_hp.values.items():
    print(f"{key}: {value}")

# =====================================
# 6. Train Best Tuned Model
# =====================================
best_model = tuner.hypermodel.build(best_hp)
history_tuned = best_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

y_pred_tuned = best_model.predict(X_test).flatten()

rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print("\n📊 Tuned DNN Model Evaluation:")
print(f"RMSE: {rmse_tuned:.4f}")
print(f"MAE:  {mae_tuned:.4f}")
print(f"R²:   {r2_tuned:.4f}")

# =====================================
# 7. Compare Baseline vs Tuned Results
# =====================================
results_df = pd.DataFrame({
    "Model": ["Baseline DNN", "Tuned DNN"],
    "RMSE": [rmse_base, rmse_tuned],
    "MAE": [mae_base, mae_tuned],
    "R²": [r2_base, r2_tuned]
})
print("\n📈 Comparison Summary:")
print(results_df)

plt.figure(figsize=(7,4))
sns.barplot(data=results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", hue="Model")
plt.title("Model Performance Comparison (Baseline vs Tuned DNN)")
plt.ylabel("Score")
plt.show()

# =====================================
# 8. Visualize Tuned Predictions
# =====================================
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_tuned, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Tuned DNN)")
plt.show()

# =====================================
# 9. Save Tuned Model
# =====================================
os.makedirs("models", exist_ok=True)
best_model.save("models/dnn_carprice_tuned.keras")
print("\n💾 Tuned DNN model saved successfully to 'models/dnn_carprice_tuned.keras'")
