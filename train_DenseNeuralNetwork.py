# # =============================================
# # Step 1: Import libraries
# # =============================================
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import tensorflow as tf
# import os

# # =============================================
# # Step 2: Load dataset
# # =============================================
# df = pd.read_csv("datasets/train-processed-data.csv")
# print(f"Dataset loaded: {df.shape}")

# # Separate features and target
# X = df.drop("Price", axis=1)
# y = df["Price"]

# # =============================================
# # Step 3: Train-test split
# # =============================================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print(f"Train shape: {X_train.shape}")
# print(f"Test shape: {X_test.shape}")

# # =============================================
# # Step 4: Build Neural Network model
# # =============================================
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dense(1, activation='linear')  # regression output
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()

# # =============================================
# # Step 5: Train model with early stopping
# # =============================================
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stop],
#     verbose=1
# )

# # =============================================
# # Step 6: Evaluate model
# # =============================================
# y_pred = model.predict(X_test).flatten()

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\n📊 Model Evaluation:")
# print(f"RMSE: {rmse:.2f}")
# print(f"MAE:  {mae:.2f}")
# print(f"R² Score: {r2:.4f}")

# # =============================================
# # Step 7: Save model
# # =============================================
# os.makedirs("models", exist_ok=True)
# model.save("models/deep_learning_carprice.h5")

# print("\n✅ Deep Learning model saved as 'models/deep_learning_carprice.h5'")

# =============================================
# Step 1: Import libraries
# =============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# =============================================
# Step 2: Load dataset
# =============================================
df = pd.read_csv("datasets/train-processed-data.csv")
print(f"Dataset loaded: {df.shape}")

# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# =============================================
# Step 3: Train-test split
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# =============================================
# Step 4: Build Neural Network model
# =============================================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer (regression)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =============================================
# Step 5: Train model with early stopping
# =============================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =============================================
# Step 6: Evaluate model
# =============================================
y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Final Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# =============================================
# Step 7: Visualization - Training History
# =============================================
plt.figure(figsize=(14, 5))

# ---- Loss plot ----
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# ---- MAE plot ----
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('Model MAE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# ---- Actual vs Predicted ----
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', edgecolor='white')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price (log scale)')
plt.ylabel('Predicted Price (log scale)')
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================
# Step 8: Save model
# =============================================
os.makedirs("models", exist_ok=True)
model.save("models/deep_learning_carprice.h5")

print("\n✅ Deep Learning model saved as 'models/deep_learning_carprice.h5'")
