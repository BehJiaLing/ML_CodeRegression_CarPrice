# =============================================
# Hybrid Model: Random Forest + DNN (Tuned)
# =============================================

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================
# Step 1: Load dataset
# =============================================
df = pd.read_csv("datasets/train-processed-data.csv")
print("Dataset loaded:", df.shape)

# Separate features and target
X = df.drop(columns=['Price'])
y = df['Price']

# Train-test split (same ratio as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =============================================
# Step 2: Load tuned models
# =============================================
rf_model = joblib.load("models/random_forest_carprice_tuned.pkl")
dnn_model = load_model("models/dnn_carprice_tuned.keras")

print("\n✅ Tuned models loaded successfully!")

# =============================================
# Step 3: Make predictions
# =============================================
rf_pred = rf_model.predict(X_test)
dnn_pred = dnn_model.predict(X_test).flatten()

# Combine predictions (Weighted Average)
hybrid_pred = (0.6 * rf_pred) + (0.4 * dnn_pred)  # tweak weights if needed

# =============================================
# Step 4: Evaluate performance
# =============================================
rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
mae = mean_absolute_error(y_test, hybrid_pred)
r2 = r2_score(y_test, hybrid_pred)

print("\n📊 Hybrid Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# =============================================
# Step 5: Visualization
# =============================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, hybrid_pred, alpha=0.6)
plt.xlabel("Actual Price (log scale)")
plt.ylabel("Predicted Price (log scale)")
plt.title("Actual vs Predicted Prices - Hybrid Model")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Residual plot
plt.figure(figsize=(6,4))
residuals = y_test - hybrid_pred
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Price (log scale)")
plt.ylabel("Residuals")
plt.title("Residual Plot - Hybrid Model")
plt.show()

# =============================================
# Step 6: Save hybrid predictions
# =============================================
hybrid_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': hybrid_pred
})
hybrid_results.to_csv("datasets/hybrid_predictions.csv", index=False)
print("\n💾 Saved hybrid predictions to 'models/hybrid_predictions.csv'")
