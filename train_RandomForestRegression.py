# # =============================================
# # Step 1: Import libraries
# # =============================================
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt

# # =============================================
# # Step 2: Load preprocessed dataset
# # =============================================
# df = pd.read_csv("datasets/train-processed-data.csv")
# print("Dataset loaded:", df.shape)

# # =============================================
# # Step 3: Split features and target
# # =============================================
# X = df.drop('Price', axis=1)
# y = df['Price']   # log-transformed target

# # Train-test split (80/20)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print("Train shape:", X_train.shape)
# print("Test shape:", X_test.shape)

# # =============================================
# # Step 4: Initialize and train Random Forest model
# # =============================================
# rf_model = RandomForestRegressor(
#     n_estimators=200,      # number of trees
#     max_depth=None,        # grow until all leaves pure
#     min_samples_split=2,
#     min_samples_leaf=1,
#     random_state=42,
#     n_jobs=-1              # use all CPU cores
# )

# print("\nTraining Random Forest model...")
# rf_model.fit(X_train, y_train)
# print("✅ Training complete!")

# # =============================================
# # Step 5: Make predictions
# # =============================================
# y_pred = rf_model.predict(X_test)

# # Convert back from log1p (to get actual price)
# y_test_exp = np.expm1(y_test)
# y_pred_exp = np.expm1(y_pred)

# # =============================================
# # Step 6: Evaluate performance
# # =============================================
# mse = mean_squared_error(y_test_exp, y_pred_exp)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test_exp, y_pred_exp)
# mae = mean_absolute_error(y_test, y_pred)

# print("\n📊 Model Evaluation:")
# print(f"RMSE: {rmse:,.2f}")
# print(f"R² Score: {r2:.4f}")
# print(f"MAE:  {mae:.2f}")

# # =============================================
# # Step 7: Feature importance visualization
# # =============================================
# importances = rf_model.feature_importances_
# indices = np.argsort(importances)[::-1]

# top_n = 10  # show top 10 features
# top_features = X.columns[indices][:top_n]
# top_importances = importances[indices][:top_n]

# plt.figure(figsize=(10,5))
# plt.barh(top_features[::-1], top_importances[::-1])
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature")
# plt.title("Top 10 Important Features - Random Forest")
# plt.tight_layout()
# plt.show()

# # =============================================
# # Step 8: Save trained model (optional)
# # =============================================
# import joblib
# joblib.dump(rf_model, "models/random_forest_carprice.pkl")
# print("\n💾 Model saved to: models/random_forest_carprice.pkl")

# =============================================
# Step 1: Import libraries
# =============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# =============================================
# Step 2: Load dataset
# =============================================
df = pd.read_csv("datasets/train-processed-data.csv")
print(f"Dataset loaded: {df.shape}")

# Split features and target
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
# Step 4: Train Random Forest model
# =============================================
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("✅ Training complete!")

# =============================================
# Step 5: Model Evaluation
# =============================================
y_pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# =============================================
# Step 6: Visualizations
# =============================================

plt.figure(figsize=(15, 5))

# ---- 1️⃣ Actual vs Predicted ----
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', edgecolor='white')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price (log scale)')
plt.ylabel('Predicted Price (log scale)')
plt.grid(True)

# ---- 2️⃣ Residual Distribution ----
residuals = y_test - y_pred
plt.subplot(1, 3, 2)
sns.histplot(residuals, bins=30, kde=True, color='teal')
plt.title('Residual Distribution')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)

# ---- 3️⃣ Feature Importance ----
plt.subplot(1, 3, 3)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)
top_features.plot(kind='barh', color='orange')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================
# Step 7: Save model
# =============================================
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest_carprice.pkl")

print("\n✅ Random Forest model saved as 'models/random_forest_carprice.pkl'")
