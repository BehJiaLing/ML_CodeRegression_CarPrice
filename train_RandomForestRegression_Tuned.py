import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# 1. Load preprocessed data
# =======================
df = pd.read_csv("datasets/train-processed-data.csv")
print(f"Dataset loaded: {df.shape}")

# Drop unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# =======================
# 2. Split features & target
# =======================
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# =======================
# 3. Train Baseline Random Forest
# =======================
print("\n🌲 Training baseline Random Forest...")
rf_base = RandomForestRegressor(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print("\n📊 Baseline Model Evaluation:")
print(f"RMSE: {rmse_base:.4f}")
print(f"MAE:  {mae_base:.4f}")
print(f"R² Score: {r2_base:.4f}")

# =======================
# 4. Hyperparameter Tuning (RandomizedSearchCV)
# =======================
param_dist = {
    # number of trees in the forest
    'n_estimators': [100, 200, 300, 400],
    # how deep each tree can grow
    'max_depth': [None, 10, 20, 30],
    # how strict to create new branches or leaves
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    # how many features to consider when s
    # # randomly samples combinations from the param_distplitting a node
    'max_features': ['sqrt', 'log2']
}

print("\n🔍 Starting hyperparameter tuning (RandomizedSearchCV)...")

# randomly samples combinations from the param_dist
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    # Test 20 random combinations: try 20 random combinations from parameter grid (random set of hyperparameters)
    n_iter=20,
    # Use 3-fold cross-validation (cv=3) to test model robustness: split three and prevent overfitting for the one lucky split
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

print("\n✅ Tuning complete!")
print("Best Parameters:", rf_random.best_params_)
print("Best Cross-Validation R²:", rf_random.best_score_)

# =======================
# 5. Evaluate Tuned Model
# =======================
y_pred_tuned = best_rf.predict(X_test)

rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print("\n📊 Tuned Model Evaluation:")
print(f"RMSE: {rmse_tuned:.4f}")
print(f"MAE:  {mae_tuned:.4f}")
print(f"R² Score: {r2_tuned:.4f}")

# =======================
# 6. Compare Before vs After Tuning
# =======================
results_df = pd.DataFrame({
    'Model': ['Baseline RF', 'Tuned RF'],
    'RMSE': [rmse_base, rmse_tuned],
    'MAE': [mae_base, mae_tuned],
    'R²': [r2_base, r2_tuned]
})
print("\n📈 Comparison Summary:")
print(results_df)

plt.figure(figsize=(7,4))
sns.barplot(data=results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Metric', y='Score', hue='Model')
plt.title("Model Performance Comparison (Baseline vs Tuned RF)")
plt.ylabel("Score")
plt.show()

# =======================
# 7. Visualize Tuned Model Predictions
# =======================
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_tuned, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Tuned RF)")
plt.show()

# =======================
# 8. Feature Importance
# =======================
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False)[:10]
plt.figure(figsize=(8, 5))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 10 Feature Importances (Tuned RF)")
plt.show()

# =======================
# 9. Save Tuned Model
# =======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf, "models/random_forest_carprice_tuned.pkl")
print("\n💾 Tuned model saved successfully!")
