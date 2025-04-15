import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import mean_squared_log_error

# Load model metadata
model_info_path = "./K_fold_LGBM/model_info.json"

if not os.path.exists(model_info_path):
    raise FileNotFoundError(f"[ERROR] Model information file not found: {model_info_path}")

with open(model_info_path, "r") as f:
    model_info = json.load(f)

best_model_fold = model_info["best_model_fold"]
best_model_path = model_info["fold_models"][f"fold_{best_model_fold}"]["model_path"]

print(f"[INFO] Best model is from Fold {best_model_fold} ({best_model_path})")

# Load the best model
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"[ERROR] Best model file not found: {best_model_path}")

best_model = lgb.Booster(model_file=best_model_path)
print("[INFO] Loaded best model successfully.")

# Load test set
test_data_path = "./K_fold_LGBM/test_split.feather"

if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"[ERROR] Test data file not found: {test_data_path}")

test_data = pd.read_feather(test_data_path)
X_test = test_data.drop(columns=["meter_reading"])
y_test = np.log1p(test_data["meter_reading"])  # Log transform target

del test_data  # Free memory
gc.collect()

# Make predictions
y_test_pred = best_model.predict(X_test)

# Convert predictions and ground truth back to original scale
y_test_actual = np.expm1(y_test)
y_test_pred_actual = np.expm1(y_test_pred)

# Compute evaluation metrics
# Now compute metrics on original scale
test_rmse = root_mean_squared_error(y_test_actual, y_test_pred_actual)
test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
test_mse = mean_squared_error(y_test_actual, y_test_pred_actual)
test_r2 = r2_score(y_test_actual, y_test_pred_actual)

# Convert predictions back from log scale
test_results = pd.DataFrame({
    "Actual": np.expm1(y_test),  # Convert log-transformed target back
    "Predicted": np.expm1(y_test_pred)
})

# Compute RMSLE (on original scale)
rmsle = mean_squared_log_error(test_results["Actual"], test_results["Predicted"]) ** 0.5

print(f"\nEvaluation Metrics on Test Set:")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"R^2 Score: {test_r2:.4f}")
print(f"RMSLE: {rmsle:.4f}")

# Sanity Check
print("Predicted Range:", test_results["Predicted"].min(), "-", test_results["Predicted"].max())
print("Actual Range:", test_results["Actual"].min(), "-", test_results["Actual"].max())

# Save results
results_path = "./Results_and_Plots/test_predictions.csv"
test_results.to_csv(results_path, index=False)
print(f"[INFO] Test predictions saved to: {results_path}")

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=test_results["Actual"], y=test_results["Predicted"], alpha=0.5)
plt.plot([test_results["Actual"].min(), test_results["Actual"].max()],
         [test_results["Actual"].min(), test_results["Actual"].max()],
         color='red', linestyle='dashed')
plt.xlabel("Actual Meter Reading")
plt.ylabel("Predicted Meter Reading")
plt.title("Actual vs. Predicted Values")
plt.savefig("./Results_and_Plots/actual_vs_predicted.png")
plt.show()

# Residual Analysis
residuals = test_results["Actual"] - test_results["Predicted"]
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.savefig("./Results_and_Plots/residual_distribution.png")
plt.show()

# Residuals vs. Fitted Values Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=test_results["Predicted"], y=residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.savefig("./Results_and_Plots/residuals_vs_fitted.png")
plt.show()

# Distribution Comparison
plt.figure(figsize=(10, 6))
sns.histplot(test_results["Actual"], color="blue", label="Actual", kde=True, stat="density", bins=100, alpha=0.6)
sns.histplot(test_results["Predicted"], color="orange", label="Predicted", kde=True, stat="density", bins=100, alpha=0.6)
plt.xlabel("Meter Reading (kWh)")
plt.ylabel("Density")
plt.title("Distribution of Actual vs. Predicted Meter Readings")
plt.legend()
plt.tight_layout()
plt.savefig("./Results_and_Plots/distribution_comparison.png")
plt.show()

# Feature Importance Plot (if feature importance is available)
try:
    importance = best_model.feature_importance()
    feature_names = X_test.columns
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(20))  # Top 20 features
    plt.title("Top 20 Feature Importances")
    plt.savefig("./Results_and_Plots/feature_importance.png")
    plt.show()
except Exception as e:
    print(f"[WARNING] Unable to generate feature importance plot: {e}")

print("[INFO] Model inference and evaluation completed successfully.")