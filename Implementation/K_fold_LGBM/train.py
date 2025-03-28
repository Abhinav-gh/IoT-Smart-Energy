from datetime import datetime
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import sys

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from Implementation.Utils.__utils__ import timer

# Load processed training data
train = pd.read_feather("../Processed_Data/train_processed.feather")

def reduce_mem_usage(df, use_float16=False):
    """ Reduce memory usage by converting columns to optimal dtypes. """
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

with timer("Encoding categorical variables"):
    print(f"[INFO] Encoding categorical variables...")
    for col in train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))

with timer("Reducing memory usage"):
    print(f"[INFO] Reducing memory usage...")
    train = reduce_mem_usage(train, use_float16=True)
    gc.collect()

# Train-test split (80% train, 20% test)
train_data, test_data = train_test_split(train, test_size=0.2, random_state=42)

# Save test split for future inference
os.makedirs("./K_fold_LGBM", exist_ok=True)
test_data.to_feather("./K_fold_LGBM/test_split.feather")
print("[INFO] Test split saved as test_split.feather")
del test_data  # Free memory
gc.collect()

# Prepare dataset
X_train_full = train_data.drop(columns=["meter_reading"])
y_train_full = np.log1p(train_data["meter_reading"])  # Log transform target


# Define K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define LightGBM parameters
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse",
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=25)
]

best_rmse = float("inf")
best_model = None
best_model_fold = -1
rmse_scores = []
fold_models_info = {}

with timer("Training K-Fold LightGBM Model"):
    print(f"[INFO] Training K-Fold LightGBM Model...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"Training Fold {fold + 1}...")

        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        train_data_lgb = lgb.Dataset(X_train, label=y_train)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with timer(f"Training the LGB model for Fold {fold + 1}"):
            model = lgb.train(
                params,
                train_data_lgb,
                num_boost_round=1000,
                valid_sets=[train_data_lgb, val_data_lgb],
                callbacks=callbacks
            )

        with timer("Prediction and Model Evaluation"):
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            rmse_scores.append(rmse)
            print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_fold = fold + 1
                print(f"[INFO] Best model found in Fold {fold + 1} with RMSE: {best_rmse:.4f}")

        # Save the model
        model_path = f"./K_fold_LGBM/lgbm_model_fold{fold + 1}.txt"
        if os.path.exists(model_path):
            os.remove(model_path)  # Overwrite if exists
        model.save_model(model_path)
        print(f"[INFO] Saved model for Fold {fold + 1}: {model_path}")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        fold_models_info[f"fold_{fold + 1}"] = {
            "model_path": model_path,
            "rmse": rmse,
            "training_start_time": start_time,
            "training_end_time": end_time,
            "best_iteration": model.best_iteration
        }
        del X_train, X_val, y_train, y_val, train_data_lgb, val_data_lgb
        gc.collect()

avg_rmse = np.mean(rmse_scores)
print(f"\nAverage RMSE across folds: {avg_rmse:.4f}")
print(f"Best RMSE: {best_rmse:.4f} achieved in Fold {best_model_fold}")

# Save model information to JSON
model_info_path = "./K_fold_LGBM/model_info.json"
model_info = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_folds": kf.n_splits,
    "shuffle": kf.shuffle,
    "random_state": kf.random_state,
    "lgbm_params": params,
    "average_rmse": avg_rmse,
    "best_rmse": best_rmse,
    "best_model_fold": best_model_fold,
    "fold_models": fold_models_info
}

with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"[INFO] Model information saved to: {model_info_path}")

