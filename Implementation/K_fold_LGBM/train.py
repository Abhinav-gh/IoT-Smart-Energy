import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import sys

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from Implementation.Utils.__utils__ import timer

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
                    df[col] = df[col].astype(np.float32)  # Avoid overflow by keeping float32
                else:
                    df[col] = df[col].astype(np.float64)  # More precision if needed
    return df

# Load processed data
train = pd.read_feather("../Processed_Data/train_processed.feather")
test = pd.read_feather("../Processed_Data/test_processed.feather")

with timer("Encoding categorical variables"):
    print(f"[INFO] Encoding categorical variables...")
    # Encode categorical variables
    for col in train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(all_values)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

with timer("Reducing memory usage"):
    print(f"[INFO] Reducing memory usage...")
    # Reduce memory usage safely
    train = reduce_mem_usage(train, use_float16=True)
    test = reduce_mem_usage(test, use_float16=True)
    gc.collect()

# Prepare dataset
X = train.drop(columns=["meter_reading"])
y = np.log1p(train["meter_reading"])  # Log transform target

# Define K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define your parameters
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse",
}
# Create callback functions
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=25)
]

# Create directory to store models
os.makedirs("./K_fold_LGBM", exist_ok=True)

rmse_scores = []

with timer("Training K-Fold LightGBM Model"):
    print(f"[INFO] Training K-Fold LightGBM Model...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training Fold {fold + 1}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        # Train the model
        with timer("Training the LGB model"):
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, val_data],
                callbacks=callbacks
            )

        with timer("Prediction and Model Evaluation"):
            # Predict and evaluate
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            rmse_scores.append(rmse)
            print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

        with timer("Saving the model"):
            # Save model
            model.save_model(f"./K_fold_LGBM/lgbm_model_fold{fold + 1}.txt")

            del X_train, X_val, y_train, y_val, train_data, val_data
            gc.collect()

# Print average RMSE across folds
avg_rmse = np.mean(rmse_scores)
print(f"\nAverage RMSE across folds: {avg_rmse:.4f}")

print("Training complete! Models saved.")
