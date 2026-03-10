"""
Part 2: Predictive Analytics — Full Training Pipeline
=====================================================
This script performs:
  2.1  Train/test split (70/30, random_state=42)
  2.2  Baseline models (Linear, Lasso, Ridge) — train on train set, evaluate on test set
  2.3  CART with GridSearchCV (max_depth, min_samples_leaf)
  2.4  Random Forest with GridSearchCV (n_estimators, max_depth)
  2.5  LightGBM with GridSearchCV (n_estimators, max_depth, learning_rate)
  2.6  MLP Neural Network (Keras, 2 hidden layers, ReLU, Adam, MSE)
  2.7  Save all results for model comparison

Run:  python train_pipeline.py
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ──────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

def evaluate_on_test(model, X_test_data, y_test_data, model_name):
    """Compute MAE, RMSE, R² on a held-out test set."""
    y_pred = model.predict(X_test_data)
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
    return {
        'model': model_name,
        'test_rmse': float(np.sqrt(mean_squared_error(y_test_data, y_pred))),
        'test_mae': float(mean_absolute_error(y_test_data, y_pred)),
        'test_r2': float(r2_score(y_test_data, y_pred)),
        'y_pred': y_pred
    }

# ──────────────────────────────────────────────────────────
# 2.1 — LOAD DATA & TRAIN/TEST SPLIT
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.1: Data Preparation — Train/Test Split")
print("=" * 60)

X = np.load(f"{ARTIFACT_DIR}/X.npy")               # (80000, 21)
X_scaled = np.load(f"{ARTIFACT_DIR}/X_scaled.npy")  # (80000, 21)
y = np.load(f"{ARTIFACT_DIR}/y.npy")                # (80000,)
with open(f"{ARTIFACT_DIR}/config.json") as f:
    config = json.load(f)
feature_cols = config['feature_cols']

print(f"  Full dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"  Target: num_skills (mean={y.mean():.2f}, std={y.std():.2f})")

# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_scaled_train, X_scaled_test, _, _ = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42
)

print(f"  Train set: {X_train.shape[0]:,} samples (70%)")
print(f"  Test  set: {X_test.shape[0]:,} samples (30%)")

# Save split data
np.save(f"{ARTIFACT_DIR}/X_train.npy", X_train)
np.save(f"{ARTIFACT_DIR}/X_test.npy", X_test)
np.save(f"{ARTIFACT_DIR}/X_scaled_train.npy", X_scaled_train)
np.save(f"{ARTIFACT_DIR}/X_scaled_test.npy", X_scaled_test)
np.save(f"{ARTIFACT_DIR}/y_train.npy", y_train)
np.save(f"{ARTIFACT_DIR}/y_test.npy", y_test)
print("  Saved train/test splits to artifacts/\n")

test_results = {}

# ──────────────────────────────────────────────────────────
# 2.2 — BASELINE: LINEAR REGRESSION, LASSO, RIDGE
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.2: Baseline Models (Linear, Lasso, Ridge)")
print("=" * 60)

baseline_models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42),
}

for name, model in baseline_models.items():
    t0 = time.time()
    model.fit(X_scaled_train, y_train)
    result = evaluate_on_test(model, X_scaled_test, y_test, name)
    elapsed = time.time() - t0
    test_results[name] = result
    safe = name.lower().replace(' ', '_')
    joblib.dump(model, f"{ARTIFACT_DIR}/model_{safe}_split.joblib")
    print(f"  {name:25s}  RMSE={result['test_rmse']:.4f}  MAE={result['test_mae']:.4f}  "
          f"R²={result['test_r2']:.4f}  ({elapsed:.1f}s)")

print()

# ──────────────────────────────────────────────────────────
# 2.3 — CART WITH GRIDSEARCHCV
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.3: CART — GridSearchCV")
print("=" * 60)

cart_param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20, 50]
}
print(f"  Param grid: {cart_param_grid}")
print(f"  Total fits: {4*4*5} = {4*4} combos x 5 folds")

t0 = time.time()
cart_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid=cart_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    return_train_score=True,
    verbose=0
)
cart_grid.fit(X_train, y_train)
elapsed = time.time() - t0

best_cart = cart_grid.best_estimator_
cart_result = evaluate_on_test(best_cart, X_test, y_test, 'CART (Tuned)')
test_results['CART (Tuned)'] = cart_result

# Save artifacts
joblib.dump(best_cart, f"{ARTIFACT_DIR}/model_cart_tuned.joblib")
with open(f"{ARTIFACT_DIR}/grid_cart_best_params.json", 'w') as f:
    json.dump(cart_grid.best_params_, f, indent=2)

# Save CV results as serializable dict
cv_df = pd.DataFrame(cart_grid.cv_results_)
cv_df.to_json(f"{ARTIFACT_DIR}/grid_cart_cv_results.json")

print(f"  Best params: {cart_grid.best_params_}")
print(f"  Best CV RMSE: {np.sqrt(-cart_grid.best_score_):.4f}")
print(f"  Test RMSE={cart_result['test_rmse']:.4f}  MAE={cart_result['test_mae']:.4f}  "
      f"R²={cart_result['test_r2']:.4f}  ({elapsed:.1f}s)\n")

# ──────────────────────────────────────────────────────────
# 2.4 — RANDOM FOREST WITH GRIDSEARCHCV
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.4: Random Forest — GridSearchCV")
print("=" * 60)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8]
}
print(f"  Param grid: {rf_param_grid}")
print(f"  Total fits: {3*3*5} = {3*3} combos x 5 folds")

t0 = time.time()
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=1,   # RF already parallelizes internally
    return_train_score=True,
    verbose=0
)
rf_grid.fit(X_train, y_train)
elapsed = time.time() - t0

best_rf = rf_grid.best_estimator_
rf_result = evaluate_on_test(best_rf, X_test, y_test, 'Random Forest (Tuned)')
test_results['Random Forest (Tuned)'] = rf_result

joblib.dump(best_rf, f"{ARTIFACT_DIR}/model_random_forest_tuned.joblib")
with open(f"{ARTIFACT_DIR}/grid_rf_best_params.json", 'w') as f:
    json.dump(rf_grid.best_params_, f, indent=2)
pd.DataFrame(rf_grid.cv_results_).to_json(f"{ARTIFACT_DIR}/grid_rf_cv_results.json")

print(f"  Best params: {rf_grid.best_params_}")
print(f"  Best CV RMSE: {np.sqrt(-rf_grid.best_score_):.4f}")
print(f"  Test RMSE={rf_result['test_rmse']:.4f}  MAE={rf_result['test_mae']:.4f}  "
      f"R²={rf_result['test_r2']:.4f}  ({elapsed:.1f}s)\n")

# ──────────────────────────────────────────────────────────
# 2.5 — LIGHTGBM WITH GRIDSEARCHCV
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.5: LightGBM — GridSearchCV (3 hyperparameters)")
print("=" * 60)

lgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}
print(f"  Param grid: {lgb_param_grid}")
print(f"  Total fits: {3*4*3*5} = {3*4*3} combos x 5 folds")

t0 = time.time()
lgb_grid = GridSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
    param_grid=lgb_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=1,   # LightGBM parallelizes internally
    return_train_score=True,
    verbose=0
)
lgb_grid.fit(X_train, y_train)
elapsed = time.time() - t0

best_lgb = lgb_grid.best_estimator_
lgb_result = evaluate_on_test(best_lgb, X_test, y_test, 'LightGBM (Tuned)')
test_results['LightGBM (Tuned)'] = lgb_result

joblib.dump(best_lgb, f"{ARTIFACT_DIR}/model_lightgbm_tuned.joblib")
with open(f"{ARTIFACT_DIR}/grid_lgb_best_params.json", 'w') as f:
    json.dump(lgb_grid.best_params_, f, indent=2)
pd.DataFrame(lgb_grid.cv_results_).to_json(f"{ARTIFACT_DIR}/grid_lgb_cv_results.json")

print(f"  Best params: {lgb_grid.best_params_}")
print(f"  Best CV RMSE: {np.sqrt(-lgb_grid.best_score_):.4f}")
print(f"  Test RMSE={lgb_result['test_rmse']:.4f}  MAE={lgb_result['test_mae']:.4f}  "
      f"R²={lgb_result['test_r2']:.4f}  ({elapsed:.1f}s)\n")

# ──────────────────────────────────────────────────────────
# 2.6 — MLP NEURAL NETWORK (Keras)
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.6: MLP Neural Network (Keras)")
print("=" * 60)

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("  Architecture: Input(21) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(1, Linear)")
print("  Loss: MSE | Optimizer: Adam (lr=0.001) | Early Stopping: patience=10")

model_mlp = Sequential([
    Input(shape=(X_scaled_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])

model_mlp.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

t0 = time.time()
history = model_mlp.fit(
    X_scaled_train, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)
elapsed = time.time() - t0

# Evaluate on test set
y_pred_mlp = model_mlp.predict(X_scaled_test, verbose=0).flatten()
mlp_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_mlp)))
mlp_mae = float(mean_absolute_error(y_test, y_pred_mlp))
mlp_r2 = float(r2_score(y_test, y_pred_mlp))

test_results['MLP'] = {
    'model': 'MLP',
    'test_rmse': mlp_rmse,
    'test_mae': mlp_mae,
    'test_r2': mlp_r2,
    'y_pred': y_pred_mlp
}

# Save model and history
model_mlp.save(f"{ARTIFACT_DIR}/model_mlp.keras")
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(f"{ARTIFACT_DIR}/mlp_history.json", 'w') as f:
    json.dump(history_dict, f)

print(f"\n  Epochs trained: {len(history.history['loss'])}")
print(f"  Test RMSE={mlp_rmse:.4f}  MAE={mlp_mae:.4f}  R²={mlp_r2:.4f}  ({elapsed:.1f}s)\n")

# ──────────────────────────────────────────────────────────
# 2.7 — SAVE ALL TEST RESULTS
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2.7: Saving Combined Test Results")
print("=" * 60)

# Save summary (without large y_pred arrays)
test_results_summary = {}
for name, res in test_results.items():
    test_results_summary[name] = {
        'test_rmse': float(res['test_rmse']),
        'test_mae': float(res['test_mae']),
        'test_r2': float(res['test_r2']),
    }

with open(f"{ARTIFACT_DIR}/test_results.json", 'w') as f:
    json.dump(test_results_summary, f, indent=2)

# Save predictions for scatter plots
for name, res in test_results.items():
    safe = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    np.save(f"{ARTIFACT_DIR}/y_pred_test_{safe}.npy", np.array(res['y_pred']))
np.save(f"{ARTIFACT_DIR}/y_test_actual.npy", y_test)

print("\n  === FINAL TEST-SET COMPARISON (30% hold-out) ===")
print(f"  {'Model':30s}  {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}")
print("  " + "-" * 60)
for name, res in test_results_summary.items():
    print(f"  {name:30s}  {res['test_rmse']:8.4f}  {res['test_mae']:8.4f}  {res['test_r2']:8.4f}")

best_name = min(test_results_summary, key=lambda k: test_results_summary[k]['test_rmse'])
print(f"\n  Best model (lowest RMSE): {best_name}")
print("\n  All artifacts saved to artifacts/")
print("  Done!")
