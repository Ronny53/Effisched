import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import metrics

# Set current time
CURRENT_TIME = datetime.now()


# Load dataset
df = pd.read_csv("flexible_tasks_dataset.csv")

# Data Preprocessing
df['deadline'] = pd.to_datetime(df['deadline'], format="%Y-%m-%d %H:%M")
df['hours_remaining'] = (df['deadline'] - CURRENT_TIME).dt.total_seconds() / 3600
df['hours_remaining'] = df['hours_remaining'].clip(lower=1)
df['time_pressure'] = df['duration'] / df['hours_remaining']

X = df[["duration", "hours_remaining", "time_pressure"]]
y = df["priority"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
base_models = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("gbr", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ("svr", SVR(kernel='rbf', C=100, gamma=0.1))
]

# Meta-model
meta_model = Ridge(alpha=1.0)

# Stacking Regressor
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1
)

# Pipeline with scaling + stacking
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("stacking", stacked_model)
])

# Train the stacked model
pipeline.fit(X_train, y_train)

# Evaluating model
y_pred = pipeline.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print(f"Model performance:\n - MAE: {mae:.2f}\n - RMSE: {rmse:.2f}")

# Dumping the model
with open("priority_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

if (__name__ == "__main__"):

    # Load the model
    with open("priority_model.pkl", "rb") as f:
        loaded_pipeline = pickle.load(f)

    # Predict with loaded model
    sample = X_test.head(5).copy()
    sample["predicted_priority"] = loaded_pipeline.predict(sample).round().astype(int)
    print("\nSample predictions with stacked model:")
    print(sample)
    print(y_test.head(5))

    cv_mae = -cross_val_score(
        pipeline, X, y,
        scoring='neg_mean_absolute_error',
        cv=5, n_jobs=-1
    )
    print("5-fold CV MAE:", np.round(cv_mae, 3))
    print("Mean CV MAE:", np.round(cv_mae.mean(), 3))
