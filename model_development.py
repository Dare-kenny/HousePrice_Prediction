import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "train.csv"  # put Kaggle train.csv here
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")

# Pick 6 smartest numeric features (no categorical encoding needed)
FEATURES = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "FullBath"]
TARGET = "SalePrice"


def main():
    # 1) Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Put Kaggle 'train.csv' in the project root."
        )

    df = pd.read_csv(DATA_PATH)

    # 2) Data preprocessing
    # a) Handling missing values -> via SimpleImputer
    # b) Feature selection -> FEATURES list
    # c) Encoding categorical variables -> not needed (we avoided Neighborhood)
    # d) Feature scaling -> not required for RandomForest

    # Keep only required columns
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")

    X = df[FEATURES]
    y = df[TARGET]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Algorithm: Random Forest Regressor (robust, strong baseline)
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 4) Train
    model.fit(X_train, y_train)

    # 5) Evaluate (MAE, MSE, RMSE, R²)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("\n--- Evaluation Metrics (Test Set) ---")
    print(f"MAE : {mae:,.2f}")
    print(f"MSE : {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2 : {r2:.4f}")

    # 6) Save trained model (joblib)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

    # 7) Ensure saved model reloads without retraining
    loaded = joblib.load(MODEL_PATH)
    _ = loaded["model"].predict(X_test.head(1))  # quick smoke test
    print(f"\nSaved model to: {MODEL_PATH}")
    print("Reload test: OK")


if __name__ == "__main__":
    main()
