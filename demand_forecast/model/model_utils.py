import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from app.config import MODEL_PATH
from data.data_utils import generate_future_data

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    model = xgb.Booster()
    model.load_model(path)
    return model

def predict(model, input_data: pd.DataFrame):
    """Prepare features and predict using XGBoost Booster."""
    drop_cols = [c for c in ("date", "unit_sales") if c in input_data.columns]
    X = input_data.drop(columns=drop_cols)

    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)

def forecast_timeseries(
    model,
    store_id: int,
    item_id: int,
    start_date,
    horizon: int,
    df_train: pd.DataFrame,
    df_stores: pd.DataFrame,
    df_items: pd.DataFrame
) -> pd.DataFrame:
    # 1) Build feature-frame
    future_df = generate_future_data(
        store_id, item_id,
        start_date, horizon,
        df_train, df_stores, df_items
    )
    if future_df.empty:
        return future_df

    # 2) Exact list of feature columns
    features = [
        "onpromotion", "year", "month", "day", "day_of_week",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "unit_sales_7d_avg", "roll7_std", "pct_chg_7d", "is_weekend",
        "city", "state", "cluster", "family", "class", "store_type"
    ]
    features = [c for c in features if c in future_df.columns]

    # 3) Predict
    dmatrix = xgb.DMatrix(future_df[features])
    future_df["prediction"] = model.predict(dmatrix)

    # 4) Return date + prediction
    return future_df[["date", "prediction"]]

