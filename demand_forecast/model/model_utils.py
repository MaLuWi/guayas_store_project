# app/model_utils.py

import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.config import MODEL_PATH
from data.data_utils import generate_future_data

def load_model():
    """Load the pickled XGBoost model directly from model/model.pkl"""
    model_fp = os.path.join(MODEL_PATH, "model.pkl")
    with open(model_fp, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, input_data: pd.DataFrame):
    """Drop non-feature cols and run model.predict."""
    drop_cols = [c for c in ("date", "unit_sales") if c in input_data.columns]
    X = input_data.drop(columns=drop_cols)
    return model.predict(X)

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
    future_df["prediction"] = model.predict(future_df[features])

    # 4) Return date + prediction
    return future_df[["date", "prediction"]]
