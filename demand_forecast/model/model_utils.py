import os
import pickle
import xgboost as xgb
import pandas as pd
from data.data_utils import download_file, generate_future_data
from app.config import MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS
from sklearn.preprocessing import LabelEncoder

def load_model(model_path=MODEL_PATH):
    files = {"xgboost_model": f"{model_path}model.xgb"}
    for key, file_path in files.items():
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        download_file(file_path, GOOGLE_DRIVE_LINKS_MODELS[key])
    
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(files["xgboost_model"])
    return xgboost_model

def predict(model, input_data):
    drop_cols = [c for c in ('date','unit_sales') if c in input_data]
    return model.predict(input_data.drop(columns=drop_cols))

def forecast_timeseries(model, store_id, item_id,
                        start_date, horizon,
                        df_train, df_stores, df_items):
    # 1) assemble future_df however you already do (generate_future_data, etc.)
    future_df = generate_future_data(store_id, item_id,
                                     start_date, horizon,
                                     df_train, df_stores, df_items)
    if future_df.empty:
        return future_df

    # 2) list out the exact feature columns your model expects
    features = [
        "onpromotion", "year", "month", "day", "day_of_week",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "unit_sales_7d_avg", "roll7_std", "pct_chg_7d", "is_weekend",
        # plus any encoded categorical cols:
        "city", "state", "cluster", "family", "class", "store_type"
    ]

    # 3) predict and attach to the frame
    future_df["prediction"] = model.predict(future_df[features])

    # 4) return only date + prediction (or however you prefer)
    return future_df[["date", "prediction"]]

