import os
import xgboost as xgb
import pandas as pd
from data.data_utils import generate_future_data
from app.config import MODEL_PATH
from sklearn.preprocessing import LabelEncoder


def load_model(model_path=MODEL_PATH):
    model_file = os.path.join(model_path, "model.xgb")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    return model

def predict(model, input_data):
    drop_cols = [c for c in ('date','unit_sales') if c in input_data]
    return model.predict(input_data.drop(columns=drop_cols))

def forecast_timeseries(model, store_id, item_id, start_date, horizon, df_train, df_stores, df_items):
    future_df = generate_future_data(store_id, item_id, start_date, horizon, df_train, df_stores, df_items)
    if future_df.empty:
        return pd.DataFrame()
    final_feature_list = [
        'store_nbr','item_nbr','onpromotion','year','month','day','day_of_week',
        'unit_sales_7d_avg','lag_1','lag_7','lag_14','lag_30',
        'roll7_std','pct_chg_7d','is_weekend',
        'city','state','cluster','family','class','perishable'
    ]
    features = [c for c in final_feature_list if c in future_df.columns]
    preds = model.predict(future_df[features])
    dates = pd.date_range(start=start_date, periods=horizon, freq='D')
    return pd.DataFrame({"date": dates, "prediction": preds})
