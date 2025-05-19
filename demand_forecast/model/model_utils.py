import pickle
import xgboost as xgb
import pandas as pd
from data.data_utils import download_file, generate_future_data
from app.config import MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS
from sklearn.preprocessing import LabelEncoder

def download_file(file_path, url):
    """Safely download a file from Google Drive to a given path."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {dir_path}: {e}")

    if not os.path.exists(file_path):
        import gdown
        try:
            gdown.download(url, file_path, quiet=False)
        except Exception as e:
            print(f"Download failed for {file_path}: {e}")
    else:
        print(f"{file_path} already exists.")


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
