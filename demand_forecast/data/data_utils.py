# data/data_utils.py
import pandas as pd
import numpy as np
import os
import gdown
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS
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


def load_data(data_path=DATA_PATH):
    """Load and filter datasets for Guayas region and top store-item combos."""
    files = {
        "stores": os.path.join(data_path, "stores.csv"),
        "items": os.path.join(data_path, "items.csv"),
        "train": os.path.join(data_path, "train.csv"),
    }
    for key, fp in files.items():
        download_file(fp, GOOGLE_DRIVE_LINKS[key])

    df_stores = pd.read_csv(files["stores"])
    df_stores = df_stores[df_stores['state'] == 'Guayas']

    df_items = pd.read_csv(files["items"])

    max_date = pd.to_datetime("2014-04-01")
    df_sample = pd.read_csv(files["train"], usecols=["store_nbr", "item_nbr", "date", "unit_sales"])
    df_sample["date"] = pd.to_datetime(df_sample["date"])
    df_sample = df_sample[df_sample['date'] < max_date]
    df_sample = df_sample[df_sample['store_nbr'].isin(df_stores['store_nbr'])]

    top_combos = (
        df_sample.groupby(["store_nbr", "item_nbr"])['unit_sales']
        .sum().nlargest(100)
        .reset_index()[['store_nbr', 'item_nbr']]
    )
    combos = set(map(tuple, top_combos.values))

    filtered = []
    for chunk in pd.read_csv(files["train"], parse_dates=["date"], chunksize=10**6):
        mask = (
            (chunk['date'] < max_date)
            & chunk.apply(lambda r: (r['store_nbr'], r['item_nbr']) in combos, axis=1)
        )
        filtered.append(chunk.loc[mask, ['store_nbr', 'item_nbr', 'date', 'unit_sales']])
    df_train = pd.concat(filtered, ignore_index=True)

    return df_stores, df_items, df_train

def generate_future_data(store_nbr, item_nbr, start_date, days_ahead, df_train, df_stores, df_items):
    # 1) Fetch historical data
    hist = df_train[
        (df_train['store_nbr'] == store_nbr)
        & (df_train['item_nbr'] == item_nbr)
    ].copy()
    if hist.empty:
        return pd.DataFrame()

    # 2) Build base DataFrame for future dates
    future_dates = pd.date_range(start=start_date, periods=days_ahead, freq='D')
    future_df = pd.DataFrame({
        'date': future_dates,
        'store_nbr': store_nbr,
        'item_nbr': item_nbr,
        'onpromotion': False,
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day,
        'day_of_week': future_dates.dayofweek
    })
    future_df = future_df.merge(df_stores, on='store_nbr', how='left')
    future_df = future_df.merge(df_items, on='item_nbr', how='left')

    # 3) Take last 30 days of history
    hist['date'] = pd.to_datetime(hist['date'])
    hist = hist.sort_values('date').tail(30)
    vals = hist['unit_sales'].values
    pad_width = max(days_ahead - len(vals), 0)

    # 4) Create lag features
    padded = np.pad(vals, (0, pad_width), mode='edge')
    future_df['lag_1'] = padded[-days_ahead:]
    def make_lag(arr, shift):
        shifted = np.pad(arr, (shift, 0), mode='edge')[:-shift]
        return np.pad(shifted, (0, pad_width), mode='edge')[-days_ahead:]
    future_df['lag_7'] = make_lag(vals, 7)
    future_df['lag_14'] = make_lag(vals, 14)
    future_df['lag_30'] = make_lag(vals, 30)

    # 5) Rolling statistics and percentage change
    roll7 = pd.Series(vals).rolling(7).mean().fillna(method='bfill').values
    std7 = pd.Series(vals).rolling(7).std().fillna(method='bfill').values
    pct7 = pd.Series(vals).pct_change(7).fillna(0).values
    future_df['unit_sales_7d_avg'] = np.pad(roll7, (0, pad_width), mode='edge')[-days_ahead:]
    future_df['roll7_std'] = np.pad(std7, (0, pad_width), mode='edge')[-days_ahead:]
    future_df['pct_chg_7d'] = np.pad(pct7, (0, pad_width), mode='edge')[-days_ahead:]

    # 6) Is weekend?
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6])

    # 7) Encode categorical columns
    for col in ['city', 'state', 'cluster', 'family', 'class', 'store_type']:
        if col in future_df:
            future_df[col] = LabelEncoder().fit_transform(future_df[col].astype(str))

    return future_df
