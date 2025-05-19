# data/data_utils.py
import pandas as pd
import os
import gdown
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS
from sklearn.preprocessing import LabelEncoder

def download_file(file_path, url):
    """Safely download from Google Drive with gdown, creating directories if needed."""
    import os
    import gdown

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download using fuzzy=True to handle confirmation pages
        result = gdown.download(url, file_path, quiet=False, fuzzy=True)

        if result is None or not os.path.exists(file_path):
            raise FileNotFoundError(f"gdown could not download: {url}")
        else:
            print(f"[INFO] Downloaded: {file_path}")

    except Exception as e:
        raise FileNotFoundError(f"Failed to download {url} to {file_path} — {e}")


def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads filtered CSV files into DataFrames."""

    # Only include the files we actually use
    files = {
        "stores": os.path.join(data_path, "stores.csv"),
        "items": os.path.join(data_path, "items.csv"),
        "train": os.path.join(data_path, "train.csv")
    }

    # Download the files if they don't already exist locally
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    # Load each downloaded CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])
    df_items = pd.read_csv(files["items"])

# Filter stores to only include relevant store_nbrs
    store_ids = [24, 26, 27, 28, 30, 32, 34, 35, 51, 36]
    df_stores = df_stores[df_stores['store_nbr'].isin(store_ids)].copy()
    # Select the same items as for "Classical methods":
    item_ids = [115611,  115892,  116017,  153267,  165550,  165551,  165704,
        165988,  168927,  168930,  207857,  214381,  215352,  219150,
        220435,  222879,  223136,  223434,  257847,  258396,  261700,
        265254,  265257,  265559,  305080,  305229,  314384,  315176,
        315179,  315460,  320682,  323013,  358231,  364413,  364606,
        364738,  368140,  368628,  410257,  414353,  414620,  414752,
        418235,  459804,  464374,  464536,  510054,  514172,  514242,
        518091,  518094,  554047,  559870,  559873,  564274,  564533,
        567623,  581078,  587069,  621300,  641042,  671039,  692531,
        730259,  749421,  749720,  759893,  769312,  789224,  801217,
        802833,  804503,  807493,  812726,  819932,  830625,  839362,
        839363,  841841,  841842,  843585,  847859,  847863,  848765,
        850333,  862454,  876663,  911429,  913363,  938566,  938567,
        938570,  938576,  939661,  939662,  964752, 1036689, 1037857,
       1047679, 1047681, 1047685, 1047743, 1047772, 1047773, 1047775,
       1047790, 1052563, 1057033, 1066900, 1066901, 1074327, 1084437,
       1084881, 1105212, 1109326,  111397,  115894,  315221,  770449,
        874593,  165594,  269029,  368136,  968432, 1109325,  155500,
        638977,  651525,  819934, 1012473,  819933, 1114566, 1114567,
       1146786, 1146795, 1146801, 1146802, 1143691, 1143685, 1157462,
       1158720, 1161572, 1157329, 1159415, 1162382, 1463810, 1463860,
       1463881, 1464086, 1464088]  # ToDo: add more items (e.g., all items from a family)
    df_items = df_items[df_items['item_nbr'].isin(item_ids)].copy()

	
    max_date = '2014-04-01'
    filtered_chunks = []
    chunk_size = 10**6

    for chunk in pd.read_csv(files["train"], chunksize=chunk_size):
        chunk_filtered = chunk[
            (chunk['store_nbr'].isin(store_ids)) &
            (chunk['item_nbr'].isin(item_ids)) &
            (chunk['date'] < max_date)
        ]
        filtered_chunks.append(chunk_filtered)
        del chunk

    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks

    df_filtered = df_filtered.groupby(['store_nbr', 'item_nbr', 'date'])['unit_sales'].sum().reset_index()

    return df_stores, df_items, df_filtered

def preprocess_input_data(store_id, item_id, split_date, df_stores, df_items, df_filtered):
    """Preprocesses input data into a format suitable for model prediction."""
    
  
    # Convert the 'date' column to datetime format for easy manipulation
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    split_date = pd.to_datetime(split_date)  # Convert the split_date to datetime

    # Get the minimum and maximum dates in the dataset to create a full date range
    min_date = df_filtered['date'].min()
    max_date = df_filtered['date'].max()
    print("Before filtering", min_date.date(), max_date.date())

    # Filter the dataset to only include dates after the specified split date
    df_filtered = df_filtered[df_filtered['date'] >= split_date]  # Filter rows by date
    
    # Group by store, item, and date, then aggregate (sum) the unit_sales for each group
    df_filtered = df_filtered.groupby(['store_nbr', 'item_nbr', 'date']).sum()['unit_sales'].reset_index()

    # Create a full date range covering all days between the min and max dates
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Create an empty DataFrame to store the final result
    df_filled = pd.DataFrame()

		# Add missing 0 sales
    # Iterate through each store and item combination in the filtered data 
    for (store, item), group in df_filtered.groupby(['store_nbr', 'item_nbr']):
        # Set 'date' as index and sort by date
        group.set_index('date', inplace=True)
        group = group.sort_index()

        # Reindex the group to fill in missing dates with 0 sales
        group = group.reindex(full_date_range, fill_value=0)

        # Add the store and item numbers back to each row
        group['store_nbr'] = store
        group['item_nbr'] = item

        # Ensure that missing sales values are filled with 0
        group['unit_sales'] = group['unit_sales'].fillna(0)

        # Append this group's data to the final DataFrame
        df_filled = pd.concat([df_filled, group])

    # Reset the index so that 'date' is a regular column again
    df_filled.reset_index(inplace=True)
    df_filled.rename(columns={'index': 'date'}, inplace=True)

    # Feature engineering: extract date-related features
    df_filled['month'] = df_filled['date'].dt.month  # Extract the month from the date
    df_filled['day'] = df_filled['date'].dt.day  # Extract the day from the date
    df_filled['weekofyear'] = df_filled['date'].dt.isocalendar().week  # Extract the ISO week of the year
    df_filled['dayofweek'] = df_filled['date'].dt.dayofweek  # Extract the day of the week (0=Monday, 6=Sunday)
    
    # Create rolling features for unit_sales (7-day rolling mean and standard deviation)
    df_filled['rolling_mean'] = df_filled['unit_sales'].rolling(window=7).mean()
    df_filled['rolling_std'] = df_filled['unit_sales'].rolling(window=7).std()

    # Create lag features (sales from the previous day, previous week)
    df_filled['lag_1'] = df_filled['unit_sales'].shift(1)  # Sales from the previous day
    df_filled['lag_7'] = df_filled['unit_sales'].shift(7)  # Sales from 7 days ago
    df_filled['lag_30'] = df_filled['unit_sales'].shift(30)  # Sales from 30 days ago

    # Drop any rows with NaN values after creating lag features (for rows without enough data)
    df_filled.dropna(inplace=True)

    # Merge the filled DataFrame with store and item data to include more information
    df_filled = df_filled.merge(df_stores, on='store_nbr', how='left').merge(df_items, on='item_nbr', how='left')

    # Encode categorical columns with LabelEncoder to convert them into numeric format
    for col in ['city', 'state', 'type', 'family', 'class']:  # List of categorical columns to encode
        le = LabelEncoder()  # Initialize the label encoder
        df_filled[col] = le.fit_transform(df_filled[col])  # Apply the encoder to the column

    # Sort the final DataFrame by store number, item number, and date
    df_filled = df_filled.sort_values(by=['store_nbr', 'item_nbr', 'date'])

    # Return the preprocessed and feature-engineered DataFrame
    return df_filled
	
def generate_future_data(store_nbr, item_nbr, start_date, days_ahead, df_train, df_stores, df_items):
    # 1) Fetch historical data
    hist = df_train[
        (df_train['store_nbr'] == store_nbr) & 
        (df_train['item_nbr'] == item_nbr)
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

    # 3) Use last 30 days of history to build lag features
    hist['date'] = pd.to_datetime(hist['date'])
    hist = hist.sort_values('date').tail(30)
    vals = hist['unit_sales'].values
    pad_width = max(days_ahead - len(vals), 0)

    padded = np.pad(vals, (0, pad_width), mode='edge')
    future_df['lag_1'] = padded[-days_ahead:]

    def make_lag(arr, shift):
        shifted = np.pad(arr, (shift, 0), mode='edge')[:-shift]
        return np.pad(shifted, (0, pad_width), mode='edge')[-days_ahead:]

    future_df['lag_7'] = make_lag(vals, 7)
    future_df['lag_14'] = make_lag(vals, 14)
    future_df['lag_30'] = make_lag(vals, 30)

    # 4) Rolling stats and pct change
    roll7 = pd.Series(vals).rolling(7).mean().fillna(method='bfill').values
    std7 = pd.Series(vals).rolling(7).std().fillna(method='bfill').values
    pct7 = pd.Series(vals).pct_change(7).fillna(0).values
    future_df['unit_sales_7d_avg'] = np.pad(roll7, (0, pad_width), mode='edge')[-days_ahead:]
    future_df['roll7_std'] = np.pad(std7, (0, pad_width), mode='edge')[-days_ahead:]
    future_df['pct_chg_7d'] = np.pad(pct7, (0, pad_width), mode='edge')[-days_ahead:]

    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6])

    # 5) Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in ['city', 'state', 'cluster', 'family', 'class', 'store_type']:
        if col in future_df:
            future_df[col] = LabelEncoder().fit_transform(future_df[col].astype(str))

    # 6) Fill missing optional feature
    if 'perishable' not in future_df:
        future_df['perishable'] = 0

    return future_df
