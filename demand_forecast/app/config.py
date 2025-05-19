# app/config.py

DATA_PATH = "data/"

# Google Drive file IDs for each dataset
your_file_id_for_stores_csv = '1OUyap9CGF59ewdkN3DM6NWw3nv-hT5pH'
your_file_id_for_items_csv = '1KPu1RTEQQCA_GoNEapFtq8oZTPmyvMFx'
your_file_id_for_transactions_csv = '1FSrI_JqbzSmZ0c4NBmfiMT-kYQG31xUS'
your_file_id_for_oil_csv = '15uMxMXXSYEv14Yv4cLyKuiCEuO1TmXrU'
your_file_id_for_holidays_csv = '1fNj8v8kDmfdslU_M9FgmmV_FPqd1tIN1'
your_file_id_for_train_csv = '1KsbtNzrfL5ZTFLpnaCyEUAmRzA2EBDLz'

# Google Drive download links for each file (use these for gdown or programmatic download!)
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}"
}

MODEL_PATH = 'model/'

# Google Drive file ID for your XGBoost model
your_file_id_for_xgboost_model_xgb = "1Sqw5QuMy9cZ-7MmRxYKg59Kjvho-Dsil"

# Google Drive download link for the model file
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost_model": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model_xgb}"
}

