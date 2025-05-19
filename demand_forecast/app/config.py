import os

# Set DATA_PATH to a writable directory in the Streamlit environment
DATA_PATH = '/tmp/data/'

# Google Drive file IDs for each dataset
your_file_id_for_stores_csv = '1OUyap9CGF59ewdkN3DM6NWw3nv-hT5pH'
your_file_id_for_items_csv = '1KPu1RTEQQCA_GoNEapFtq8oZTPmyvMFx'
your_file_id_for_transactions_csv = '1FSrI_JqbzSmZ0c4NBmfiMT-kYQG31xUS'
your_file_id_for_oil_csv = '15uMxMXXSYEv14Yv4cLyKuiCEuO1TmXrU'
your_file_id_for_holidays_csv = '1fNj8v8kDmfdslU_M9FgmmV_FPqd1tIN1'
your_file_id_for_train_csv = '1KsbtNzrfL5ZTFLpnaCyEUAmRzA2EBDLz'

# Google Drive download links for each file
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}"
}

# Define BASE_DIR as the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define MODEL_PATH to point to the model directory
MODEL_PATH = os.path.join(BASE_DIR, "model")
