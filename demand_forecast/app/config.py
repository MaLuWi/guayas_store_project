import os

# Path where files will be downloaded in the Streamlit Cloud environment
DATA_PATH = '/tmp/data/'

# Google Drive file IDs
GOOGLE_DRIVE_LINKS = {
    "stores": "https://drive.google.com/uc?id=1OUyap9CGF59ewdkN3DM6NWw3nv-hT5pH",
    "items": "https://drive.google.com/uc?id=1KPu1RTEQQCA_GoNEapFtq8oZTPmyvMFx",
    "transactions": "https://drive.google.com/uc?id=1FSrI_JqbzSmZ0c4NBmfiMT-kYQG31xUS",
    "oil": "https://drive.google.com/uc?id=15uMxMXXSYEv14Yv4cLyKuiCEuO1TmXrU",
    "holidays_events": "https://drive.google.com/uc?id=1fNj8v8kDmfdslU_M9FgmmV_FPqd1tIN1",
    "train": "https://drive.google.com/uc?id=1KsbtNzrfL5ZTFLpnaCyEUAmRzA2EBDLz"
}

# Automatically resolve base project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to your model file (.xgb)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.xgb")  # <-- Replace with your actual filename
