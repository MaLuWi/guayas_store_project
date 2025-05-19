# app/config.py

DATA_PATH = "data/"

# Google Drive file IDs for each dataset
your_file_id_for_stores_csv = '1OUyap9CGF59ewdkN3DM6NWw3nv-hT5pH'
your_file_id_for_items_csv = '1KPu1RTEQQCA_GoNEapFtq8oZTPmyvMFx'
your_file_id_for_train_csv = '14cGjnrhqQaBZV9BZmJpO0qGftMENKOAM'

# Google Drive download links for each file (use these for gdown or programmatic download!)
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}"
}

MODEL_PATH = "model/"
