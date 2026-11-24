"""Download the Kaggle imagesoasis dataset and print its local cache path."""

import kagglehub

path = kagglehub.dataset_download("ninadaithal/imagesoasis")
print("Downloaded to:", path)
