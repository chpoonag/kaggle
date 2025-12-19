import os
import kagglehub
from google.colab import drive

def download_kaggle_competition_data(api_key_dir: str, competition_name: str, is_api_key_in_google_drive: bool, dst_dir: str = None) -> str:
    if is_api_key_in_google_drive:
        # Mount Google Drive to access the API key
        drive.mount('/content/drive')

    # Setup Kaggle API credentials
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    os.system(f"cp {api_key_dir} ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")
    
    # Download the competition dataset using kagglehub
    data_path = kagglehub.competition_download(competition_name)
    print(f'Data source import complete; path: \n{data_path}')
    
    # Prepare input directory structure for Kaggle competition data
    if dst_dir is not None:
        new_data_path = os.path.join(dst_dir, competition_name)
        os.makedirs(new_data_path, exist_ok=True)
        os.system(f"mv {data_path} {dst_dir}")
        print(f'New data path: \n{new_data_path}')
    return new_data_path
