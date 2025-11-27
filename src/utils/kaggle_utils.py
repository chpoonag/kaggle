import os
import ast
import json

def setup_kaggle(kaggle_secrets_name=None, credential_path=None):
    """
    Set up Kaggle API authentication using either Kaggle Secrets (Databricks environment)
    or a provided Kaggle credential JSON file path.

    Parameters:
    - kaggle_secrets_name: str or None, the name of the Kaggle secret (Databricks).
    - credential_path: str or None, local path to kaggle.json credential file.

    Returns:
    - authenticated KaggleApi instance if successful, else None.
    """

    assert not (kaggle_secrets_name is None) or not (credential_path is None), \
        "Need to provide at least one of these: kaggle_secrets_name, or credential_path."

    if kaggle_secrets_name is not None:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        data_str = user_secrets.get_secret(kaggle_secrets_name)
        data = ast.literal_eval(data_str)
        os.makedirs("/root/.config/kaggle", exist_ok=True)
        with open("/root/.config/kaggle/kaggle.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
        os.system("chmod 600 /root/.config/kaggle/kaggle.json")
    else:
        # One-time setup for Kaggle API with provided credential file
        os.makedirs("/root/.config/kaggle", exist_ok=True)
        os.system(f"cp {credential_path} /root/.config/kaggle/")
        os.system("chmod 600 /root/.config/kaggle/kaggle.json")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authenticated successfully!")
        return api
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        return None
