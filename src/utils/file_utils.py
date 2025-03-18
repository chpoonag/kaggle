import os
import pickle

def get_folder_size(folder_path):
    """
    Calculate the total size of a folder (including all files and subdirectories).

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Total size of the folder in bytes.
    """
    total_size = 0

    # Walk through all files and subdirectories in the folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # Get the full path of the file
            file_path = os.path.join(dirpath, filename)

            # Add the file size to the total size
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                # Skip files that no longer exist (e.g., broken symbolic links)
                continue

    return total_size

def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not already exist.

    Parameters:
    directory (str): The path of the directory to create.

    Returns:
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")

def save_object(obj, file_path):
    """
    Save a Python object to a file using pickle serialization.
    
    Parameters:
    obj (any): The Python object to save.
    file_path (str): The file path to save the object to.

    Returns:
    None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
        
def load_object(file_path):
    """
    Load a Python object from a file using pickle deserialization.
    
    Parameters:
    file_path (str): The file path from which to load the object.
    
    Returns:
    obj (any): The loaded Python object.
    """
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj