import os
import pickle

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