import os
import pickle, joblib
from typing import Literal

def folder_structure_to_list(path, max_depth=10, depth=0, list_folders_only=True):
    """
    Recursively builds a nested list representing the folder structure.

    Each element in the list is either a folder name (str) or a nested list
    where the first element is a folder name and the subsequent elements are
    its children.

    Args:
        path (str): The starting path of the directory.
        max_depth (int, optional): Maximum depth of recursion. Defaults to 10.
        depth (int, optional): Current recursion depth (internal use). Defaults to 0.
        list_folders_only (bool, optional): If True, include only folders;
                                            if False, include files as well.
                                            Defaults to True.

    Returns:
        list: A nested list representing the folder structure, with folder names
              and their nested children.
    """
    d = [os.path.basename(path),]
    if depth <= max_depth:
        if os.path.isdir(path):
            children = [
                folder_structure_to_list(os.path.join(path, x), max_depth, depth+1, list_folders_only) \
                for x in os.listdir(path) \
                if (list_folders_only and os.path.isdir(os.path.join(path, x))) or not(list_folders_only)            
            ]                
            d += children
    return d

def folder_structure_to_dict(path, max_depth=10, depth=0, list_folders_only=True):
    """
    Recursively builds a dictionary representing the folder structure.
    
    Structure: { "folder_name": { "subfolder": { ... } } }

    Args:
        path (str): The starting path of the directory.
        max_depth (int): Maximum recursion depth.
        depth (int): Current depth (internal use).
        list_folders_only (bool): If True, only lists directories.

    Returns:
        dict: A dictionary representing the folder structure.
    """
    # Get the base name of the current path
    name = os.path.basename(path)
    if not name:  # Handle cases where path ends with a slash
        name = os.path.basename(os.path.dirname(path))

    # Initialize the dictionary to hold children
    children = {}

    if depth <= max_depth and os.path.isdir(path):
        try:
            for x in os.listdir(path):
                child_path = os.path.join(path, x)
                
                # Check criteria (folder vs file)
                if list_folders_only and not os.path.isdir(child_path):
                    continue
                
                # Recursively call the function and update the children dictionary
                # The recursive call returns { "child_name": { ... } }
                children.update(
                    folder_structure_to_dict(child_path, max_depth, depth + 1, list_folders_only)
                )
        except PermissionError:
            # Handle cases where the script doesn't have access to the folder
            pass

    return {name: children}

def get_folder_size(
    folder_path: str,
    unit: Literal["b", "kb", "mb", "gb", "kib", "mib", "gib"] = "b",
) -> float:
    """
    Calculate the total size of a folder (including all files and subdirectories).

    Args:
        folder_path (str): Path to the folder.
        unit (str): One of "b", "kb", "mb", "gb", "kib", "mib", "gib". 
                   kb/mb/gb use 1000 base, kib/mib/gib use 1024 base. Defaults to "b" (bytes).

    Returns:
        float: Total size of the folder in the requested unit.
    """
    unit = unit.lower()
    factor_map = {
        "b": 1,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
    }
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                # Skip files that no longer exist (e.g., broken symbolic links)
                continue

    if unit not in factor_map:
        raise ValueError(f"Unsupported unit: {unit!r}. Use 'b', 'kb', 'mb', 'gb', 'kib', 'mib', or 'gib'.")

    return total_size / factor_map[unit]

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
    Save a Python object to a file using appropriate serialization method based on extension.
    
    Automatically detects format from file extension (.pkl, .pickle, .joblib) and uses
    pickle for standard serialization or joblib for large numpy/scipy objects.
    
    Args:
        obj: The Python object to save (any picklable/joblib-compatible type)
        file_path (str): Path to save file. Extension determines serializer:
                        - .pkl, .pickle → pickle.dump()
                        - .joblib → joblib.dump()
    
    Raises:
        ValueError: If file extension is not 'pkl', 'pickle', or 'joblib'
        pickle.PickleError: If object cannot be serialized
        OSError: If file cannot be written
    
    Returns:
        None
    """
    ext = file_path.split(".")[-1]
    with open(file_path, 'wb') as file:
        if ext in ['pkl', 'pickle']:
            pickle.dump(obj, file)
        elif ext in ['joblib']:
            joblib.dump(obj, file)
        else:
            raise ValueError(f"Unexpected file extension '{ext}'. Use .pkl, .pickle, or .joblib.")

def load_object(file_path):
    """
    Load a Python object from a file using appropriate deserialization method.
    
    Automatically detects format from file extension (.pkl, .pickle, .joblib) and uses
    pickle for standard deserialization or joblib for large numpy/scipy objects.
    
    Args:
        file_path (str): Path to load file. Extension determines deserializer:
                        - .pkl, .pickle → pickle.load()
                        - .joblib → joblib.load()
    
    Raises:
        ValueError: If file extension is not 'pkl', 'pickle', or 'joblib'
        pickle.UnpicklingError: If file is corrupted/invalid pickle
        FileNotFoundError: If file_path does not exist
        OSError: If file cannot be read
    
    Returns:
        The deserialized Python object (original type preserved)
    """
    ext = file_path.split(".")[-1]
    with open(file_path, 'rb') as file:
        if ext in ['pkl', 'pickle']:
            obj = pickle.load(file)
        elif ext in ['joblib']:
            obj = joblib.load(file)
        else:
            raise ValueError(f"Unexpected file extension '{ext}'. Use .pkl, .pickle, or .joblib.")
    return obj




