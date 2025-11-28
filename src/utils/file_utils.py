import os
import pickle

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
