from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_image(image_path):
    """
    Load an image without applying any processing.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        Image: The loaded image object, or None if an error occurred.
    """
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_images_parallel(image_paths, num_processes=4):
    """
    Load images in parallel using multi-processing.

    Parameters:
        image_paths (list): List of image file paths.
        num_processes (int): Number of processes to use.

    Returns:
        dict: A dictionary where keys are image filenames and values are the loaded images.
    """
    loaded_images = {}
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit tasks to the executor
        future_to_image = {executor.submit(load_image, path): os.path.basename(path) for path in image_paths}
        
        # Use tqdm to show progress
        for future in tqdm(as_completed(future_to_image), total=len(image_paths), desc="Loading images"):
            image_name = future_to_image[future]
            try:
                loaded_images[image_name] = future.result()
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
    
    # Sort the dictionary by keys (image filenames)
    sorted_images = dict(sorted(loaded_images.items()))
    return sorted_images
