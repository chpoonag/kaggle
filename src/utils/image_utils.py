from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import bm3d
import cv2
from PIL import ImageFilter

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

class Denoiser:
    def denoise_bm3d(img, sigma=25):
        """
        Apply BM3D denoising.
    
        Parameters:
            img (PIL.Image): The input image.
            sigma (float): Noise standard deviation.
    
        Returns:
            PIL.Image: The denoised image.
        """
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        
        # Apply BM3D denoising
        denoised_array = bm3d.bm3d(img_array, sigma)
        
        # Convert back to PIL image
        return Image.fromarray(denoised_array.astype(np.uint8))
        
    def denoise_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
        """
        Apply Bilateral Filter for edge-preserving denoising.
    
        Parameters:
            img (PIL.Image): The input image.
            d (int): Diameter of the pixel neighborhood.
            sigma_color (float): Filter sigma in the color space.
            sigma_space (float): Filter sigma in the coordinate space.
    
        Returns:
            PIL.Image: The denoised image.
        """
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        
        # Apply Bilateral Filter
        denoised_array = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
        
        # Convert back to PIL image
        return Image.fromarray(denoised_array)
    
    def denoise_nlm(img, h=10, template_window_size=7, search_window_size=21):
        """
        Apply Non-Local Means (NLM) denoising.
    
        Parameters:
            img (PIL.Image): The input image.
            h (float): Strength of denoising.
            template_window_size (int): Size of the template patch.
            search_window_size (int): Size of the search window.
    
        Returns:
            PIL.Image: The denoised image.
        """
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        
        # Apply NLM denoising
        denoised_array = cv2.fastNlMeansDenoising(img_array, None, h, template_window_size, search_window_size)
        
        # Convert back to PIL image
        return Image.fromarray(denoised_array)

    def denoise_gaussian_blur(img, radius=4):
        """
        Apply Gaussian blur to denoise the image.
    
        Parameters:
            img (PIL.Image): The input image.
            radius (int): Radius of the Gaussian blur kernel.
    
        Returns:
            PIL.Image: The denoised image.
        """
        return img.filter(ImageFilter.GaussianBlur(radius))