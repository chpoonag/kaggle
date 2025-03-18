import csv
import math
import random
import os
from datetime import datetime
import numpy as np
import torch

def load_tsv(tsv_file_path, start_row: int = 0, last_row: int = -1):
    """
    Load data from a TSV file.

    Args:
        tsv_file_path (str): Path to the TSV file.
        start_row (int): The starting row to read from.
        last_row (int): The last row to read up to. If -1, read all rows.

    Returns:
        list: A list of rows read from the TSV file.
    """
    last_row = math.inf if last_row == -1 else last_row
    data = []
    with open(tsv_file_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for i, row in enumerate(reader):
            if start_row <= i <= last_row:
                data.append(row)
    return data

def tensor_mask_by_value_range(inp_tensor, min_val, max_val, exclude_max_val=True):
    """
    Create a mask for a tensor based on a value range.

    Args:
        inp_tensor (torch.Tensor): Input tensor.
        min_val (float): Minimum value for the mask.
        max_val (float): Maximum value for the mask.
        exclude_max_val (bool): Whether to exclude the maximum value from the mask.

    Returns:
        torch.Tensor: A boolean mask tensor.
    """
    upper_bound = inp_tensor < max_val if exclude_max_val else inp_tensor <= max_val
    return (min_val <= inp_tensor) & upper_bound

def get_now_str(fmt='%Y.%m.%d_%H.%M.%S'):
    """
    Get the current time as a formatted string.

    Args:
        fmt (str): The format string for strftime.

    Returns:
        str: The current time formatted as a string.
    """
    now = datetime.now()
    now_str = now.strftime(fmt)
    return now_str

def seed_everything(seed: int):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed value.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_one_hot_masks(probabilities: list, num_samples: int, num_masks: int = 3, dtype: torch.dtype = torch.bool):
    """
    Create one-hot encoded masks based on specified probabilities.

    Args:
        probabilities (list): List of probabilities for each mask.
        num_samples (int): Number of samples to generate.
        num_masks (int): Number of different masks.
        dtype (torch.dtype): Data type of the output tensor.

    Returns:
        torch.Tensor: A tensor of one-hot encoded masks.
    """
    assert torch.tensor(probabilities).sum() == 1, "Sum of probabilities should be 1, but got {}.".format(torch.tensor(probabilities).sum())
    if len(probabilities) != num_masks:
        raise ValueError("Length of probabilities must match num_masks.")
    
    if not torch.isclose(torch.sum(torch.tensor(probabilities)), torch.tensor(1.0)):
        raise ValueError("Probabilities must sum to 1.")

    # Generate random integers based on the specified probabilities
    random_tensor = torch.multinomial(torch.tensor(probabilities), num_samples=num_samples, replacement=True)

    # One-hot encode the random tensor
    one_hot_tensor = torch.nn.functional.one_hot(random_tensor, num_classes=num_masks).to(dtype)

    return one_hot_tensor