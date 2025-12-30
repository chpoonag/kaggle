import os
import sys
import subprocess
import psutil
import torch
# import nvidia_smi
from typing import Literal
# import socket

def has_internet(timeout: float = 3.0) -> bool:
    """
    Check if internet connection is available.
    
    Pings Google DNS (8.8.8.8:53) with configurable timeout.
    
    Args:
        timeout (float): Connection timeout in seconds. Defaults to 3.0.
        
    Returns:
        bool: True if internet available, False otherwise.
        
    Example:
        >>> has_internet()                    # 3s timeout
        >>> has_internet(timeout=1.0)         # 1s timeout
        >>> if has_internet(2.0): print("ON")
    """
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout).close()
        return True
    except:
        return False

def run_bash_cmd(cmd):
    """Execute a bash command and return the result."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result

def get_notebook_env()-> Literal['colab', 'kaggle', 'jupyter', 'other']:
    """
    Detect notebook environment: Colab, Kaggle, Jupyter, or other.
    Returns: 'colab' | 'kaggle' | 'jupyter' | 'other'
    """
    # Kaggle [web:654 adapted]
    if os.path.exists('/kaggle') and 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
        
    # Google Colab [web:654]
    if 'google.colab' in sys.modules and os.getenv('COLAB_RELEASE_TAG'):
        return 'colab'
    
    # Jupyter/JupyterLab [web:660]
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return 'jupyter'
    except:
        pass
    
    return 'other'

def enable_cpu_affinity(cpu_ids: list):
    pid = os.getpid()
    psutil.Process(pid).cpu_affinity(cpu_ids)

def get_system_usage(device: int = 0):
    import nvidia_smi
    # CPU
    cpu_percentage = psutil.cpu_percent()
    # RAM
    ram_percentage = psutil.virtual_memory().percent
    
    ret = {"cpu_percentage": cpu_percentage, "ram_percentage": ram_percentage}
    if torch.cuda.is_available():
        gpu_memory_alloc = torch.cuda.memory_allocated(device) / (1024**3)  # usage in GB
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # usage in GB
        ret[f'gpu_cuda_{device}_memory_alloc'] = gpu_memory_alloc
        ret[f'gpu_cuda_{device}_memory_reserved'] = gpu_memory_reserved
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)  # zero is the device index. set to 0 at the moment.
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        nvidia_smi.nvmlShutdown()
        ret[f'gpu_nvidia_smi_{device}_free'] = info.free
        ret[f'gpu_nvidia_smi_{device}_used'] = info.used

    return ret





