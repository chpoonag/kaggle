import os
import psutil
import torch
import nvidia_smi

def enable_cpu_affinity(cpu_ids: list):
    pid = os.getpid()
    psutil.Process(pid).cpu_affinity(cpu_ids)

def get_system_usage(device: int = 0):
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