# utils/device.py
"""
Centralized device management.
Prevents: RuntimeError: Expected all tensors to be on the same device
"""
import torch
import os


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(obj, device: torch.device):
    """
    Recursively move tensors/dicts/lists to device.
    Safe for nested structures returned by dataloaders.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def configure_memory(max_split_size_mb: int = 512):
    """
    Configure PyTorch memory allocator to reduce fragmentation.
    Call once at startup.
    Prevents: CUDA out of memory (fragmentation variant).
    """
    if torch.cuda.is_available():
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            f"max_split_size_mb:{max_split_size_mb}"
        )


def log_gpu_memory(tag: str = ""):
    """Print current GPU memory stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"[GPU Memory {tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
