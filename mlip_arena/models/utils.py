"""Utility functions for MLIP models."""

import torch


def get_freer_device() -> torch.device:
    """Get the GPU with the most free memory, or use MPS if available.
    s
        Returns:
            torch.device: The selected GPU device or MPS.

        Raises:
            ValueError: If no GPU or MPS is available.
    """
    device_count = torch.cuda.device_count()
    if device_count > 0:
        # If CUDA GPUs are available, select the one with the most free memory
        mem_free = [
            torch.cuda.get_device_properties(i).total_memory
            - torch.cuda.memory_allocated(i)
            for i in range(device_count)
        ]
        free_gpu_index = mem_free.index(max(mem_free))
        device = torch.device(f"cuda:{free_gpu_index}")
        print(
            f"Selected GPU {device} with {mem_free[free_gpu_index] / 1024**2:.2f} MB free memory from {device_count} GPUs"
        )
    elif torch.backends.mps.is_available():
        # If no CUDA GPUs are available but MPS is, use MPS
        print("No GPU available. Using MPS.")
        device = torch.device("mps")
    else:
        # Fallback to CPU if neither CUDA GPUs nor MPS are available
        print("No GPU or MPS available. Using CPU.")
        device = torch.device("cpu")

    return device
