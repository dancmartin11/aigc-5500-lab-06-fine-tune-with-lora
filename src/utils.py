
'''Reproducibility and environment utilities for Lab 06.'''

import platform
import random
import torch


def set_seed(seed: int) -> None:
    '''Set random seeds for reproducible experiments.'''
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hardware_info() -> dict:
    '''Return the execution environment and hardware information.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return {
        'Python': platform.python_version(),
        'PyTorch': torch.__version__,
        'CUDA available': torch.cuda.is_available(),
        'CUDA version': torch.version.cuda,
        'Device': device,
        'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }
