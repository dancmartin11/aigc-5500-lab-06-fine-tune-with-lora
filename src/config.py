'''Experiment configuration for Lab 06: Fine-Tune with LoRA.'''

from pathlib import Path
from scripts import lab06_starter as S

# Model and classification task
MODEL_NAME = S.MODEL_NAME
TASK_NAME = 'emotion'

# Reproducibility and dataset configuration
SEEDS = [0, 1, 2]
DATA_SEED = 0

N_TRAIN = 600
N_TEST = 500

# Fine-tuning configurations
LORA_RANKS = [1, 4, 16]
TRAINING_STEPS = 120
BATCH_SIZE = 8

# Learning rates for each fine-tuning method
FULL_LR = 5e-5
LORA_LR = 1e-3

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def create_output_directories():
    '''Create directories for experiment outputs and figures.'''
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)
