'''Controlled fine-tuning experiments for Lab 06.'''

import json
import math
import torch
from pathlib import Path
from scripts import lab06_starter as S
from src.config import FULL_LR, LORA_LR, OUTPUT_DIR, TRAINING_STEPS, BATCH_SIZE
from src.losses import answer_only_labels
from src.utils import set_seed


def run_experiment(mode, seed, train_examples, test_examples, label_ids, rank = None, loss_type = 'answer_only', steps = TRAINING_STEPS, batch_size = BATCH_SIZE, save = True):
    '''Execute one controlled fine-tuning experiment.'''

    # Validate the experimental configuration
    if mode not in ('full', 'lora'):
        raise ValueError('Mode must be full or lora.')

    if loss_type not in ('answer_only', 'all_tokens'):
        raise ValueError('Unsupported loss type.')

    if mode == 'lora' and (rank is None or rank <= 0):
        raise ValueError('LoRA experiments require a positive rank.')

    # Configure the learning rate and loss function
    lr = FULL_LR if mode == 'full' else LORA_LR

    labels_fn = answer_only_labels if loss_type == 'answer_only' else S.all_tokens_labels

    # Initialize reproducibility for this training run
    set_seed(seed)

    # Execute the experiment using the provided training harness
    result = S.run(
        mode = mode,
        seed = seed,
        train_examples = train_examples,
        test = test_examples,
        label_ids = label_ids,
        r = rank if rank is not None else 8,
        steps = steps,
        lr = lr,
        labels_fn = labels_fn,
        batch_size = batch_size
        )

    # Include the experimental settings in the results
    result.update({
        'mode': mode,
        'rank': rank if mode == 'lora' else None,
        'seed': seed,
        'loss_type': loss_type,
        'learning_rate': lr,
        'steps': steps,
        'batch_size': batch_size
        })

    # Save results independently for each configuration
    if save:
        results_dir = OUTPUT_DIR / 'results'
        results_dir.mkdir(parents = True, exist_ok = True)

        rank_name = f'r{rank}' if mode == 'lora' else 'full'
        filename = f'{rank_name}_{loss_type}_seed{seed}.json'

        output_path = results_dir / filename

        with output_path.open('w', encoding = 'utf-8') as file:
            json.dump(result, file, indent = 4)

    return result

def run_sweep(experiment_plan, train_examples, test_examples, label_ids):
    '''Execute the experiment plan and reuse matching saved results.'''

    results = []
    results_dir = OUTPUT_DIR / 'results'

    for index, config in enumerate(experiment_plan, start = 1):
        mode = config['mode']
        rank = config['rank']
        seed = config['seed']
        loss_type = config['loss_type']

        # Identify the saved experiment results
        rank_name = f'r{rank}' if mode == 'lora' else 'full'
        filename = f'{rank_name}_{loss_type}_seed{seed}.json'
        result_path = results_dir / filename

        learning_rate = FULL_LR if mode == 'full' else LORA_LR

        # Reuse saved results only when the configuration matches
        if result_path.exists():
            with result_path.open('r', encoding = 'utf-8') as file:
                saved = json.load(file)

            valid = (
                saved['mode'] == mode
                and saved['rank'] == rank
                and saved['seed'] == seed
                and saved['loss_type'] == loss_type
                and math.isclose(saved['learning_rate'], learning_rate)
                and saved['steps'] == TRAINING_STEPS
                and saved['batch_size'] == BATCH_SIZE
            )

            if valid:
                print(f'[{index}/{len(experiment_plan)}] Reused: {filename}', flush = True)
                results.append(saved)
                continue

            raise ValueError(f'Configuration mismatch: {filename}')

        # Train a fresh model for the current configuration
        print(f'[{index}/{len(experiment_plan)}] Training: {filename}', flush = True)

        result = run_experiment(mode = mode, rank = rank, seed = seed, loss_type = loss_type, train_examples = train_examples, test_examples = test_examples, label_ids = label_ids)
        results.append(result)

        # Release unused CUDA allocations between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results