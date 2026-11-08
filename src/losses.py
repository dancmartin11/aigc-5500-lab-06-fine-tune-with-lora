'''Custom loss masking utilities for Lab 06.'''

import torch

def answer_only_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    '''Return labels that score only the answer token in each example.'''

    # Initialize all positions as ignored
    labels = torch.full_like(input_ids, -100)

    # Identify the answer token position for each example
    batch_indices = torch.arange(input_ids.size(0))
    answer_positions = prompt_lens.to(input_ids.device)

    # Preserve only the target answer token
    labels[batch_indices, answer_positions] = input_ids[batch_indices, answer_positions]

    return labels
