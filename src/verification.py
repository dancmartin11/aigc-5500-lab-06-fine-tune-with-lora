'''Verification utilities for custom LoRA implementations.'''

import torch.nn as nn

class ReferenceModel(nn.Module):
    '''Simple linear model for comparing custom LoRA against PEFT.'''

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 24)

    def forward(self, x):
        return self.linear(x)

def expected_lora_params(model, rank, targets = ('q_proj', 'v_proj')):
    '''Calculate expected LoRA parameters across target linear layers.'''

    total = 0
    matched_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split('.')[-1] in targets:
            total += rank * (module.in_features + module.out_features)
            matched_layers += 1

    return total, matched_layers
