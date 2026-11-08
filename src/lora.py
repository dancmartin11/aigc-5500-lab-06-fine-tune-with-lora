'''Custom Low-Rank Adaptation implementation for Lab 06.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    '''Wrap a frozen linear layer with a trainable low-rank update.'''

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()

        if r <= 0:
            raise ValueError('LoRA rank must be positive.')

        # Store the original linear layer and adaptation settings
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the pretrained weights and bias
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        # Initialize the trainable low-rank matrices
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

        # Initialize A with small random values
        nn.init.kaiming_uniform_(self.A, a = 5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply the frozen linear transformation and LoRA update.'''

        # Compute the original pretrained transformation
        base_output = self.base(x)

        # Compute the scaled low-rank adaptation
        lora_output = F.linear(F.linear(x, self.A), self.B)

        return base_output + self.scaling * lora_output

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        '''Return a standard linear layer with the LoRA update merged.'''

        # Create a new linear layer with matching dimensions and bias.
        merged_layer = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias = self.base.bias is not None,
            device = self.base.weight.device,
            dtype = self.base.weight.dtype
            )

        # Combine the frozen weights with the low-rank update.
        merged_weight = self.base.weight + self.scaling * (self.B @ self.A)

        # Copy the merged weights and original bias.
        merged_layer.weight.copy_(merged_weight)

        if self.base.bias is not None:
            merged_layer.bias.copy_(self.base.bias)

        return merged_layer
