import math
import torch
import torch.nn as nn
from typing import List


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: W x + alpha / r * (A (B x)).
    This keeps the original weight and adds a low-rank trainable update.
    """
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.orig = orig
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, orig.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((orig.out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = self.alpha / max(1, self.r)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = self.orig(x)
        if self.r > 0:
            lora_update = (x @ self.lora_A.t()) @ self.lora_B.t()
            out = out + self.scaling * lora_update
        return out


def apply_lora_to_model(model: nn.Module, r: int = 4, alpha: float = 16.0, target_modules: List[str] = None):
    import types, math
    if target_modules is None:
        target_modules = ['q', 'k', 'v', 'proj', 'fc', 'mlp', 'head']

    for name, module in list(model.named_modules()):
        # Replace nn.Linear instances
        if isinstance(module, nn.Linear):
            parent = _get_parent_module(model, name)
            attr = name.split('.')[-1]
            setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha))


def _get_parent_module(root: nn.Module, dotted_name: str):
    parts = dotted_name.split('.')
    if len(parts) == 1:
        return root
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent
