import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """A simple bottleneck adapter: down-project -> non-linearity -> up-project.
    Inserted as a residual after MLP blocks for transformer layers.
    """
    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.dropout(z)
        z = self.up(z)
        return x + z


def attach_adapter_to_vit(vit_model: nn.Module, adapter_dim: int = 64):
    """Attach adapters after transformer MLP outputs in timm ViT models.
    This function tries to locate transformer blocks and append a `adapter` module.
    """
    for name, module in vit_model.named_modules():
        # look for blocks named 'blocks' containing an 'mlp' or 'mlp.fc2'
        if name.endswith('blocks') and hasattr(module, '__iter__'):
            # iterate children
            for idx, blk in enumerate(module):
                try:
                    mlp = getattr(blk, 'mlp', None)
                    if mlp is not None:
                        # attach adapter as attribute on block
                        setattr(blk, 'adapter', BottleneckAdapter(mlp.fc2.out_features, adapter_dim))
                except Exception:
                    continue
    return vit_model
