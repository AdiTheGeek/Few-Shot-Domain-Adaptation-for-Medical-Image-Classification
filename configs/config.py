from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    # Data
    data_root: str = "./data"
    img_size: int = 224
    num_classes: int = 14
    batch_size: int = 32
    num_workers: int = 4

    # Model
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    vit_variant: str = "timm"  # or 'hf'

    # Adaptation
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 32.0
    use_adapter: bool = False
    adapter_dim: int = 64
    use_prompt: bool = False
    prompt_tokens: int = 10

    # Training
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    # Few-shot
    few_shot_k: int = 50

    # Logging/checkpoint
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    use_wandb: bool = False
    project_name: str = "fewshot-da-medical"

    # Misc
    seed: int = 42
