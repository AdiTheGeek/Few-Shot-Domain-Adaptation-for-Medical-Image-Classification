# ============================================================================
# config.py - Configuration Management
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class Config:
    # Dataset
    data_root: str = "./chexpert_data"
    img_size: int = 224
    num_classes: int = 14  # CheXpert has 14 pathologies
    batch_size: int = 32
    num_workers: int = 4
    
    # Model Selection
    backbone: str = "resnet50"  # Options: resnet18/34/50, densenet121, efficientnet_b0, unet
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.3
    
    # Domain Adaptation
    use_domain_adaptation: bool = False
    da_method: str = "none"  # Options: none, dann, coral, mmd, mcd, adda
    da_weight: float = 0.1  # Weight for DA loss
    source_domain: str = "frontal"  # For DA experiments
    target_domain: str = "lateral"
    
    # Training
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # Options: step, cosine, plateau
    early_stopping_patience: int = 10
    
    # Optimization
    optimizer: str = "adam"  # Options: adam, adamw, sgd
    label_smoothing: float = 0.0
    mixed_precision: bool = True
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Experiment tracking
    experiment_name: str = "baseline"
    seed: int = 42
    
    # Multi-label settings
    pos_weight: Optional[List[float]] = None  # For handling class imbalance
