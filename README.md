# Few-Shot Domain Adaptation for Medical Image Classification

**Complete production-ready pipeline for evaluating few-shot domain adaptation techniques on Vision Transformers for medical imaging.**

This repository implements multiple parameter-efficient adaptation strategies (LoRA, Adapters, Prompt Tuning) and compares them against baseline approaches for medical chest X-ray classification with limited target-domain samples.

---

## Key Features

- **Vision Transformer backbone** (timm) with feature extraction
- **Parameter-efficient adaptation**:
  - LoRA (Low-Rank Adaptation)
  - Adapter layers (bottleneck modules)
  - Visual Prompt Tuning
- **CNN baselines**: ResNet-50, DenseNet-121
- **Dataset loaders** for CheXpert and NIH ChestX-ray14
- **Few-shot sampling** utilities
- **PyTorch Lightning** training harness with multi-GPU support
- **Comprehensive evaluation**: AUROC, sensitivity, specificity, bootstrapped CIs
- **Colab Pro+ optimized**: mixed precision, gradient checkpointing, multi-GPU

---

## Project Structure

```
├── configs/
│   └── config.py                 # Configuration dataclass
├── data/
│   └── datasets.py              # Dataset loaders and few-shot sampling
├── models/
│   ├── vit_backbone.py          # Vision Transformer wrapper
│   └── cnn_backbones.py         # CNN baselines (ResNet, DenseNet)
├── lora/
│   └── lora.py                  # LoRA implementation
├── adapters/
│   └── adapter.py               # Adapter layers
├── prompts/
│   └── prompt_tuning.py         # Visual prompt tuning
├── train/
│   └── trainer.py               # PyTorch Lightning training module
├── eval/
│   └── evaluator.py             # Metrics and evaluation utilities
├── utils/
│   └── utils.py                 # Helper functions
├── scripts/
│   ├── run_train.sh             # Shell script for training
│   └── run_eval.sh              # Shell script for evaluation
├── notebooks/
│   └── colab_end_to_end.ipynb   # Complete Colab notebook
├── run_training.py              # Main training driver (CLI)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Prepare your datasets in CSV format with the following structure:

**CSV columns:**
- `Path`: Relative path to image file
- 14 pathology columns (e.g., `No Finding`, `Cardiomegaly`, etc.) with 0/1 labels

**Directory structure:**
```
data/
├── chexpert_train.csv
├── chexpert_val.csv
├── nih_train.csv
├── nih_val.csv
├── nih_test.csv
└── images/
    ├── patient001/
    │   └── study1/
    │       └── image.jpg
```

### 3. Run Training

#### Option A: Using the CLI Driver

**Baseline ViT on source domain (CheXpert):**
```bash
python run_training.py \
    --train_csv data/chexpert_train.csv \
    --val_csv data/chexpert_val.csv \
    --backbone vit_base_patch16_224 \
    --epochs 30 \
    --experiment_name baseline_vit_chexpert
```

**LoRA adaptation (CheXpert → NIH, 50-shot):**
```bash
python run_training.py \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --source_csv data/chexpert_train.csv \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --epochs 30 \
    --experiment_name lora_chexpert_to_nih
```

**Adapter adaptation:**
```bash
python run_training.py \
    --use_adapter \
    --adapter_dim 64 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --experiment_name adapter_adaptation
```

**Prompt tuning:**
```bash
python run_training.py \
    --use_prompt \
    --prompt_tokens 10 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --experiment_name prompt_tuning
```

**CNN baseline (ResNet-50):**
```bash
python run_training.py \
    --use_cnn \
    --backbone resnet50 \
    --train_csv data/chexpert_train.csv \
    --val_csv data/chexpert_val.csv \
    --experiment_name resnet50_baseline
```

#### Option B: Using Google Colab

1. Upload `notebooks/colab_end_to_end.ipynb` to Google Colab
2. Set runtime to GPU (preferably Colab Pro+ with multiple GPUs)
3. Run all cells sequentially

The notebook includes:
- Complete environment setup
- Baseline training on source domain
- LoRA, Adapter, and Prompt Tuning experiments
- CNN baselines
- Results comparison and visualization

---

## Evaluation

The training script automatically computes:
- **AUROC** (macro-averaged across 14 pathologies)
- **Mean Average Precision (mAP)**
- **Sensitivity** and **Specificity**
- **Parameter efficiency** (% of trainable parameters)

Best checkpoints are saved to `./checkpoints/{experiment_name}/`.

For bootstrapped confidence intervals:
```python
from eval.evaluator import bootstrap_confidence_interval, compute_metrics

ci_results = bootstrap_confidence_interval(
    predictions, labels, 
    lambda p, l: compute_metrics(p, l)['auc_roc'],
    n_bootstrap=1000
)
print(f"AUC: {ci_results['mean']:.4f} (95% CI: [{ci_results['ci_lower']:.4f}, {ci_results['ci_upper']:.4f}])")
```

---

## Configuration

Key hyperparameters in `configs/config.py`:

```python
@dataclass
class Config:
    # Model
    backbone: str = "vit_base_patch16_224"
    
    # Adaptation
    use_lora: bool = False
    lora_r: int = 8              # LoRA rank
    lora_alpha: float = 32.0     # LoRA scaling
    
    use_adapter: bool = False
    adapter_dim: int = 64        # Adapter bottleneck dimension
    
    use_prompt: bool = False
    prompt_tokens: int = 10      # Number of prompt tokens
    
    # Training
    epochs: int = 30
    lr: float = 1e-4
    batch_size: int = 32
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Few-shot
    few_shot_k: int = 50         # Samples per class
```

---

## Adaptation Strategies

### 1. **LoRA (Low-Rank Adaptation)**
- Adds trainable low-rank matrices to linear layers
- Freezes original model weights
- Typical efficiency: **<1% trainable parameters**

### 2. **Adapter Layers**
- Inserts lightweight bottleneck modules after transformer blocks
- Freezes backbone, trains only adapters
- Typical efficiency: **1-3% trainable parameters**

### 3. **Prompt Tuning**
- Prepends learnable prompt tokens to input embeddings
- Freezes entire model, trains only prompt embeddings
- Typical efficiency: **<0.1% trainable parameters**

### 4. **Baselines**
- **Full fine-tuning**: All parameters trainable (100%)
- **Partial fine-tuning**: Selective layer unfreezing
- **CNN baselines**: ResNet-50, DenseNet-121

---

## Expected Results

Based on typical few-shot domain adaptation performance (CheXpert → NIH, 50-shot):

| Method | Val AUC | Trainable Params | Training Time |
|--------|---------|------------------|---------------|
| Baseline ViT | 0.75-0.78 | 86M (100%) | ~3 hours |
| LoRA (r=8) | 0.78-0.81 | <1M (<1%) | ~45 min |
| Adapter (dim=64) | 0.77-0.80 | ~2M (2-3%) | ~1 hour |
| Prompt Tuning | 0.76-0.79 | <100K (<0.1%) | ~30 min |
| ResNet-50 | 0.73-0.76 | 23M (100%) | ~2 hours |

*Note: Results vary based on dataset splits, hyperparameters, and random seeds.*

---

## Multi-GPU Training (Colab Pro+)

The training script automatically detects and uses all available GPUs:

```bash
python run_training.py \
    --use_lora \
    --gpus -1 \  # Use all available GPUs
    --batch_size 64 \  # Increase batch size for multiple GPUs
    ...
```

PyTorch Lightning handles distributed training automatically with:
- **Data parallelism** across GPUs
- **Gradient synchronization**
- **Mixed precision (fp16)** for faster training
- **Gradient checkpointing** to reduce memory usage

---

## References

1. Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.
2. Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
3. Irvin et al. (2019). "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels." *AAAI*.
4. Wang et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database." *CVPR*.

---
