# Project File Summary

Complete file-by-file breakdown of the Few-Shot Domain Adaptation codebase.

---

## 📁 Core Configuration

### `configs/config.py`
**Purpose:** Central configuration dataclass for all hyperparameters
- Model settings (backbone, pretrained weights)
- Adaptation hyperparameters (LoRA rank, adapter dim, prompt tokens)
- Training settings (epochs, learning rate, batch size)
- Few-shot configuration (k samples per class)
- Hardware settings (mixed precision, gradient checkpointing)

**Key classes:**
- `Config`: Main configuration dataclass

---

## 📁 Data Pipeline

### `data/datasets.py`
**Purpose:** Dataset loading and preprocessing
- `SimpleMedicalDataset`: CSV-based medical image dataset loader
- `get_transforms()`: Image augmentation pipelines
- `make_dataloaders()`: DataLoader factory
- `sample_few_shot_indices()`: K-shot sampling utility

**Features:**
- Handles CheXpert and NIH ChestX-ray14 formats
- Uncertainty label handling (-1 → 0/1)
- Multi-label support (14 pathologies)
- Domain-specific augmentations

---

## 📁 Model Architectures

### `models/vit_backbone.py`
**Purpose:** Vision Transformer wrapper
- `ViTWrapper`: Unified ViT interface using timm
- Feature extraction capability
- Custom classifier head
- Gradient checkpointing support

**Supported models:**
- `vit_base_patch16_224`
- `vit_large_patch16_224`
- Other timm ViT variants

### `models/cnn_backbones.py`
**Purpose:** CNN baseline models
- `build_cnn()`: Factory for CNN models via timm
- ResNet-18/34/50/101
- DenseNet-121/169
- EfficientNet-B0/B1

---

## 📁 Adaptation Modules

### `lora/lora.py`
**Purpose:** Low-Rank Adaptation (LoRA) implementation
- `LoRALinear`: LoRA wrapper for nn.Linear
- `apply_lora_to_model()`: Inject LoRA into model
- Trainable parameters: <1% of original model

**Hyperparameters:**
- `r`: Rank (typical: 4-16)
- `alpha`: Scaling factor (typical: 16-32)

### `adapters/adapter.py`
**Purpose:** Adapter layer implementation
- `BottleneckAdapter`: Down-project → ReLU → Up-project
- `attach_adapter_to_vit()`: Insert adapters into ViT blocks
- Trainable parameters: 1-3% of original model

**Hyperparameters:**
- `adapter_dim`: Bottleneck dimension (typical: 32-128)

### `prompts/prompt_tuning.py`
**Purpose:** Visual prompt tuning
- `VisualPrompt`: Learnable prompt embeddings
- `attach_visual_prompt_to_vit()`: Prepend prompts to patches
- Trainable parameters: <0.1% of original model

**Hyperparameters:**
- `prompt_tokens`: Number of prompts (typical: 5-20)

---

## 📁 Training Infrastructure

### `train/trainer.py`
**Purpose:** PyTorch Lightning training module
- `LitModel`: Lightning wrapper for any model
- Multi-GPU support (DDP)
- Mixed precision training (fp16)
- Automatic metric logging
- Test evaluation hooks

**Features:**
- Validation metrics: AUC, AP, sensitivity, specificity
- Distributed training with gradient sync
- Learning rate scheduling (cosine annealing)

### `run_training.py`
**Purpose:** Main CLI training driver
- Argument parsing for all experiments
- Model building and adaptation application
- Checkpoint saving
- Multi-GPU configuration
- Experiment tracking

**Usage examples:**
```bash
# Baseline
python run_training.py --train_csv data/train.csv --val_csv data/val.csv

# LoRA
python run_training.py --use_lora --lora_r 8 --target_csv data/target.csv

# Adapter
python run_training.py --use_adapter --adapter_dim 64 --target_csv data/target.csv

# Prompt
python run_training.py --use_prompt --prompt_tokens 10 --target_csv data/target.csv
```

---

## 📁 Evaluation

### `eval/evaluator.py`
**Purpose:** Comprehensive evaluation metrics
- `compute_auc()`: Multi-label AUROC
- `compute_sensitivity_specificity()`: Per-class performance
- `compute_metrics()`: Aggregate metrics (AUC, AP, sensitivity, specificity)
- `bootstrap_confidence_interval()`: Statistical confidence intervals

**Metrics:**
- AUC-ROC (macro-averaged)
- Mean Average Precision (mAP)
- Sensitivity (recall)
- Specificity
- Bootstrapped 95% CIs

---

## 📁 Utilities

### `utils/utils.py`
**Purpose:** Helper functions
- `count_parameters()`: Total and trainable parameter counts
- `set_seed()`: Reproducibility via random seeds

---

## 📁 Scripts

### `scripts/run_train.sh`
**Purpose:** Bash script for running multiple experiments
- Automated experiment pipeline
- Runs baseline, LoRA, adapter, prompt tuning
- Unix/Linux compatible

### `scripts/run_eval.sh`
**Purpose:** Bash script for model evaluation
- Load checkpoint
- Run inference on test set
- Compute metrics with CIs

---

## 📁 Notebooks

### `notebooks/colab_end_to_end.ipynb`
**Purpose:** Complete Colab notebook with all experiments
- Environment setup
- Dataset preparation
- Baseline training
- All adaptation methods
- CNN baselines
- Results comparison
- Visualization

**Sections:**
1. Setup and installation
2. Dataset loading
3. Baseline ViT training
4. LoRA adaptation
5. Adapter adaptation
6. Prompt tuning
7. CNN baselines
8. Results visualization
9. Test evaluation with CIs

---

## 📁 Documentation

### `README.md`
**Purpose:** Main project documentation
- Quick start guide
- Installation instructions
- Usage examples
- API reference
- Expected results
- Citation

### `WORKFLOW.md`
**Purpose:** Detailed experimental workflow
- Phase-by-phase guide
- Code examples for each step
- Hyperparameter tuning tips
- Ablation study templates
- Troubleshooting

### `example_quick_start.py`
**Purpose:** Minimal working example
- End-to-end pipeline in one file
- Quick testing script
- Demonstrates all components
- Fallback to dummy data if datasets missing

---

## 📁 Configuration Files

### `requirements.txt`
**Purpose:** Python dependencies
- PyTorch and torchvision
- timm (model library)
- transformers (HuggingFace)
- PyTorch Lightning
- scikit-learn (metrics)
- wandb (logging)
- Others: pandas, Pillow, matplotlib, opencv

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 🔄 Typical Workflow

1. **Prepare data** → `data/datasets.py`
2. **Configure experiment** → `configs/config.py` or CLI args
3. **Build model** → `models/vit_backbone.py` or `models/cnn_backbones.py`
4. **Apply adaptation** → `lora/`, `adapters/`, or `prompts/`
5. **Train model** → `run_training.py` or `train/trainer.py`
6. **Evaluate** → `eval/evaluator.py`
7. **Analyze results** → Notebooks or custom scripts

---

## 📊 File Dependency Graph

```
run_training.py
├── configs/config.py
├── data/datasets.py
├── models/
│   ├── vit_backbone.py
│   └── cnn_backbones.py
├── lora/lora.py
├── adapters/adapter.py
├── prompts/prompt_tuning.py
├── train/trainer.py
│   └── eval/evaluator.py
└── utils/utils.py
```

---

## 🎯 Key Entry Points

For different use cases:

1. **Run full experiment suite:**
   - Use: `notebooks/colab_end_to_end.ipynb`
   - Platform: Google Colab Pro+

2. **Run single experiment:**
   - Use: `run_training.py`
   - Platform: Any GPU server

3. **Quick test:**
   - Use: `example_quick_start.py`
   - Platform: Local machine

4. **Custom experiment:**
   - Import modules directly
   - Build custom pipeline

---

## 📝 File Size Estimates

| File | Lines | Purpose |
|------|-------|---------|
| `run_training.py` | ~250 | Main driver |
| `data/datasets.py` | ~130 | Data pipeline |
| `models/vit_backbone.py` | ~40 | ViT wrapper |
| `lora/lora.py` | ~60 | LoRA |
| `adapters/adapter.py` | ~45 | Adapters |
| `prompts/prompt_tuning.py` | ~55 | Prompts |
| `train/trainer.py` | ~80 | Lightning module |
| `eval/evaluator.py` | ~70 | Metrics |
| `notebooks/colab_end_to_end.ipynb` | ~500 | Full notebook |

**Total codebase:** ~1,500 lines of production-ready Python

---

## 🚀 Getting Started Recommendations

1. **First time users:**
   - Start with `example_quick_start.py`
   - Read `README.md`
   - Try Colab notebook

2. **Experienced ML practitioners:**
   - Go straight to `run_training.py`
   - Modify `configs/config.py` as needed
   - Reference `WORKFLOW.md` for advanced usage

3. **Researchers:**
   - Study implementation files in `lora/`, `adapters/`, `prompts/`
   - Extend base classes for new adaptation methods
   - Use `eval/evaluator.py` for fair comparisons

---

**Last updated:** 2025-11-24
