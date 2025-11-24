# Complete Workflow Guide

## Overview

This guide walks through the complete experimental workflow for few-shot domain adaptation on medical imaging datasets.

---

## Phase 1: Environment Setup

### Local Setup
```bash
git clone <your-repo>
cd Few-Shot-Domain-Adaptation-for-Medical-Image-Classification
pip install -r requirements.txt
```

### Google Colab Setup
```python
# In Colab cell
!pip install -q torch torchvision timm transformers scikit-learn pandas Pillow matplotlib pytorch-lightning wandb opencv-python scipy

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

---

## Phase 2: Dataset Preparation

### Step 1: Download Datasets
- **CheXpert**: https://stanfordmlgroup.github.io/competitions/chexpert/
- **NIH ChestX-ray14**: https://nihcc.app.box.com/v/ChestXray-NIHCC

### Step 2: Prepare CSV Files
Each CSV must have:
- `Path` column: relative path to image
- 14 pathology columns with 0/1/-1 labels

Example CSV structure:
```
Path,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,...
patient001/study1/view1.jpg,0,0,1,...
patient002/study1/view1.jpg,1,0,0,...
```

### Step 3: Organize Directory
```
data/
├── chexpert_train.csv      # Source domain training
├── chexpert_val.csv        # Source domain validation
├── nih_train.csv           # Target domain training (full)
├── nih_val.csv             # Target domain validation
├── nih_test.csv            # Target domain test
└── images/                 # All image files
    └── ...
```

---

## Phase 3: Baseline Training (Source Domain)

Train a Vision Transformer on the source domain (CheXpert) to establish baseline performance.

### Command Line
```bash
python run_training.py \
    --train_csv data/chexpert_train.csv \
    --val_csv data/chexpert_val.csv \
    --backbone vit_base_patch16_224 \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --mixed_precision \
    --gradient_checkpointing \
    --experiment_name baseline_vit_source
```

### Python Script
```python
from configs.config import Config
from models.vit_backbone import ViTWrapper
from train.trainer import LitModel
from data.datasets import SimpleMedicalDataset, get_transforms
import pytorch_lightning as pl

config = Config(epochs=30, backbone='vit_base_patch16_224')
model = ViTWrapper(config.backbone, num_classes=14, pretrained=True)

train_ds = SimpleMedicalDataset('data/chexpert_train.csv', 'data', transform=get_transforms(224, True))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

lit = LitModel(model, config)
trainer = pl.Trainer(max_epochs=30, accelerator='gpu', devices=-1, precision=16)
trainer.fit(lit, train_loader)

# Save checkpoint
torch.save({'model_state_dict': model.state_dict()}, 'checkpoints/baseline_source.pth')
```

**Expected output:**
- Validation AUC: 0.75-0.78
- Training time: ~3 hours (single GPU)
- Checkpoint: `checkpoints/baseline_vit_source/best-*.pth`

---

## Phase 4: Few-Shot Sampling (Target Domain)

Sample limited labeled examples from the target domain (NIH).

### Using Built-in Function
```python
from data.datasets import SimpleMedicalDataset, sample_few_shot_indices
from torch.utils.data import Subset

# Load full target dataset
target_ds_full = SimpleMedicalDataset('data/nih_train.csv', 'data', transform=get_transforms(224, True))

# Sample k=50 examples per class
few_shot_indices = sample_few_shot_indices(target_ds_full, k_per_class=50, seed=42)
print(f"Selected {len(few_shot_indices)} samples")

# Create few-shot subset
target_ds_fewshot = Subset(target_ds_full, few_shot_indices)
```

### Experiment with Different K Values
```bash
# 10-shot
python run_training.py --few_shot_k 10 --target_csv data/nih_train.csv ...

# 25-shot
python run_training.py --few_shot_k 25 --target_csv data/nih_train.csv ...

# 50-shot
python run_training.py --few_shot_k 50 --target_csv data/nih_train.csv ...

# 100-shot
python run_training.py --few_shot_k 100 --target_csv data/nih_train.csv ...
```

---

## Phase 5: Adaptation Experiments

### Experiment 1: LoRA Adaptation

**Command:**
```bash
python run_training.py \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --epochs 30 \
    --experiment_name lora_r8_k50
```

**Python:**
```python
from lora.lora import apply_lora_to_model

# Load baseline checkpoint
model = ViTWrapper('vit_base_patch16_224', 14, pretrained=False)
checkpoint = torch.load('checkpoints/baseline_source.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Apply LoRA
apply_lora_to_model(model, r=8, alpha=32.0)

# Freeze original parameters
for name, param in model.named_parameters():
    if 'lora_' not in name:
        param.requires_grad = False

# Train on few-shot target data
# ... (see full code in notebook)
```

**Hyperparameter sweep:**
```bash
# Vary LoRA rank
for r in 4 8 16 32; do
    python run_training.py --use_lora --lora_r $r --lora_alpha 32 --experiment_name lora_r${r}
done
```

---

### Experiment 2: Adapter Adaptation

**Command:**
```bash
python run_training.py \
    --use_adapter \
    --adapter_dim 64 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --experiment_name adapter_dim64_k50
```

**Python:**
```python
from adapters.adapter import attach_adapter_to_vit

# Load and attach adapters
attach_adapter_to_vit(model.backbone, adapter_dim=64)

# Freeze backbone
for name, param in model.named_parameters():
    if 'adapter' not in name and 'classifier' not in name:
        param.requires_grad = False
```

**Hyperparameter sweep:**
```bash
# Vary adapter dimension
for dim in 32 64 128 256; do
    python run_training.py --use_adapter --adapter_dim $dim --experiment_name adapter_dim${dim}
done
```

---

### Experiment 3: Prompt Tuning

**Command:**
```bash
python run_training.py \
    --use_prompt \
    --prompt_tokens 10 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --experiment_name prompt_t10_k50
```

**Python:**
```python
from prompts.prompt_tuning import attach_visual_prompt_to_vit

# Attach prompts
attach_visual_prompt_to_vit(model.backbone, prompt_tokens=10)

# Freeze everything except prompts
for name, param in model.named_parameters():
    if 'visual_prompt' not in name and 'classifier' not in name:
        param.requires_grad = False
```

---

### Experiment 4: CNN Baselines

**ResNet-50:**
```bash
python run_training.py \
    --use_cnn \
    --backbone resnet50 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --experiment_name resnet50_k50
```

**DenseNet-121:**
```bash
python run_training.py \
    --use_cnn \
    --backbone densenet121 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --experiment_name densenet121_k50
```

---

## Phase 6: Evaluation

### Load Best Checkpoint and Evaluate

```python
from eval.evaluator import compute_metrics, bootstrap_confidence_interval

# Load model
model.load_state_dict(torch.load('checkpoints/lora_r8_k50/best-*.pth')['model_state_dict'])
model.eval()

# Run inference on test set
test_ds = SimpleMedicalDataset('data/nih_test.csv', 'data', transform=get_transforms(224, False))
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        imgs = batch['image'].cuda()
        labels = batch['labels'].numpy()
        logits = model(imgs)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Compute metrics
metrics = compute_metrics(all_preds, all_labels)
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Mean AP: {metrics['mean_ap']:.4f}")
print(f"Sensitivity: {metrics['mean_sens']:.4f}")
print(f"Specificity: {metrics['mean_spec']:.4f}")

# Bootstrap confidence intervals
ci = bootstrap_confidence_interval(
    all_preds, all_labels,
    lambda p, l: compute_metrics(p, l)['auc_roc'],
    n_bootstrap=1000
)
print(f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
```

---

## Phase 7: Results Comparison

### Collect All Results

```python
import pandas as pd

results = {
    'Method': ['Baseline ViT', 'LoRA r=8', 'Adapter dim=64', 'Prompt t=10', 'ResNet-50'],
    'Val AUC': [0.77, 0.80, 0.79, 0.78, 0.75],
    'Trainable Params': [86000000, 700000, 2000000, 90000, 23000000],
}

df = pd.DataFrame(results)
df['Param Efficiency (%)'] = 100.0 * df['Trainable Params'] / 86000000
print(df.to_string(index=False))
```

### Visualize

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Performance comparison
ax1.bar(df['Method'], df['Val AUC'])
ax1.set_ylabel('Validation AUC')
ax1.set_title('Performance Comparison')
ax1.tick_params(axis='x', rotation=45)

# Parameter efficiency
ax2.bar(df['Method'], df['Param Efficiency (%)'])
ax2.set_ylabel('Trainable Parameters (%)')
ax2.set_title('Parameter Efficiency')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=300)
```

---

## Phase 8: Ablation Studies

### 1. Sample Efficiency Curve

```bash
# Run experiments with different k values
for k in 10 25 50 100 200; do
    python run_training.py --use_lora --few_shot_k $k --experiment_name lora_k${k}
done

# Plot AUC vs. k
```

### 2. LoRA Rank Sensitivity

```bash
for r in 2 4 8 16 32; do
    python run_training.py --use_lora --lora_r $r --experiment_name lora_r${r}
done
```

### 3. Cross-Domain Comparison

```bash
# CheXpert → NIH
python run_training.py --use_lora --source_csv data/chexpert_train.csv --target_csv data/nih_train.csv

# NIH → CheXpert
python run_training.py --use_lora --source_csv data/nih_train.csv --target_csv data/chexpert_train.csv
```

---

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python run_training.py --batch_size 16 ...

# Enable gradient checkpointing (already default)
python run_training.py --gradient_checkpointing ...

# Use mixed precision (already default)
python run_training.py --mixed_precision ...
```

### Slow Training
```bash
# Use multiple GPUs
python run_training.py --gpus -1 ...

# Increase number of workers
python run_training.py --num_workers 8 ...
```

### Poor Performance
- Check data quality and preprocessing
- Verify CSV format and labels
- Try different learning rates: `--lr 5e-5` or `--lr 2e-4`
- Increase training epochs: `--epochs 50`
- Adjust adaptation hyperparameters (LoRA rank, adapter dim, etc.)

---

## Summary Checklist

- [ ] Environment setup complete
- [ ] Datasets downloaded and prepared
- [ ] Baseline model trained on source domain
- [ ] Few-shot sampling implemented
- [ ] LoRA adaptation tested
- [ ] Adapter adaptation tested
- [ ] Prompt tuning tested
- [ ] CNN baselines evaluated
- [ ] Results collected and compared
- [ ] Ablation studies conducted
- [ ] Final report generated

---

**For questions or issues, please open a GitHub issue or refer to the README.md.**
