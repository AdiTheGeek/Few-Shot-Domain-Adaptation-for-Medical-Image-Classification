#!/usr/bin/env bash
# Complete training script for few-shot domain adaptation experiments
# Usage: bash scripts/run_train.sh

set -e  # Exit on error

echo "=================================="
echo "Few-Shot Domain Adaptation Training"
echo "=================================="

# Install dependencies
echo "Installing dependencies..."
python -m pip install -q -r requirements.txt

# Verify installation
python -c "from configs.config import Config; print('✓ Config loaded')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'✓ Lightning {pl.__version__}')"

# Check GPU
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=================================="
echo "Running experiments..."
echo "=================================="

# Example 1: Baseline ViT training
echo ""
echo "1. Baseline ViT on source domain..."
python run_training.py \
    --train_csv data/chexpert_train.csv \
    --val_csv data/chexpert_val.csv \
    --backbone vit_base_patch16_224 \
    --epochs 30 \
    --batch_size 32 \
    --experiment_name baseline_vit_source

# Example 2: LoRA adaptation
echo ""
echo "2. LoRA adaptation (50-shot)..."
python run_training.py \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --epochs 30 \
    --experiment_name lora_adaptation_k50

# Example 3: Adapter adaptation
echo ""
echo "3. Adapter adaptation (50-shot)..."
python run_training.py \
    --use_adapter \
    --adapter_dim 64 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --epochs 30 \
    --experiment_name adapter_adaptation_k50

# Example 4: Prompt tuning
echo ""
echo "4. Prompt tuning (50-shot)..."
python run_training.py \
    --use_prompt \
    --prompt_tokens 10 \
    --target_csv data/nih_train.csv \
    --val_csv data/nih_val.csv \
    --few_shot_k 50 \
    --epochs 30 \
    --experiment_name prompt_tuning_k50

echo ""
echo "=================================="
echo "✅ All experiments complete!"
echo "=================================="
echo "Checkpoints saved in: ./checkpoints/"
echo "View results with: ls -lh checkpoints/*/"
