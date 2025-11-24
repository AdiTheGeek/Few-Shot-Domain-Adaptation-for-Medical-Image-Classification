#!/usr/bin/env bash
# Evaluation script for trained models
# Usage: bash scripts/run_eval.sh <checkpoint_path> <test_csv>

set -e

CHECKPOINT=${1:-"checkpoints/lora_adaptation_k50/best-*.pth"}
TEST_CSV=${2:-"data/nih_test.csv"}

echo "=================================="
echo "Model Evaluation"
echo "=================================="
echo "Checkpoint: $CHECKPOINT"
echo "Test CSV: $TEST_CSV"
echo ""

# Run evaluation
python -c "
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.vit_backbone import ViTWrapper
from data.datasets import SimpleMedicalDataset, get_transforms
from eval.evaluator import compute_metrics, bootstrap_confidence_interval

print('Loading model...')
model = ViTWrapper('vit_base_patch16_224', 14, pretrained=False)
checkpoint = torch.load('$CHECKPOINT')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

print('Loading test data...')
test_ds = SimpleMedicalDataset('$TEST_CSV', './data', transform=get_transforms(224, False))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

print('Running inference...')
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

print('\\nComputing metrics...')
metrics = compute_metrics(all_preds, all_labels)

print('\\n' + '='*70)
print('TEST RESULTS')
print('='*70)
print(f'AUC-ROC: {metrics[\"auc_roc\"]:.4f}')
print(f'Mean AP: {metrics[\"mean_ap\"]:.4f}')
print(f'Sensitivity: {metrics[\"mean_sens\"]:.4f}')
print(f'Specificity: {metrics[\"mean_spec\"]:.4f}')

print('\\nComputing bootstrap confidence intervals...')
ci = bootstrap_confidence_interval(all_preds, all_labels, 
                                   lambda p, l: compute_metrics(p, l)['auc_roc'],
                                   n_bootstrap=1000)
print(f'AUC 95% CI: [{ci[\"ci_lower\"]:.4f}, {ci[\"ci_upper\"]:.4f}]')
print('='*70)
"

echo ""
echo "✅ Evaluation complete!"
