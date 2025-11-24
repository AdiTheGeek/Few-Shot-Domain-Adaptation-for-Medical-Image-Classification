"""
example_quick_start.py - Minimal example for quick testing

This script demonstrates a minimal working example of the entire pipeline.
Use this to verify your setup before running full experiments.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import project modules
from configs.config import Config
from data.datasets import SimpleMedicalDataset, get_transforms, sample_few_shot_indices
from models.vit_backbone import ViTWrapper
from lora.lora import apply_lora_to_model
from train.trainer import LitModel
from eval.evaluator import compute_metrics
from utils.utils import set_seed, count_parameters

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    print("="*70)
    print("QUICK START EXAMPLE - Few-Shot Domain Adaptation")
    print("="*70)
    
    # Configuration
    config = Config(
        data_root='./data',
        batch_size=8,  # Small batch for testing
        epochs=3,  # Few epochs for quick test
        lr=1e-4,
        mixed_precision=True,
        gradient_checkpointing=True,
        few_shot_k=10,  # Small k for quick test
        seed=42
    )
    
    set_seed(config.seed)
    print(f"\n✓ Configuration loaded (epochs={config.epochs}, k={config.few_shot_k})")
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load datasets (using dummy CSVs for this example)
    # In practice, replace with your actual CSV paths
    try:
        train_ds = SimpleMedicalDataset(
            'data/train.csv', 
            config.data_root, 
            transform=get_transforms(config.img_size, True)
        )
        val_ds = SimpleMedicalDataset(
            'data/val.csv',
            config.data_root,
            transform=get_transforms(config.img_size, False)
        )
        print(f"\n✓ Datasets loaded")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val: {len(val_ds)} samples")
        
        # Few-shot sampling
        few_shot_indices = sample_few_shot_indices(train_ds, k_per_class=config.few_shot_k, seed=config.seed)
        train_subset = Subset(train_ds, few_shot_indices)
        print(f"  Few-shot subset: {len(train_subset)} samples")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Dataset files not found. Please prepare your CSV files.")
        print(f"  Error: {e}")
        print(f"\n  Using dummy mode for demonstration...")
        # Create dummy data for testing code structure
        from torch.utils.data import TensorDataset
        dummy_imgs = torch.randn(100, 3, 224, 224)
        dummy_labels = torch.randint(0, 2, (100, 14)).float()
        train_ds = TensorDataset(dummy_imgs, dummy_labels)
        val_ds = TensorDataset(dummy_imgs[:20], dummy_labels[:20])
        train_subset = train_ds
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Build model
    print(f"\n✓ Building ViT model: {config.backbone}")
    model = ViTWrapper(
        model_name=config.backbone,
        num_classes=config.num_classes,
        pretrained=True
    )
    
    # Optional: Apply LoRA for parameter efficiency
    USE_LORA = True
    if USE_LORA:
        print(f"✓ Applying LoRA (r=8, alpha=32)")
        apply_lora_to_model(model, r=8, alpha=32.0)
        
        # Freeze original parameters
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    
    # Count parameters
    params = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"  Total params: {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    print(f"  Efficiency: {100.0 * params['trainable'] / params['total']:.2f}% trainable")
    
    # Setup Lightning module
    lit_model = LitModel(model, config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/quick_test',
        filename='test-{epoch:02d}',
        monitor='val/auc',
        mode='max'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator='auto',
        devices=1,  # Single GPU for quick test
        precision=16 if config.mixed_precision else 32,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    
    print(f"\n🚀 Starting training (epochs={config.epochs})...")
    print("="*70)
    
    # Train
    try:
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        
        # Quick evaluation
        print("\n📊 Running quick evaluation...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    imgs = batch['image'].to(device)
                    labels = batch['labels'].numpy()
                else:
                    imgs, labels = batch
                    imgs = imgs.to(device)
                    labels = labels.numpy()
                
                logits = model(imgs)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = compute_metrics(all_preds, all_labels)
        
        print(f"\nFinal Metrics:")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Mean AP: {metrics['mean_ap']:.4f}")
        print(f"  Sensitivity: {metrics['mean_sens']:.4f}")
        print(f"  Specificity: {metrics['mean_spec']:.4f}")
        
        print("\n" + "="*70)
        print("✅ Quick start example completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Prepare your full datasets (CheXpert, NIH)")
        print("  2. Run full experiments with run_training.py")
        print("  3. Use the Colab notebook for comprehensive experiments")
        print("  4. See WORKFLOW.md for detailed guidance")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check your dataset paths and configuration.")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
