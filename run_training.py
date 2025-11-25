"""
run_training.py - Complete training driver for few-shot domain adaptation experiments

Usage examples:
    # Baseline ViT training on source domain
    python run_training.py --backbone vit_base_patch16_224 --data_root ./data --train_csv data/chexpert_train.csv --val_csv data/chexpert_val.csv --epochs 30

    # LoRA adaptation with few-shot target
    python run_training.py --use_lora --lora_r 8 --lora_alpha 32 --few_shot_k 50 --source_csv data/chexpert_train.csv --target_csv data/nih_train.csv --val_csv data/nih_val.csv

    # Adapter adaptation
    python run_training.py --use_adapter --adapter_dim 64 --few_shot_k 50 --target_csv data/nih_train.csv --val_csv data/nih_val.csv

    # Prompt tuning
    python run_training.py --use_prompt --prompt_tokens 10 --few_shot_k 50 --target_csv data/nih_train.csv --val_csv data/nih_val.csv

    # CNN baseline (ResNet50)
    python run_training.py --use_cnn --backbone resnet50 --train_csv data/chexpert_train.csv --val_csv data/chexpert_val.csv
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Subset

from configs.config import Config
from data.datasets import SimpleMedicalDataset, get_transforms, make_dataloaders, sample_few_shot_indices
from models.vit_backbone import ViTWrapper
from models.cnn_backbones import build_cnn
from lora.lora import apply_lora_to_model
from adapters.adapter import attach_adapter_to_vit
from prompts.prompt_tuning import attach_visual_prompt_to_vit
from train.trainer import LitModel
from utils.utils import set_seed, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot domain adaptation training")
    
    # Data
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for images')
    parser.add_argument('--train_csv', type=str, default=None, help='Training CSV (source domain)')
    parser.add_argument('--source_csv', type=str, default=None, help='Source CSV (alias for train_csv)')
    parser.add_argument('--target_csv', type=str, default=None, help='Target domain CSV for few-shot adaptation')
    parser.add_argument('--val_csv', type=str, required=True, help='Validation CSV')
    parser.add_argument('--test_csv', type=str, default=None, help='Test CSV (optional)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_224', help='Model backbone name')
    parser.add_argument('--use_cnn', action='store_true', help='Use CNN backbone instead of ViT')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--num_classes', type=int, default=14)
    
    # Adaptation methods
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA adaptation')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='LoRA alpha')
    parser.add_argument('--use_adapter', action='store_true', help='Enable adapter layers')
    parser.add_argument('--adapter_dim', type=int, default=64, help='Adapter bottleneck dimension')
    parser.add_argument('--use_prompt', action='store_true', help='Enable prompt tuning')
    parser.add_argument('--prompt_tokens', type=int, default=10, help='Number of prompt tokens')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone (for adapter/prompt only training)')
    
    # Few-shot
    parser.add_argument('--few_shot_k', type=int, default=50, help='K samples per class for few-shot')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Enable mixed precision')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True, help='Enable gradient checkpointing')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--project_name', type=str, default='fewshot-da-medical')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs (-1 for all available)')
    
    return parser.parse_args()


def build_model(config, args):
    """Build model based on configuration"""
    if args.use_cnn:
        print(f"Building CNN backbone: {config.backbone}")
        model = build_cnn(backbone_name=config.backbone, num_classes=config.num_classes, pretrained=config.pretrained)
    else:
        print(f"Building ViT backbone: {config.backbone}")
        model = ViTWrapper(model_name=config.backbone, num_classes=config.num_classes, pretrained=config.pretrained)
        
        # Enable gradient checkpointing for ViT
        if config.gradient_checkpointing:
            if hasattr(model.backbone, 'set_grad_checkpointing'):
                model.backbone.set_grad_checkpointing(True)
                print("✓ Gradient checkpointing enabled")
    
    return model


def apply_adaptation(model, config, args):
    """Apply adaptation method to model"""
    if args.use_lora:
        print(f"Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
        apply_lora_to_model(model, r=config.lora_r, alpha=config.lora_alpha)
        # Freeze original parameters, keep only LoRA trainable
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        print("✓ LoRA applied, original params frozen")
    
    if args.use_adapter:
        print(f"Attaching adapters: bottleneck_dim={config.adapter_dim}")
        attach_adapter_to_vit(model.backbone if hasattr(model, 'backbone') else model, adapter_dim=config.adapter_dim)
        # Freeze backbone, keep adapters trainable
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
        print("✓ Adapters attached, backbone frozen")
    
    if args.use_prompt:
        print(f"Attaching visual prompts: n_tokens={config.prompt_tokens}")
        attach_visual_prompt_to_vit(model.backbone if hasattr(model, 'backbone') else model, prompt_tokens=config.prompt_tokens)
        # Freeze everything except prompts
        for name, param in model.named_parameters():
            if 'visual_prompt' not in name and 'classifier' not in name:
                param.requires_grad = False
        print("✓ Visual prompts attached, backbone frozen")
    
    if args.freeze_backbone and not (args.use_lora or args.use_adapter or args.use_prompt):
        print("Freezing backbone (manual freeze)")
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False


def save_checkpoint(model, config, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, path)
    print(f"✓ Checkpoint saved: {path}")


def main():
    args = parse_args()
    
    # Create config from args
    config = Config(
        data_root=args.data_root,
        img_size=args.img_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backbone=args.backbone,
        pretrained=args.pretrained,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_adapter=args.use_adapter,
        adapter_dim=args.adapter_dim,
        use_prompt=args.use_prompt,
        prompt_tokens=args.prompt_tokens,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        few_shot_k=args.few_shot_k,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        seed=args.seed
    )
    
    # Set seed
    set_seed(config.seed)
    print(f"🌱 Seed set to {config.seed}")
    
    # Determine train CSV (source_csv or train_csv)
    train_csv = args.source_csv if args.source_csv else args.train_csv
    if train_csv is None:
        raise ValueError("Must provide --train_csv or --source_csv")
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {args.experiment_name}")
    print("="*70)
    print(f"Backbone: {config.backbone}")
    print(f"Adaptation: LoRA={args.use_lora}, Adapter={args.use_adapter}, Prompt={args.use_prompt}")
    print(f"Few-shot K: {config.few_shot_k}")
    print(f"Epochs: {config.epochs}, LR: {config.lr}")
    print(f"Mixed precision: {config.mixed_precision}, Grad checkpoint: {config.gradient_checkpointing}")
    print("="*70 + "\n")
    
    # Build model
    model = build_model(config, args)
    
    # Apply adaptation methods
    apply_adaptation(model, config, args)
    
    # Count parameters
    param_stats = count_parameters(model)
    print(f"\n📊 Model parameters:")
    print(f"   Total: {param_stats['total']:,}")
    print(f"   Trainable: {param_stats['trainable']:,}")
    print(f"   Efficiency: {100.0 * param_stats['trainable'] / param_stats['total']:.2f}% trainable\n")
    
    # Prepare dataloaders
    print("Loading datasets...")
    
    if args.target_csv:
        # Few-shot adaptation scenario
        print(f"Source: {train_csv}")
        print(f"Target (few-shot): {args.target_csv}")
        
        # Load full target dataset
        target_ds_full = SimpleMedicalDataset(args.target_csv, config.data_root, transform=get_transforms(config.img_size, True))
        
        # Sample few-shot indices
        few_shot_indices = sample_few_shot_indices(target_ds_full, k_per_class=config.few_shot_k, seed=config.seed)
        print(f"Few-shot sampling: {len(few_shot_indices)} samples selected (k={config.few_shot_k} per class)")
        
        # Create few-shot subset
        target_ds = Subset(target_ds_full, few_shot_indices)
        train_loader = torch.utils.data.DataLoader(target_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    else:
        # Standard training (no adaptation)
        print(f"Training: {train_csv}")
        train_ds = SimpleMedicalDataset(train_csv, config.data_root, transform=get_transforms(config.img_size, True))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    # Validation loader
    val_ds = SimpleMedicalDataset(args.val_csv, config.data_root, transform=get_transforms(config.img_size, False))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Setup Lightning module
    lit_model = LitModel(model, config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint_dir, args.experiment_name),
        filename='best-{epoch:02d}-{val/auc:.4f}',
        monitor='val/auc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/auc',
        patience=args.early_stopping_patience,
        mode='max',
        verbose=True
    )
    
    callbacks = [checkpoint_callback, early_stop_callback]
    
    # W&B logger (optional)
    logger = None
    if config.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=config.project_name, name=args.experiment_name)
    
    # Trainer configuration (multi-GPU, mixed precision, gradient checkpointing)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator='gpu',  # Use GPU acceleration (Colab Pro+ multi-GPU support)
        devices=args.gpus if args.gpus > 0 else -1,  # -1 = all available GPUs
        precision='16-mixed' if config.mixed_precision else 32,  # Mixed precision (fp16)
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    print("🚀 Starting training...\n")
    
    # Train
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(config.checkpoint_dir, args.experiment_name, 'final_model.pth')
    save_checkpoint(model, config, final_checkpoint_path)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val AUC: {checkpoint_callback.best_model_score:.4f}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print("="*70 + "\n")
    
    # Optional: test evaluation
    if args.test_csv:
        print("Running test evaluation...")
        test_ds = SimpleMedicalDataset(args.test_csv, config.data_root, transform=get_transforms(config.img_size, False))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        test_results = trainer.test(lit_model, dataloaders=test_loader)
        print(f"Test results: {test_results}")


if __name__ == '__main__':
    main()
