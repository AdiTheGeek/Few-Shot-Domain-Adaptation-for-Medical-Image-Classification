# ============================================================================
# main.py - Orchestration Script
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training script"""
    
    # ========================================================================
    # CONFIGURATION - Modify this section for different experiments
    # ========================================================================
    
    config = Config(
        # Data
        data_root="./chexpert_data",
        img_size=224,
        batch_size=32,
        
        # Model Selection - CHANGE THIS TO EXPERIMENT
        backbone="resnet50",  # Try: resnet18, resnet50, densenet121, efficientnet_b0, unet
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.3,
        
        # Domain Adaptation - CHANGE THIS TO EXPERIMENT
        use_domain_adaptation=False,  # Set to True for DA experiments
        da_method="none",  # Try: dann, coral, mmd, mcd
        da_weight=0.1,
        source_domain="frontal",
        target_domain="lateral",
        
        # Training
        epochs=50,
        lr=1e-4,
        weight_decay=1e-4,
        scheduler="cosine",
        early_stopping_patience=10,
        
        # Experiment tracking
        experiment_name="resnet50_baseline",  # Change for each experiment
        seed=42
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    if config.use_domain_adaptation:
        train_loader, target_loader, val_loader = get_dataloaders(config)
        print(f"Source domain: {config.source_domain}, Target domain: {config.target_domain}")
        print(f"Source batches: {len(train_loader)}, Target batches: {len(target_loader)}")
    else:
        train_loader, _, val_loader = get_dataloaders(config)
        target_loader = None
        print(f"Standard training")
        print(f"Train batches: {len(train_loader)}")
    
    print(f"Val batches: {len(val_loader)}")
    
    # ========================================================================
    # MODEL BUILDING
    # ========================================================================
    
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    # Create base classifier
    classifier = FlexibleClassifier(
        backbone=config.backbone,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout_rate
    )
    
    # Optionally freeze backbone
    if config.freeze_backbone:
        classifier.freeze_backbone()
        print(f"Backbone frozen for feature extraction")
    
    # Wrap with domain adaptation if needed
    if config.use_domain_adaptation and config.da_method != 'none':
        model = DomainAdaptationWrapper(classifier, config.da_method)
        print(f"Model wrapped with {config.da_method.upper()} domain adaptation")
    else:
        model = classifier
        print(f"Using standard classifier")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    trainer = Trainer(model, config, device=device)
    trainer.fit(train_loader, val_loader, target_loader)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation AUC: {trainer.best_val_auc:.4f}")
    print(f"Checkpoint saved at: {config.checkpoint_dir}/{config.experiment_name}_best.pth")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # Load best checkpoint
    checkpoint = torch.load(f"{config.checkpoint_dir}/{config.experiment_name}_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    val_metrics = trainer.validate(val_loader)
    metrics = compute_metrics(val_metrics['predictions'], val_metrics['labels'])
    
    print("\nFinal Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Mean AP: {metrics['mean_ap']:.4f}")
    print(f"  Mean F1: {metrics['mean_f1']:.4f}")
    
    return model, metrics
