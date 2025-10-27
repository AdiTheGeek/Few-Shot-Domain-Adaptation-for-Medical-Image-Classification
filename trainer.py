# ============================================================================
# trainer.py - Training Loop with DA Support
# ============================================================================

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

class Trainer:
    """Unified trainer for baseline and domain adaptation"""
    
    def __init__(self, model, config: Config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9
            )
        
        # Scheduler
        if config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5
            )
        
        # Loss function (BCEWithLogitsLoss for multi-label)
        if config.pos_weight is not None:
            pos_weight = torch.tensor(config.pos_weight, device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Tracking
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0
    
    def train_epoch(self, train_loader, target_loader=None):
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_da_loss = 0.0
        
        # For domain adaptation, iterate both loaders
        if target_loader is not None:
            target_iter = iter(target_loader)
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, source_batch in enumerate(pbar):
            # Source data
            source_img = source_batch['image'].to(self.device)
            source_labels = source_batch['labels'].to(self.device)
            source_mask = source_batch.get('mask')
            if source_mask is not None:
                source_mask = source_mask.to(self.device)
            
            # Target data (for DA)
            if target_loader is not None:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                
                target_img = target_batch['image'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.mixed_precision):
                if not self.config.use_domain_adaptation:
                    # Standard training
                    logits = self.model(source_img)
                    if source_mask is not None:
                        loss = (self.criterion(logits, source_labels) * source_mask).mean()
                    else:
                        loss = self.criterion(logits, source_labels)
                    task_loss = loss
                    da_loss = torch.tensor(0.0)
                
                else:
                    # Domain adaptation training
                    # Forward source
                    source_logits, source_feats, _ = self.model(
                        source_img, domain_label=torch.zeros(source_img.size(0), device=self.device),
                        return_da_loss=True
                    )
                    task_loss = self.criterion(source_logits, source_labels)
                    
                    # Forward target
                    target_logits, target_feats, dann_loss = self.model(
                        target_img, domain_label=torch.ones(target_img.size(0), device=self.device),
                        return_da_loss=True
                    )
                    
                    # Compute DA loss based on method
                    if self.config.da_method == 'dann':
                        da_loss = dann_loss
                    elif self.config.da_method == 'coral':
                        da_loss = DomainAdaptationWrapper.coral_loss(source_feats, target_feats)
                    elif self.config.da_method == 'mmd':
                        da_loss = DomainAdaptationWrapper.mmd_loss(source_feats, target_feats)
                    else:
                        da_loss = torch.tensor(0.0, device=self.device)
                    
                    loss = task_loss + self.config.da_weight * da_loss
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_da_loss += da_loss.item() if isinstance(da_loss, torch.Tensor) else da_loss
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'da': f'{da_loss.item():.4f}' if isinstance(da_loss, torch.Tensor) else '0.0'
            })
        
        return {
            'total_loss': total_loss / len(train_loader),
            'task_loss': total_task_loss / len(train_loader),
            'da_loss': total_da_loss / len(train_loader)
        }
    
    def validate(self, val_loader):
        """Validation epoch"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        return {
            'loss': total_loss / len(val_loader),
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def fit(self, train_loader, val_loader, target_loader=None):
        """Full training loop"""
        print(f"Starting training: {self.config.experiment_name}")
        print(f"Model: {self.config.backbone}, DA: {self.config.da_method}")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, target_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Compute AUC (implement in evaluator.py)
            val_auc = compute_auc(val_metrics['predictions'], val_metrics['labels'])
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_auc:.4f}")
            
            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_auc)
            else:
                self.scheduler.step()
            
            # Early stopping and checkpointing
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_auc)
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print("Early stopping triggered")
                break
    
    def save_checkpoint(self, epoch, val_auc):
        """Save model checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = f"{self.config.checkpoint_dir}/{self.config.experiment_name}_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_auc': val_auc,
            'config': self.config
        }, path)
        print(f"Checkpoint saved: {path}")
