import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.optim import AdamW
from typing import Any, Optional
from eval.evaluator import compute_auc, compute_metrics
from utils.utils import count_parameters


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs = batch['image']
        labels = batch['labels']
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['image']
        labels = batch['labels']
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        output = {'preds': preds, 'labels': labels_np, 'loss': loss}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if len(outputs) == 0:
            return
        preds = np.vstack([o['preds'] for o in outputs])
        labels = np.vstack([o['labels'] for o in outputs])
        auc = compute_auc(preds, labels)
        metrics = compute_metrics(preds, labels)
        self.log('val/auc', auc, prog_bar=True, sync_dist=True)
        self.log('val/mean_ap', metrics['mean_ap'], sync_dist=True)
        self.log('val/mean_sens', metrics['mean_sens'], sync_dist=True)
        self.log('val/mean_spec', metrics['mean_spec'], sync_dist=True)
        if self.config.use_wandb:
            import wandb
            wandb.log({'val/auc': auc, 'val/mean_ap': metrics['mean_ap'], 
                      'val/sensitivity': metrics['mean_sens'], 'val/specificity': metrics['mean_spec']})
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        imgs = batch['image']
        labels = batch['labels']
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        output = {'preds': preds, 'labels': labels_np, 'loss': loss}
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        if len(outputs) == 0:
            return
        preds = np.vstack([o['preds'] for o in outputs])
        labels = np.vstack([o['labels'] for o in outputs])
        metrics = compute_metrics(preds, labels)
        self.log('test/auc', metrics['auc_roc'], sync_dist=True)
        self.log('test/mean_ap', metrics['mean_ap'], sync_dist=True)
        self.log('test/mean_sens', metrics['mean_sens'], sync_dist=True)
        self.log('test/mean_spec', metrics['mean_spec'], sync_dist=True)
        print(f"\n📊 Test Metrics:")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   Mean AP: {metrics['mean_ap']:.4f}")
        print(f"   Sensitivity: {metrics['mean_sens']:.4f}")
        print(f"   Specificity: {metrics['mean_spec']:.4f}\n")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if self.config.optimizer == 'adamw':
            opt = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            opt = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config.epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}


def build_trainer_and_run(model, config, train_loader, val_loader):
    lit = LitModel(model, config)
    callbacks = []
    trainer = pl.Trainer(max_epochs=config.epochs, precision=16 if config.mixed_precision else 32,
                        accelerator='auto', callbacks=callbacks)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer, lit
