# ============================================================================
# data_loader.py - CheXpert Dataset and Preprocessing
# ============================================================================

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class CheXpertDataset(Dataset):
    """
    CheXpert multi-label chest X-ray dataset
    Handles uncertainty labels (-1) by policy (U-Zeros, U-Ones, U-Ignore, U-MultiClass)
    """
    
    # CheXpert 14 conditions
    CONDITIONS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(self, csv_path, img_root, transform=None, 
                 uncertainty_policy="U-Zeros", domain_label=None):
        """
        Args:
            csv_path: Path to train/val/test CSV
            img_root: Root directory for images
            transform: Augmentation pipeline
            uncertainty_policy: How to handle -1 labels
            domain_label: Integer label for domain (for DA)
        """
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        self.policy = uncertainty_policy
        self.domain_label = domain_label
        
        # Handle uncertainty labels
        self._process_labels()
    
    def _process_labels(self):
        """Convert uncertainty labels based on policy"""
        for cond in self.CONDITIONS:
            if cond in self.df.columns:
                if self.policy == "U-Zeros":
                    self.df[cond] = self.df[cond].fillna(0).replace(-1, 0)
                elif self.policy == "U-Ones":
                    self.df[cond] = self.df[cond].fillna(0).replace(-1, 1)
                elif self.policy == "U-Ignore":
                    # Keep -1 for masking during loss computation
                    self.df[cond] = self.df[cond].fillna(0)
                else:  # U-MultiClass
                    self.df[cond] = self.df[cond].fillna(0).replace(-1, 2)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Image path
        img_path = os.path.join(self.img_root, self.df.iloc[idx]['Path'])
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Extract labels
        labels = self.df.iloc[idx][self.CONDITIONS].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Create mask for valid labels (if U-Ignore policy)
        mask = (labels != -1).float() if self.policy == "U-Ignore" else None
        
        sample = {
            'image': image,
            'labels': labels,
            'mask': mask,
            'path': img_path
        }
        
        if self.domain_label is not None:
            sample['domain'] = torch.tensor(self.domain_label, dtype=torch.long)
        
        return sample


def get_transforms(img_size=224, is_training=True):
    """Get augmentation pipeline"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(config: Config):
    """Create train/val/test dataloaders"""
    train_transform = get_transforms(config.img_size, is_training=True)
    val_transform = get_transforms(config.img_size, is_training=False)
    
    # Standard training
    if not config.use_domain_adaptation:
        train_dataset = CheXpertDataset(
            csv_path=f"{config.data_root}/train.csv",
            img_root=config.data_root,
            transform=train_transform
        )
        val_dataset = CheXpertDataset(
            csv_path=f"{config.data_root}/valid.csv",
            img_root=config.data_root,
            transform=val_transform
        )
    else:
        # Domain adaptation: separate source/target loaders
        train_dataset = CheXpertDataset(
            csv_path=f"{config.data_root}/train_{config.source_domain}.csv",
            img_root=config.data_root,
            transform=train_transform,
            domain_label=0
        )
        target_dataset = CheXpertDataset(
            csv_path=f"{config.data_root}/train_{config.target_domain}.csv",
            img_root=config.data_root,
            transform=train_transform,
            domain_label=1
        )
        val_dataset = CheXpertDataset(
            csv_path=f"{config.data_root}/valid_{config.target_domain}.csv",
            img_root=config.data_root,
            transform=val_transform
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
    if config.use_domain_adaptation:
        target_loader = DataLoader(
            target_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
        return train_loader, target_loader, val_loader
    
    return train_loader, None, val_loader
