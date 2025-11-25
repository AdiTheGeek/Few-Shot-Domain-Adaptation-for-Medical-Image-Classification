import os
from typing import Optional, List, Tuple
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

CHEXPERT_CONDITIONS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]


class SimpleMedicalDataset(Dataset):
    """Generic CSV-based dataset for CheXpert / ChestX-ray14 style CSVs.
    CSV must contain a path column (e.g., 'Path', 'Image Index') and one column per condition.
    """
    def __init__(self, csv_path: str, img_root: str, conditions: List[str]=CHEXPERT_CONDITIONS,
                 transform=None, uncertainty_policy: str = "U-Zeros", path_col: str = None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.conditions = conditions
        self.transform = transform
        self.policy = uncertainty_policy
        
        # Auto-detect path column if not specified
        if path_col is None:
            if 'Path' in self.df.columns:
                self.path_col = 'Path'
            elif 'Image Index' in self.df.columns:
                self.path_col = 'Image Index'
            elif 'image_path' in self.df.columns:
                self.path_col = 'image_path'
            else:
                raise ValueError(f"Could not find path column. Available columns: {list(self.df.columns)}")
        else:
            self.path_col = path_col
        
        self._process()

    def _process(self):
        # Ensure condition columns exist
        for c in self.conditions:
            if c not in self.df.columns:
                self.df[c] = 0
        # handle -1 uncertainties according to policy
        if self.policy == "U-Zeros":
            self.df[self.conditions] = self.df[self.conditions].fillna(0).replace(-1, 0)
        elif self.policy == "U-Ones":
            self.df[self.conditions] = self.df[self.conditions].fillna(0).replace(-1, 1)
        elif self.policy == "U-Ignore":
            self.df[self.conditions] = self.df[self.conditions].fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path_raw = row[self.path_col]
        
        # Handle different path formats
        # If path contains 'CheXpert-v1.0-small', strip it and use just the relative path
        if 'CheXpert-v1.0-small' in str(img_path_raw):
            # Extract the part after 'CheXpert-v1.0-small/'
            img_path_rel = img_path_raw.split('CheXpert-v1.0-small/')[-1]
        else:
            img_path_rel = img_path_raw
        
        img_path = os.path.join(self.img_root, img_path_rel)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[self.conditions].values.astype(np.float32))
        return {
            'image': img,
            'labels': labels,
            'path': img_path,
            'index': idx
        }


def get_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])


def make_dataloaders(config, train_csv: str, val_csv: str, test_csv: Optional[str]=None,
                     use_domain_adaptation: bool=False, target_csv: Optional[str]=None):
    train_ds = SimpleMedicalDataset(train_csv, config.data_root, transform=get_transforms(config.img_size, True))
    val_ds = SimpleMedicalDataset(val_csv, config.data_root, transform=get_transforms(config.img_size, False))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    if use_domain_adaptation and target_csv is not None:
        target_ds = SimpleMedicalDataset(target_csv, config.data_root, transform=get_transforms(config.img_size, True))
        target_loader = DataLoader(target_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    else:
        target_loader = None
    return train_loader, target_loader, val_loader


def sample_few_shot_indices(dataset: SimpleMedicalDataset, k_per_class: int, seed: int = 42):
    """Sample a few-shot subset selecting up to k positive examples per class.
    Returns a list of indices (may include duplicates across classes; deduplicated at end).
    """
    rng = random.Random(seed)
    per_class_indices = []
    df = dataset.df
    for i, c in enumerate(dataset.conditions):
        positives = df.index[df[c] == 1].tolist()
        if len(positives) == 0:
            continue
        chosen = rng.sample(positives, min(k_per_class, len(positives)))
        per_class_indices.extend(chosen)
    # always include some negatives to avoid degenerate training
    all_indices = set(per_class_indices)
    # add random negatives until we have at least k_per_class * num_classes * 0.5 samples
    target_n = max(100, int(0.5 * k_per_class * len(dataset.conditions)))
    available = [i for i in range(len(dataset)) if i not in all_indices]
    if len(available) > 0:
        extra = rng.sample(available, min(len(available), max(0, target_n - len(all_indices))))
        all_indices.update(extra)
    return list(all_indices)
