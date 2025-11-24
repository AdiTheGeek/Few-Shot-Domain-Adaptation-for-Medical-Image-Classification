import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional


class ViTWrapper(nn.Module):
    """Wrapper around timm Vision Transformer to expose features and classifier.
    Supports prompt tokens and simple integration points for adapters/LoRA.
    """
    def __init__(self, model_name: str = 'vit_base_patch16_224', num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # timm ViT has classifier head named 'head'
        # expose feature dimension
        if hasattr(self.backbone, 'head'):
            # replace head with identity to get features
            try:
                feat_dim = self.backbone.head.in_features
            except Exception:
                feat_dim = self.backbone.num_features
            self.backbone.head = nn.Identity()
        else:
            feat_dim = self.backbone.num_features

        self.feat_dim = feat_dim
        # add a simple classification head
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_features: bool = False, **kwargs):
        features = self.backbone(x)  # [B, feat_dim]
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
