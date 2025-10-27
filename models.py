# ============================================================================
# models.py - Flexible Model Architectures
# ============================================================================

import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class FlexibleClassifier(nn.Module):
    """
    Modular classifier with swappable backbones
    Supports: ResNet, DenseNet, EfficientNet, UNet, Vision Transformer
    """
    
    def __init__(self, backbone: str, num_classes: int, 
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Get backbone and feature dimension
        self.backbone, self.feature_dim = self._build_backbone(backbone, pretrained)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # For domain adaptation
        self.feature_extractor = self.backbone  # Alias for clarity
    
    def _build_backbone(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Build backbone and return feature dimension"""
        
        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                model = models.resnet18(pretrained=pretrained)
                feat_dim = 512
            elif backbone == 'resnet34':
                model = models.resnet34(pretrained=pretrained)
                feat_dim = 512
            elif backbone == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
                feat_dim = 2048
            elif backbone == 'resnet101':
                model = models.resnet101(pretrained=pretrained)
                feat_dim = 2048
            else:
                raise ValueError(f"Unknown ResNet variant: {backbone}")
            
            # Remove classifier
            backbone_net = nn.Sequential(*list(model.children())[:-1])
            return backbone_net, feat_dim
        
        elif backbone.startswith('densenet'):
            if backbone == 'densenet121':
                model = models.densenet121(pretrained=pretrained)
                feat_dim = 1024
            elif backbone == 'densenet169':
                model = models.densenet169(pretrained=pretrained)
                feat_dim = 1664
            else:
                raise ValueError(f"Unknown DenseNet variant: {backbone}")
            
            backbone_net = nn.Sequential(*list(model.children())[:-1])
            return backbone_net, feat_dim
        
        elif backbone.startswith('efficientnet'):
            if backbone == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=pretrained)
                feat_dim = 1280
            elif backbone == 'efficientnet_b1':
                model = models.efficientnet_b1(pretrained=pretrained)
                feat_dim = 1280
            else:
                raise ValueError(f"Unknown EfficientNet variant: {backbone}")
            
            backbone_net = nn.Sequential(*list(model.children())[:-1])
            return backbone_net, feat_dim
        
        elif backbone == 'unet':
            # Simple U-Net encoder (you can expand this)
            backbone_net = SimpleUNetEncoder()
            feat_dim = 512
            return backbone_net, feat_dim
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        Args:
            x: Input images [B, C, H, W]
            return_features: If True, return (logits, features) for DA
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Classify
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone for feature extraction"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class SimpleUNetEncoder(nn.Module):
    """Simple U-Net encoder for demonstration"""
    def __init__(self):
        super().__init__()
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return self.global_pool(x4)
