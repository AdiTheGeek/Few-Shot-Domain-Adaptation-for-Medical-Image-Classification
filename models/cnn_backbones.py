import torch.nn as nn
import timm


def build_cnn(backbone_name: str = 'resnet50', num_classes: int = 14, pretrained: bool = True):
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    return model
