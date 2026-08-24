import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model