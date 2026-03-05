from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        w = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        m = tvm.resnet18(weights=w)
        self.features = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = out_dim

    def forward(self, x):
        z = self.features(x)  # [B,512,1,1]
        return z.flatten(1)


class FrameClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.head = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):
        f = self.encoder(x)
        return self.head(f)
