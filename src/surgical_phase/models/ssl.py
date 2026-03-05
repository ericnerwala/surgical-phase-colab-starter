from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNet18Encoder


class SimCLRProjection(nn.Module):
    def __init__(self, in_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(reps, reps.T) / temperature
    n = z1.size(0)
    labels = torch.arange(n, device=z1.device)
    labels = torch.cat([labels + n, labels], dim=0)
    mask = torch.eye(2 * n, device=z1.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    return F.cross_entropy(sim, labels)


class SimCLRFrameModel(nn.Module):
    def __init__(self, pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.proj = SimCLRProjection(in_dim=self.encoder.out_dim, proj_dim=proj_dim)

    def forward(self, x):
        return self.proj(self.encoder(x))
