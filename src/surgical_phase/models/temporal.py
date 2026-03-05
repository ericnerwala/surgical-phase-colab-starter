from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import ResNet18Encoder


class CNNLSTMPhaseModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, layers: int = 1, pretrained_cnn: bool = True):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained_cnn)
        self.lstm = nn.LSTM(self.encoder.out_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B,T,C,H,W]
        b, t, c, h, w = x.shape
        f = self.encoder(x.view(b * t, c, h, w)).view(b, t, -1)
        out, _ = self.lstm(f)
        return self.cls(out[:, -1, :])


class TemporalConvNetPhaseModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, levels: int = 4, kernel_size: int = 3, pretrained_cnn: bool = True):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained_cnn)
        layers = []
        in_ch = self.encoder.out_dim
        for i in range(levels):
            dil = 2 ** i
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=dil * (kernel_size - 1) // 2, dilation=dil),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ]
            in_ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        f = self.encoder(x.view(b * t, c, h, w)).view(b, t, -1).transpose(1, 2)  # [B,D,T]
        y = self.tcn(f).transpose(1, 2)
        return self.cls(y[:, -1, :])


class TransformerPhaseModel(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 256, nhead: int = 8, layers: int = 4, pretrained_cnn: bool = True):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained_cnn)
        self.proj = nn.Linear(self.encoder.out_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        f = self.encoder(x.view(b * t, c, h, w)).view(b, t, -1)
        z = self.tf(self.proj(f))
        return self.cls(z[:, -1, :])
