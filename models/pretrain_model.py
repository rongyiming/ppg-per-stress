"""
Pretraining model: predicts a segment's 10 handcrafted descriptors from the
segment itself (self-supervised pretext task).

Supports ResNet-18/34/50 and InceptionTime backbones.  Output: ``(B, 10)``.
"""

import torch
from torch import nn

from models.backbones.resnet1d import resnet18_1d


class PretrainModel(nn.Module):
    def __init__(self, input_channels, resnet_depth=18, hidden_size=64,
                 output_size=10, backbone='resnet18'):
        super(PretrainModel, self).__init__()

        if backbone == 'inception_time':
            from models.backbones.inception_time import InceptionTime
            self.backbone = InceptionTime(input_channels=input_channels)
            self._backbone_resnet = False
        elif backbone == 'resnet18':
            self.backbone = resnet18_1d(input_channels=input_channels)
            self._backbone_resnet = True
        elif backbone == 'resnet34':
            from models.backbones.resnet1d import resnet34_1d
            self.backbone = resnet34_1d(input_channels=input_channels)
            self._backbone_resnet = True
        elif backbone == 'resnet50':
            from models.backbones.resnet1d import resnet50_1d
            self.backbone = resnet50_1d(input_channels=input_channels)
            self._backbone_resnet = True
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.resnet_output_size = 512
        self.linear = nn.Linear(self.resnet_output_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), x.size(1), -1)   # (B, L) → (B, 1, L)
        x = x.permute(0, 2, 1)                  # (B, C=1, L)

        if self._backbone_resnet:
            x = self.backbone(x)                 # (B, 512, T)
        else:
            # InceptionTime returns [l0, l1, l2, l3]; use deepest level
            f_levels = self.backbone(x)
            x = f_levels[-1]                     # (B, 512, T)

        x = x.permute(0, 2, 1)                  # (B, T, 512)
        x = self.linear(x)                       # (B, T, hidden_size)
        x = self.relu(x)
        x = torch.mean(x, dim=1)                 # (B, hidden_size) — GAP
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x
