# models/simple_cnn.py (3D Version for BraTS)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    用于联邦学习快速验证的轻量级 3D CNN
    适配 BraTS 3D 数据 (Input: BxCxDxHxW)
    """

    def __init__(self, num_classes=4, in_channels=4):
        """
        Args:
            num_classes: 分类类别数
            in_channels: 输入通道数 (BraTS 默认为 4)
        """
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels

        # 特征提取层 (3D 卷积)
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # /2

            # Conv 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # /4

            # Conv 3
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # /8
        )

        # 分类层
        # 使用自适应池化，无论输入尺寸如何，都固定输出尺寸
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 128),  # 4x4x4 after AdaptivePool
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)

        # 自适应池化到 4x4x4
        x = F.adaptive_avg_pool3d(x, (4, 4, 4))

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x