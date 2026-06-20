import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, base_filters=16):
        """
        BraTS 输入通常是 4 通道。
        base_filters 设为 16 或 32。如果是 32，显存消耗会很大，FedMed 边缘设备通常用 16。
        """
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv3D(n_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters * 4, base_filters * 8))

        # Decoder
        self.up1 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(base_filters * 8, base_filters * 4)  # Concat后通道翻倍

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(base_filters * 4, base_filters * 2)

        self.up3 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(base_filters * 2, base_filters)

        self.outc = nn.Conv3d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 4, D, H, W) e.g., (B, 4, 96, 96, 96)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # Bottleneck

        x = self.up1(x4)
        # 3D skip connection
        # 假设输入尺寸是 16 的倍数，不需要 padding 处理，直接 cat
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return logits