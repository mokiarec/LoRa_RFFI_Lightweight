import torch.nn as nn
import torch.nn.functional as F


# SCSK 核心模块：空间与通道锐化内核
class SCSKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, first_layer=False):
        super(SCSKBlock, self).__init__()
        self.first_layer = first_layer

        # 1. 通道注意力 (Channel Sharpening)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2 if in_channels > 1 else 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2 if in_channels > 1 else 1, in_channels, 1),
            nn.Sigmoid()
        )

        # 2. 空间增强卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 快捷连接
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if first_layer else nn.Identity()

    def forward(self, x):
        # 通道加权
        res = self.shortcut(x)
        x = x * self.ca(x)

        # 特征提取
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out += res
        return self.relu(out)


# SCSKNet 主体结构
class SCSKNet(nn.Module):
    def __init__(self, in_channels):
        super(SCSKNet, self).__init__()
        # 初始特征层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)

        # SCSK 层级堆叠
        self.layer1 = SCSKBlock(32, 32)
        self.layer2 = SCSKBlock(32, 32)
        self.layer3 = SCSKBlock(32, 64, first_layer=True)
        self.layer4 = SCSKBlock(64, 64)

        # 全局池化与特征投影
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)

class SCSKNet_prune(nn.Module):
    def __init__(self, in_channels):
        super(SCSKNet_prune, self).__init__()
        # 初始特征层
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=7, stride=2, padding=3)

        # SCSK 层级堆叠
        self.layer1 = SCSKBlock(4, 8, first_layer=True)
        self.layer2 = SCSKBlock(8, 8)
        self.layer3 = SCSKBlock(8, 8)
        self.layer4 = SCSKBlock(8, 8)

        # 全局池化与特征投影
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)