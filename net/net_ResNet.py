import torch.nn as nn
import torch.nn.functional as F


# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, first_layer=False):
        super(ResBlock, self).__init__()
        self.first_layer = first_layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if self.first_layer:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.first_layer:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


# 特征提取模型
class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(32, 32)
        self.layer2 = ResBlock(32, 32)
        self.layer3 = ResBlock(32, 64, first_layer=True)
        self.layer4 = ResBlock(64, 64)
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
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x

class ResNet_prune(nn.Module):
    def __init__(self, in_channels):
        super(ResNet_prune, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(8, 4, first_layer=True)
        self.layer2 = ResBlock(4, 8, first_layer=True)
        self.layer3 = ResBlock(8, 12, first_layer=True)
        self.layer4 = ResBlock(12, 16, first_layer=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x
