import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 Dense 块中的单层结构
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # 标准 Bottleneck 结构：1x1 卷积压缩 + 3x3 卷积提取
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x 为之前所有层的输出拼接结果
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # 将输入与当前层输出在通道维度(dim=1)拼接
        return torch.cat([x, out], 1)


# 定义过渡层（Transition Layer）：用于控制通道数并减小特征图尺寸
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate=32, block_config=(6, 12, 24, 16)):
        super(DenseNet, self).__init__()

        # 初始特征层
        num_features = 64
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建 DenseBlocks
        layers = []
        for i, num_layers in enumerate(block_config):
            # 添加一个 Dense Block
            for j in range(num_layers):
                layers.append(DenseLayer(num_features, growth_rate))
                num_features += growth_rate

            # 如果不是最后一个 block，添加 Transition 层
            if i != len(block_config) - 1:
                out_features = num_features // 2
                layers.append(Transition(num_features, out_features))
                num_features = out_features

        self.features = nn.Sequential(*layers)

        # 分类/特征投影头
        self.final_bn = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_features, 512)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.features(x)
        x = F.relu(self.final_bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class DenseNet_prune(nn.Module):
    def __init__(self, in_channels, growth_rate=2, block_config=(4, 6, 8, 6)):
        super(DenseNet_prune, self).__init__()

        # 初始特征层 - 通道数减少到8
        num_features = 8
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建 DenseBlocks
        layers = []
        for i, num_layers in enumerate(block_config):
            # 添加一个 Dense Block
            for j in range(num_layers):
                layers.append(DenseLayer(num_features, growth_rate))
                num_features += growth_rate

            # 如果不是最后一个 block，添加 Transition 层
            if i != len(block_config) - 1:
                out_features = num_features // 2
                layers.append(Transition(num_features, out_features))
                num_features = out_features

        self.features = nn.Sequential(*layers)

        # 分类/特征投影头 - 输出维度减小到8
        self.final_bn = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_features, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.features(x)
        x = F.relu(self.final_bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)