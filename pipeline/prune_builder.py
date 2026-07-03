"""从 PCA 通道配置构建剪枝网络 —— 支持全架构动态剪枝。

每个 builder 产出一个 nn.Module（embedding net），通道配置由
pca_conv.analyze_conv_redundancy() 的结果推导而来。
"""

import sys
from pathlib import Path

_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 共享构建块
# ============================================================

class ResBlock(nn.Module):
    """残差块，通道数变化时使用 1x1 卷积做 shortcut 投影"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)


class SCSKBlock(nn.Module):
    """空间-通道锐化核模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // 2), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // 2), in_channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        res = self.shortcut(x)
        x = x * self.ca(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + res)


# ============================================================
# PCA 可配置 ResNet
# ============================================================

class PrunableResNet(nn.Module):
    """通道数由构造参数决定的 ResNet。

    Args:
        in_channels: 输入通道数（来自预处理）
        channels: 长度为 5 的列表 [c0, c1, c2, c3, c4]
        embedding_dim: 最终 L2 归一化 embedding 维度
    """

    def __init__(self, in_channels, channels, embedding_dim=8):
        super().__init__()
        c0, c1, c2, c3, c4 = channels

        self.conv1 = nn.Conv2d(in_channels, c0, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(c0, c1)
        self.layer2 = ResBlock(c1, c2)
        self.layer3 = ResBlock(c2, c3)
        self.layer4 = ResBlock(c3, c4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c4, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)


# ============================================================
# PCA 可配置 SCSKNet
# ============================================================

class PrunableSCSKNet(nn.Module):
    """SCSKNet，通道数可配置。参数布局与 PrunableResNet 相同。"""

    def __init__(self, in_channels, channels, embedding_dim=8):
        super().__init__()
        c0, c1, c2, c3, c4 = channels

        self.conv1 = nn.Conv2d(in_channels, c0, kernel_size=7, stride=2, padding=3)
        self.layer1 = SCSKBlock(c0, c1)
        self.layer2 = SCSKBlock(c1, c2)
        self.layer3 = SCSKBlock(c2, c3)
        self.layer4 = SCSKBlock(c3, c4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c4, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)


# ============================================================
# PCA 可配置 DenseNet
# ============================================================

class PrunableDenseLayer(nn.Module):
    """DenseNet 基本层：BN-ReLU-Conv1x1-BN-ReLU-Conv3x3"""

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class PrunableTransition(nn.Module):
    """过渡层：1x1 卷积 + 平均池化，将通道数减半"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


class PrunableDenseNet(nn.Module):
    """通道数可配置的 DenseNet。

    Args:
        in_channels: 输入通道数
        init_channels: 初始 conv1 输出通道数（原版 64，剪枝版 8）
        growth_rate: 每个 DenseLayer 新增的通道数（原版 32，剪枝版 2）
        block_config: 每个 DenseBlock 的层数，如 (4, 6, 8, 6)
        embedding_dim: 最终 embedding 维度
    """

    def __init__(self, in_channels, init_channels=8, growth_rate=2,
                 block_config=(4, 6, 8, 6), embedding_dim=8):
        super().__init__()
        num_features = init_channels

        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i, num_layers in enumerate(block_config):
            for _ in range(num_layers):
                layers.append(PrunableDenseLayer(num_features, growth_rate))
                num_features += growth_rate
            if i != len(block_config) - 1:
                out_features = num_features // 2
                layers.append(PrunableTransition(num_features, out_features))
                num_features = out_features

        self.features = nn.Sequential(*layers)
        self.final_bn = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_features, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.features(x)
        x = F.relu(self.final_bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ============================================================
# PCA 可配置 GoogleNet
# ============================================================

class PrunableInception(nn.Module):
    """可配置通道数的 Inception 模块。

    Args:
        in_channels: 输入通道数
        c1:  分支1 的 1x1 卷积输出通道
        c2:  (c2_in, c2_out) 分支2 的 1x1→3x3 通道
        c3:  (c3_in, c3_out) 分支3 的 1x1→5x5 通道
        c4:  分支4 的 maxpool 后 1x1 卷积输出通道
    """

    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        c2_in, c2_out = c2
        c3_in, c3_out = c3

        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2_in, kernel_size=1)
        self.p2_2 = nn.Conv2d(c2_in, c2_out, kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3_in, kernel_size=1)
        self.p3_2 = nn.Conv2d(c3_in, c3_out, kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

    @property
    def out_channels(self):
        """本 Inception 的总输出通道数"""
        return self.p1_1.out_channels + self.p2_2.out_channels + \
               self.p3_2.out_channels + self.p4_2.out_channels


class PrunableGoogleNet(nn.Module):
    """通道数可配置的 GoogleNet。

    Args:
        in_channels: 输入通道数
        b1_out: block1 输出通道（conv 7x7），原版 64
        b2_mid: block2 中间通道（conv 1x1），原版 64
        b2_out: block2 输出通道（conv 3x3），原版 192
        inception_configs: 9 个 Inception 的配置列表，
            每个为 (c1, (c2_in, c2_out), (c3_in, c3_out), c4)
        embedding_dim: 最终 embedding 维度
    """

    def __init__(self, in_channels, b1_out=8, b2_mid=8, b2_out=16,
                 inception_configs=None, embedding_dim=8):
        super().__init__()
        if inception_configs is None:
            # 默认使用 GoogleNet_prune 的硬编码配置
            inception_configs = [
                (4, (5, 8), (2, 4), 8),     # 3a
                (2, (2, 4), (2, 2), 4),     # 3b
                (4, (4, 8), (2, 4), 4),     # 4a
                (4, (4, 8), (2, 4), 4),     # 4b
                (4, (4, 8), (2, 4), 4),     # 4c
                (4, (4, 8), (2, 4), 4),     # 4d
                (4, (4, 8), (2, 4), 4),     # 4e
                (4, (4, 8), (2, 4), 4),     # 5a
                (4, (4, 8), (2, 4), 4),     # 5b
            ]

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, b1_out, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(b1_out, b2_mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b2_mid, b2_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 构建 Inception 序列，自动计算级联输入通道
        in_ch = b2_out
        b3_modules = []
        for i, cfg in enumerate(inception_configs[:2]):
            c1, c2, c3, c4 = cfg
            b3_modules.append(PrunableInception(in_ch, c1, c2, c3, c4))
            in_ch = b3_modules[-1].out_channels
        b3_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = nn.Sequential(*b3_modules)

        b4_modules = []
        for i, cfg in enumerate(inception_configs[2:7]):
            c1, c2, c3, c4 = cfg
            b4_modules.append(PrunableInception(in_ch, c1, c2, c3, c4))
            in_ch = b4_modules[-1].out_channels
        b4_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = nn.Sequential(*b4_modules)

        b5_modules = []
        for i, cfg in enumerate(inception_configs[7:9]):
            c1, c2, c3, c4 = cfg
            b5_modules.append(PrunableInception(in_ch, c1, c2, c3, c4))
            in_ch = b5_modules[-1].out_channels
        b5_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.b5 = nn.Sequential(*b5_modules)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_ch, embedding_dim)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ============================================================
# PCA 可配置 ShuffleNetV2
# ============================================================

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class PrunableShuffleBlock(nn.Module):
    """ShuffleNetV2 基本单元（通道数可配置）"""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_features,
                      branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3,
                      stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class PrunableShuffleNetV2(nn.Module):
    """通道数可配置的 ShuffleNetV2。

    Args:
        in_channels: 输入通道数
        stages_out_channels: [c0, c1, c2, c3]
            c0 = conv1 输出通道
            c1 = stage0 输出通道
            c2 = stage1 输出通道
            c3 = stage2 输出通道
        stages_repeats: 每个 stage 的重复次数，默认 (4, 8, 4)
        conv5_out: conv5 输出通道
        embedding_dim: 最终 embedding 维度
    """

    def __init__(self, in_channels, stages_out_channels=(8, 24, 48, 96),
                 stages_repeats=(4, 8, 4), conv5_out=32, embedding_dim=8):
        super().__init__()
        input_channels = stages_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        for i in range(len(stages_repeats)):
            num_repeat = stages_repeats[i]
            output_channels = stages_out_channels[i + 1]
            layers = [PrunableShuffleBlock(input_channels, output_channels, stride=2)]
            for _ in range(num_repeat - 1):
                layers.append(PrunableShuffleBlock(output_channels, output_channels, stride=1))
            self.stages.append(nn.Sequential(*layers))
            input_channels = output_channels

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, conv5_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(conv5_out),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv5_out, embedding_dim)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        for stage in self.stages:
            x = stage(x)
        x = self.conv5(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ============================================================
# PCA 可配置 MobileNetV1
# ============================================================

class PrunableMobileNetV1(nn.Module):
    """通道数可配置的 MobileNetV1。

    Args:
        in_channels: 输入通道数
        channels: 14 个 int 列表，每个深度可分离块的输出通道数
        embedding_dim: 最终 embedding 维度
    """

    def __init__(self, in_channels, channels, embedding_dim=8):
        super().__init__()
        # 默认通道配置（与 LightNet_prune 的宽度乘数 1/16 对应）
        if channels is None or len(channels) < 14:
            wm = 1.0 / 16
            channels = [int(c * wm) for c in
                        [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]]

        strides = [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        # 初始层
        layers = [conv_bn(in_channels, channels[0], strides[0])]
        # 深度可分离层
        for i in range(1, len(channels)):
            layers.append(conv_dw(channels[i - 1], channels[i], strides[i]))

        self.model = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)


# ============================================================
# PCA 可配置 LightNet
# ============================================================

class PrunableLightNet(nn.Module):
    """通道数可配置的 LightNet（带残差的 MobileNetV1）。

    Args:
        in_channels: 输入通道数
        channels: N 个 int 列表，每个残差深度可分离块的输出通道数。
                  默认 8 层（与 LightNet_prune 一致）
        embedding_dim: 最终 embedding 维度
    """

    def __init__(self, in_channels, channels=None, embedding_dim=8):
        super().__init__()
        if channels is None:
            wm = 1.0 / 16
            channels = [int(c * wm) for c in
                        [32, 64, 128, 128, 256, 256, 512, 512, 1024]]

        strides = [2]  # 初始卷积步长
        # 残差块步长：2, 1, 2, 1, 2, 1, 2, 1（与 LightNet_prune 保持一致）
        for i in range(len(channels) - 1):
            if i % 2 == 0:
                strides.append(2)
            else:
                strides.append(1)

        # 初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, strides[0], 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True)
        )

        class ResidualDW(nn.Module):
            def __init__(self, inp, oup, stride):
                super().__init__()
                self.stride = stride
                self.use_res = stride == 1 and inp == oup
                self.depthwise = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU6(inplace=True)
                )
                self.pointwise = nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            def forward(self, x):
                if self.use_res:
                    return x + self.pointwise(self.depthwise(x))
                return self.pointwise(self.depthwise(x))

        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(
                ResidualDW(channels[i], channels[i + 1], strides[i + 1]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)


# ============================================================
# 构建器统一入口
# ============================================================

def build_pruned_embedding_net(net_type: str, in_channels: int, pca_channels,
                               embedding_dim: int = 8) -> nn.Module:
    """根据 PCA 通道配置构建剪枝后的 embedding 网络。

    Args:
        net_type: 网络类型字符串
        in_channels: 预处理后的输入通道数
        pca_channels: derive_channels_from_pca 返回的通道配置。
                      各类型格式不同，见对应 derive_* 函数。
        embedding_dim: 输出 embedding 维度

    Returns:
        nn.Module: embedding 网络（未包裹 TripletNet）
    """
    t = net_type.lower().replace('_', '').replace('-', '')

    if 'resnet' in t:
        ch = pca_channels.get('channels', pca_channels) if isinstance(pca_channels, dict) else pca_channels
        return PrunableResNet(in_channels, ch, embedding_dim)

    if 'scsknet' in t:
        ch = pca_channels.get('channels', pca_channels) if isinstance(pca_channels, dict) else pca_channels
        return PrunableSCSKNet(in_channels, ch, embedding_dim)

    if 'densenet' in t:
        return PrunableDenseNet(in_channels,
                                init_channels=pca_channels.get('init_channels', 8),
                                growth_rate=pca_channels.get('growth_rate', 2),
                                block_config=pca_channels.get('block_config', (4, 6, 8, 6)),
                                embedding_dim=embedding_dim)

    if 'googlenet' in t:
        return PrunableGoogleNet(in_channels,
                                 b1_out=pca_channels.get('b1_out', 8),
                                 b2_mid=pca_channels.get('b2_mid', 8),
                                 b2_out=pca_channels.get('b2_out', 16),
                                 inception_configs=pca_channels.get('inception_configs'),
                                 embedding_dim=embedding_dim)

    if 'shufflenet' in t:
        return PrunableShuffleNetV2(in_channels,
                                    stages_out_channels=pca_channels.get(
                                        'stages_out_channels', (8, 24, 48, 96)),
                                    conv5_out=pca_channels.get('conv5_out', 32),
                                    embedding_dim=embedding_dim)

    if 'mobilenetv1' in t:
        return PrunableMobileNetV1(in_channels,
                                   channels=pca_channels.get('channels'),
                                   embedding_dim=embedding_dim)

    if 'mobilenetv2' in t:
        return PrunableLightNet(in_channels,
                                channels=pca_channels.get('channels'),
                                embedding_dim=embedding_dim)

    if 'lightnet' in t:
        return PrunableLightNet(in_channels,
                                channels=pca_channels.get('channels'),
                                embedding_dim=embedding_dim)

    raise ValueError(f"build_pruned_embedding_net 不支持 {net_type}。"
                     f"支持: ResNet, SCSKNet, DenseNet, GoogleNet, ShuffleNet, "
                     f"MobileNetV1, MobileNetV2, LightNet")


# ============================================================
# PCA 通道推导 —— 辅助函数
# ============================================================

def _filter_main_convs(pca_results, *exclude_patterns):
    """过滤掉 shortcut/downsample/depthwise 等辅助卷积。

    返回按原始顺序排列的 (key, result) 列表。
    """
    if not exclude_patterns:
        exclude_patterns = ('shortcut', 'downsample', 'depthwise')
    return [(k, v) for k, v in pca_results.items()
            if not any(p in k for p in exclude_patterns)]


def _safe_effective_dim(r, default=2):
    """安全获取 effective_dim，至少为 1"""
    d = r['effective_dim']
    return max(1, int(d))


# ============================================================
# PCA 通道推导 —— ResNet / SCSKNet
# ============================================================

def _derive_sequential_5ch(pca_results, min_channels=2):
    """ResNet/SCSKNet 风格：取每 block 最后一个 conv 的有效维度。

    典型 ResNet Conv2d 顺序（过滤 shortcut 后）：
    conv1 + 4 blocks × 2 convs = 9 层
    → 取 [0, 2, 4, 6, 8] = [stem, b1_out, b2_out, b3_out, b4_out]
    """
    main = _filter_main_convs(pca_results)
    if not main:
        main = list(pca_results.items())

    dims = [max(min_channels, _safe_effective_dim(v)) for _, v in main]

    if len(dims) >= 9:
        channels = [dims[0], dims[2], dims[4], dims[6], dims[8]]
    elif len(dims) >= 5:
        step = max(1, len(dims) // 5)
        channels = [dims[i] for i in range(0, len(dims), step)][:5]
    else:
        channels = (dims + [dims[-1]] * (5 - len(dims)))[:5]

    return channels


# ============================================================
# PCA 通道推导 —— DenseNet
# ============================================================

def _derive_densenet_channels(pca_results, min_channels=2):
    """从 DenseNet PCA 结果推导 (init_channels, growth_rate, block_config)。

    策略：
    - init_channels: conv1 的有效维度
    - growth_rate: 所有 DenseLayer.conv2 有效维度的中位数
    - block_config: 保持剪枝版默认 (4, 6, 8, 6)
    """
    main = _filter_main_convs(pca_results)  # (key, result) 列表
    if not main:
        return {'init_channels': 8, 'growth_rate': 2,
                'block_config': (4, 6, 8, 6)}

    # conv1 是第一个（stem）
    init_channels = max(min_channels, _safe_effective_dim(main[0][1]))

    # DenseLayer conv2 产出 growth_rate 通道
    # conv1 (bottleneck) 的 effective_dim ≈ 4 * growth_rate
    # conv2 的 effective_dim ≈ growth_rate
    # DenseNet conv2 的名称通常包含 '.conv2'
    conv2_dims = []
    for key, r in main[1:]:
        if '.conv2' in key:
            conv2_dims.append(_safe_effective_dim(r))

    if conv2_dims:
        growth_rate = max(min_channels,
                          int(np.median(conv2_dims)))
    else:
        growth_rate = 2

    return {
        'init_channels': init_channels,
        'growth_rate': growth_rate,
        'block_config': (4, 6, 8, 6),
    }


# ============================================================
# PCA 通道推导 —— GoogleNet
# ============================================================

def _derive_googlenet_channels(pca_results, min_channels=2):
    """从 GoogleNet PCA 结果推导通道配置。

    GoogleNet Conv2d 顺序（在 named_modules 遍历中）：
    b1.0, b2.0, b2.2, [9 Inceptions × 6 convs 每组] = 3 + 54 = 57
    """
    main = _filter_main_convs(pca_results)
    if len(main) < 3 + 6:  # 至少要有 b1, b2 和 1 个 Inception
        return {
            'b1_out': 8, 'b2_mid': 8, 'b2_out': 16,
            'inception_configs': None,
        }

    b1_out = max(min_channels, _safe_effective_dim(main[0][1]))
    b2_mid = max(min_channels, _safe_effective_dim(main[1][1]))
    b2_out = max(min_channels, _safe_effective_dim(main[2][1]))

    # 剩余 54 个 conv，每 6 个一组对应 1 个 Inception
    inception_convs = main[3:]
    n_inceptions = len(inception_convs) // 6
    if n_inceptions > 9:
        n_inceptions = 9

    inception_configs = []
    for i in range(n_inceptions):
        group = inception_convs[i * 6:(i + 1) * 6]
        if len(group) < 6:
            break
        # 顺序: p1_1, p2_1, p2_2, p3_1, p3_2, p4_2
        c1 = max(min_channels, _safe_effective_dim(group[0][1]))
        c2_in = max(min_channels, _safe_effective_dim(group[1][1]))
        c2_out = max(min_channels, _safe_effective_dim(group[2][1]))
        c3_in = max(min_channels, _safe_effective_dim(group[3][1]))
        c3_out = max(min_channels, _safe_effective_dim(group[4][1]))
        c4 = max(min_channels, _safe_effective_dim(group[5][1]))
        inception_configs.append((c1, (c2_in, c2_out), (c3_in, c3_out), c4))

    return {
        'b1_out': b1_out,
        'b2_mid': b2_mid,
        'b2_out': b2_out,
        'inception_configs': inception_configs if len(inception_configs) == 9 else None,
    }


# ============================================================
# PCA 通道推导 —— ShuffleNetV2
# ============================================================

def _derive_shufflenet_channels(pca_results, min_channels=2):
    """从 ShuffleNetV2 PCA 结果推导 stages_out_channels。

    策略：
    - c0: conv1 有效维度
    - c1..c3: 每 stage 的 conv 有效维度取中位数 × 2（考虑 channel split）
    - conv5_out: conv5 有效维度
    """
    main = _filter_main_convs(pca_results)
    if len(main) < 4:
        return {'stages_out_channels': (8, 24, 48, 96), 'conv5_out': 32}

    # conv1 是第一个
    c0 = max(min_channels, _safe_effective_dim(main[0][1]))

    # group convs by stage using key patterns
    stage_dims = [[], [], []]  # 3 stages
    conv5_dims = []

    for key, r in main[1:]:
        ed = _safe_effective_dim(r)
        if 'conv5' in key:
            conv5_dims.append(ed)
        elif 'stages.0' in key:
            stage_dims[0].append(ed)
        elif 'stages.1' in key:
            stage_dims[1].append(ed)
        elif 'stages.2' in key:
            stage_dims[2].append(ed)

    def _stage_ch(dims):
        if not dims:
            return 24
        return max(min_channels, int(np.median(dims)))

    c1 = _stage_ch(stage_dims[0])
    c2 = _stage_ch(stage_dims[1])
    c3 = _stage_ch(stage_dims[2])
    # ShuffleNet 要求每个 stage 的输出通道为偶数（channel split）
    c1 = max(min_channels * 2, c1 if c1 % 2 == 0 else c1 + 1)
    c2 = max(min_channels * 2, c2 if c2 % 2 == 0 else c2 + 1)
    c3 = max(min_channels * 2, c3 if c3 % 2 == 0 else c3 + 1)

    conv5_out = max(min_channels,
                    int(np.median(conv5_dims)) if conv5_dims else 32)

    return {
        'stages_out_channels': (c0, c1, c2, c3),
        'conv5_out': conv5_out,
    }


# ============================================================
# PCA 通道推导 —— MobileNetV1
# ============================================================

def _derive_mobilenetv1_channels(pca_results, min_channels=2):
    """从 MobileNetV1 PCA 结果推导每层输出通道。

    MobileNetV1 Conv2d 顺序（在 named_modules 遍历中）：
    model.0.0 (conv_bn), model.1.0 (dw), model.1.3 (pw),
    model.2.0 (dw), model.2.3 (pw), ...

    我们只关心通道定义层（conv_bn 和 pointwise），
    即每两个 conv 取一个（depthwise 的 in==out，不改变通道维度）。
    """
    main = _filter_main_convs(pca_results, 'downsample')
    if len(main) < 2:
        return {'channels': [int(c * 1 / 16) for c in
                             [32, 64, 128, 128, 256, 256, 512, 512, 512,
                              512, 512, 512, 1024, 1024]]}

    # 取所有 conv 的有效维度，每隔一个取（跳过 depthwise）
    all_dims = [max(min_channels, _safe_effective_dim(r)) for _, r in main]
    # MobileNetV1: 1 standard + 13 depthwise-separable = 27 convs
    # 通道定义层在偶数索引（0, 2, 4, ...）
    channel_dims = all_dims[::2]

    # 确保 14 个通道值
    if len(channel_dims) < 14:
        # 用最后一个补齐
        channel_dims = channel_dims + [channel_dims[-1]] * (14 - len(channel_dims))
    channel_dims = channel_dims[:14]

    return {'channels': channel_dims}


# ============================================================
# PCA 通道推导 —— LightNet
# ============================================================

def _derive_lightnet_channels(pca_results, min_channels=2):
    """从 LightNet PCA 结果推导每层输出通道。

    LightNet 结构（prune 版有 8 个残差块 + 1 initial = 9 个通道定义点）：
    initial_conv.0, layers.0.depthwise.0, layers.0.pointwise.0,
    layers.1.depthwise.0, layers.1.pointwise.0, ...

    通道定义层：initial_conv 和 pointwise convs（索引 0, 2, 4, 6, ...）
    """
    main = _filter_main_convs(pca_results, 'downsample', 'shortcut')
    if len(main) < 2:
        return {'channels': [int(c * 1 / 16) for c in
                             [32, 64, 128, 128, 256, 256, 512, 512, 1024]]}

    all_dims = [max(min_channels, _safe_effective_dim(r)) for _, r in main]

    # 如果来自大 LightNet（14 个通道定义点），下采样到剪枝版 9 个
    channel_dims = all_dims[::2]  # 取通道定义层

    # 目标 9 个通道（prune 版有 8 个 ResidualDW + 1 initial = 9）
    target_len = 9
    if len(channel_dims) > target_len:
        indices = np.linspace(0, len(channel_dims) - 1, target_len, dtype=int)
        channel_dims = [channel_dims[i] for i in indices]
    elif len(channel_dims) < target_len:
        channel_dims = channel_dims + [channel_dims[-1]] * (target_len - len(channel_dims))

    return {'channels': channel_dims}


# ============================================================
# PCA 通道推导 —— 统一入口
# ============================================================

def derive_channels_from_pca(pca_results, net_type: str = None,
                             min_channels: int = 2) -> dict:
    """从逐层 PCA 有效维度推导网络的动态通道配置。

    Args:
        pca_results: analyze_conv_redundancy 返回的 dict
        net_type: 网络类型字符串。None 时尝试自动检测。
        min_channels: 每层的最小通道数下限

    Returns:
        dict: 类型相关的通道配置，可直接传给 build_pruned_embedding_net
    """
    if not pca_results:
        raise ValueError("PCA 结果为空")

    if net_type is None:
        # 自动检测：根据 PCA key 模式判断架构
        net_type = _detect_architecture(pca_results)

    t = net_type.lower().replace('_', '').replace('-', '')

    if 'resnet' in t:
        channels = _derive_sequential_5ch(pca_results, min_channels)
        return {'channels': channels}

    if 'scsknet' in t:
        channels = _derive_sequential_5ch(pca_results, min_channels)
        return {'channels': channels}

    if 'densenet' in t:
        return _derive_densenet_channels(pca_results, min_channels)

    if 'googlenet' in t:
        return _derive_googlenet_channels(pca_results, min_channels)

    if 'shufflenet' in t:
        return _derive_shufflenet_channels(pca_results, min_channels)

    if 'mobilenetv1' in t:
        return _derive_mobilenetv1_channels(pca_results, min_channels)

    if 'mobilenetv2' in t:
        # MobileNetV2 使用 LightNet 风格的 pruned 架构
        return _derive_lightnet_channels(pca_results, min_channels)

    if 'lightnet' in t:
        return _derive_lightnet_channels(pca_results, min_channels)

    # fallback: 尝试顺序 5 通道推导
    channels = _derive_sequential_5ch(pca_results, min_channels)
    return {'channels': channels}


def _detect_architecture(pca_results):
    """根据 PCA key 模式自动检测大网络架构"""
    keys = list(pca_results.keys())
    joined = ' '.join(keys)

    if 'dense' in joined or 'features' in joined:
        if any('dense' in k.lower() for k in keys):
            return 'DenseNet'
    if 'inception' in joined or 'b3.' in joined or 'b4.' in joined:
        return 'GoogleNet'
    if 'shuffle' in joined or 'stages.' in joined:
        return 'ShuffleNet'
    if 'scsk' in joined or 'ca.' in joined:
        return 'SCSKNet'
    if 'light' in joined or 'depthwise' in joined:
        if 'residual' in joined.lower():
            return 'LightNet'
        return 'MobileNetV1'
    if 'layer' in joined and any('layer' in k for k in keys):
        return 'ResNet'

    return 'ResNet'
