import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 Inception 核心模块
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路 1：单 1x1 卷积分支
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 线路 2：1x1 卷积降维 + 3x3 卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路 3：1x1 卷积降维 + 5x5 卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路 4：3x3 最大池化 + 1x1 卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上拼接所有分支的特征图
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogleNet(nn.Module):
    def __init__(self, in_channels):
        super(GoogleNet, self).__init__()
        # Block 1 & 2: 基础卷积与下采样
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 3: Inception 3a, 3b
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 4: Inception 4a, 4b, 4c, 4d, 4e
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 5: Inception 5a, 5b
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class GoogleNet_prune(nn.Module):
    def __init__(self, in_channels):
        super(GoogleNet_prune, self).__init__()
        # Block 1 & 2: 基础卷积与下采样 - 根据PCA结果大幅减少通道数
        # b1.0: Out 64 -> 95%: 2, 使用较小的通道数
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # b2.0: Out 64 -> 95%: 5; b2.2: Out 192 -> 95%: 1
        self.b2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 3: Inception 3a, 3b - 根据PCA结果调整各分支通道数
        # b3.0: in=16, p1_1(64->4), p2_1(96->5), p2_2(128->2), p3_1(16->2), p3_2(32->2), p4_2(32->9)
        # 输出 = 4 + 8 + 4 + 8 = 24
        self.b3 = nn.Sequential(
            Inception(16, 4, (5, 8), (2, 4), 8),
            # b3.1: in=24, p1_1(128->2), p2_1(128->2), p2_2(192->1), p3_1(32->2), p3_2(96->1), p4_2(64->2)
            # 输出 = 2 + 4 + 2 + 4 = 12
            Inception(24, 2, (2, 4), (2, 2), 4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 4: Inception 4a-4e - PCA显示大部分层95%维度为1，大幅剪枝
        # b4.0: in=12, 输出需要保持一致性
        # 输出 = 4 + 8 + 4 + 4 = 20
        self.b4 = nn.Sequential(
            Inception(12, 4, (4, 8), (2, 4), 4),
            # b4.1-4.4: in=20
            # 输出 = 4 + 8 + 4 + 4 = 20
            Inception(20, 4, (4, 8), (2, 4), 4),
            Inception(20, 4, (4, 8), (2, 4), 4),
            Inception(20, 4, (4, 8), (2, 4), 4),
            Inception(20, 4, (4, 8), (2, 4), 4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 5: Inception 5a, 5b - 同样根据PCA结果剪枝
        # b5.0-b5.1: in=20
        # 输出 = 4 + 8 + 4 + 4 = 20
        self.b5 = nn.Sequential(
            Inception(20, 4, (4, 8), (2, 4), 4),
            Inception(20, 4, (4, 8), (2, 4), 4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        # 最终输出维度根据PCA结果调整，fc输入为最后一个Inception的输出通道数
        self.fc = nn.Linear(20, 8)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)