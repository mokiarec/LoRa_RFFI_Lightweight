import torch
import torch.nn as nn
import torch.nn.functional as F


# 通道洗牌操作
def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # 重塑并转置通道维度
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # 展平回原始维度
    x = x.view(batch_size, -1, height, width)
    return x


# ShuffleNet V2 基本单元
class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        branch_features = out_channels // 2

        if self.stride > 1:
            # 空间下采样分支 (Branch 1)
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1, groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Identity()

        # 特征提取分支 (Branch 2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_features, branch_features, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,
                      groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            # Stride 1: 将输入切分为两半，一半直接跳连，一半进卷积
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # Stride 2: 两个分支并行处理，最后拼接（通道数翻倍）
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels, stages_repeats=(4, 8, 4), stages_out_channels=(24, 116, 232, 464)):
        super(ShuffleNetV2, self).__init__()

        # 1. 初始层
        input_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. 构建各个 Stage
        self.stages = nn.ModuleList()
        for i in range(len(stages_repeats)):
            num_repeat = stages_repeats[i]
            output_channels = stages_out_channels[i + 1]

            # 每个 Stage 的第一层负责下采样 (stride=2)
            layers = [ShuffleBlock(input_channels, output_channels, stride=2)]
            # 后续层保持维度 (stride=1)
            for _ in range(num_repeat - 1):
                layers.append(ShuffleBlock(output_channels, output_channels, stride=1))

            self.stages.append(nn.Sequential(*layers))
            input_channels = output_channels

        # 3. 输出层
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        for stage in self.stages:
            x = stage(x)
        x = self.conv5(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class ShuffleNetV2_prune(nn.Module):
    def __init__(self, in_channels, stages_repeats=(4, 8, 4), stages_out_channels=(8, 24, 48, 96)):
        super(ShuffleNetV2_prune, self).__init__()

        # 1. 初始层 - 根据PCA结果，初始通道设为8
        input_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. 构建各个 Stage - 根据PCA有效维度剪枝
        # Stage 0: PCA 99%维度~32-62 -> 设输出48
        # Stage 1: PCA 99%维度~28-62 -> 设输出48 (保持一致)
        # Stage 2: PCA 99%维度~9-45 -> 设输出96
        self.stages = nn.ModuleList()
        for i in range(len(stages_repeats)):
            num_repeat = stages_repeats[i]
            output_channels = stages_out_channels[i + 1]

            # 每个 Stage 的第一层负责下采样 (stride=2)
            layers = [ShuffleBlock(input_channels, output_channels, stride=2)]
            # 后续层保持维度 (stride=1)
            for _ in range(num_repeat - 1):
                layers.append(ShuffleBlock(output_channels, output_channels, stride=1))

            self.stages.append(nn.Sequential(*layers))
            input_channels = output_channels

        # 3. 输出层 - 根据PCA结果，conv5的99%维度只有6，设为32足够
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终输出维度设为8，与系统其他prune模型保持一致
        self.fc = nn.Linear(32, 8)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        for stage in self.stages:
            x = stage(x)
        x = self.conv5(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)