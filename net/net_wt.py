import torch.nn as nn
import torch.nn.functional as F

from utils.TripletDataset import TripletLoss
from utils.WTConv import DepthwiseSeparableConvWithWTConv2d

NET_TYPE = "net_wt_1"


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
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.WtConv1 = DepthwiseSeparableConvWithWTConv2d(1, 32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(32, 32)
        self.layer2 = ResBlock(32, 32)
        self.layer3 = ResBlock(32, 64, first_layer=True)
        self.layer4 = ResBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = F.relu(self.WtConv1(x))
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x


# TripletNet类，用于创建三元组网络
class TripletNet(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletNet, self).__init__()
        self.margin = margin
        self.embedding_net = FeatureExtractor()

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding_net(anchor)
        embedded_positive = self.embedding_net(positive)
        embedded_negative = self.embedding_net(negative)
        return embedded_anchor, embedded_positive, embedded_negative

    def triplet_loss(self, anchor, positive, negative):
        loss_fn = TripletLoss(margin=self.margin)
        return loss_fn(anchor, positive, negative)
