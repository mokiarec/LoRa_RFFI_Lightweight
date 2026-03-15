# TripletNet类，用于创建三元组网络
import torch
import torch.nn as nn

from net.net_MobileNet import mobilenet
from net.net_DRSN import drsnet18
from net.net_ResNet import FeatureExtractor
from training_utils.TripletDataset import TripletLoss

from core.config import NetworkType


class TripletNet(nn.Module):
    def __init__(self, net_type, in_channels,
                 margin=0.1,
                 width_multiplier=1 / 16,
        ):
        super(TripletNet, self).__init__()
        self.margin = margin
        if net_type == NetworkType.ResNet:
            self.embedding_net = FeatureExtractor(in_channels=in_channels)

        elif net_type == NetworkType.DRSN:
            self.embedding_net = drsnet18(in_channels=in_channels)

        elif net_type == NetworkType.MobileNetV1:  # 添加 MobileNetV1 支持
            self.embedding_net = mobilenet(version='v1', in_channels=in_channels, width_multiplier=width_multiplier)

        elif net_type == NetworkType.MobileNetV2:  # 添加 MobileNetV2 支持
            self.embedding_net = mobilenet(version='v2', in_channels=in_channels, width_multiplier=width_multiplier)

        elif net_type == NetworkType.LightNet:
            self.embedding_net = mobilenet(version='light', in_channels=in_channels, width_multiplier=width_multiplier)

        # 其他网络类型可以继续添加...

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding_net(anchor)
        embedded_positive = self.embedding_net(positive)
        embedded_negative = self.embedding_net(negative)
        return embedded_anchor, embedded_positive, embedded_negative

    def triplet_loss(self, anchor, positive, negative):
        loss_fn = TripletLoss(margin=self.margin)
        return loss_fn(anchor, positive, negative)

    def predict(self, anchor):
        with torch.no_grad():
            return self.embedding_net(anchor)
