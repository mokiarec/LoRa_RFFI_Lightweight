# TripletNet类，用于创建三元组网络
import torch
import torch.nn as nn

from net import Network, NetworkType
from training_utils.TripletDataset import TripletLoss


class TripletNet(nn.Module):
    def __init__(self, net_type: NetworkType, in_channels,
                 margin=0.1,
                 width_multiplier=1 / 16,
        ):
        super(TripletNet, self).__init__()
        self.margin = margin
        
        # 使用工厂模式创建嵌入网络
        self.embedding_net = Network.create(
            net_type, 
            in_channels=in_channels,
            width_multiplier=width_multiplier
        )

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
