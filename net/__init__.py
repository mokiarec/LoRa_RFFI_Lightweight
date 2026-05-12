# 定义网络类型枚举
from enum import Enum
from typing import Dict, Callable


class NetworkType(str, Enum):
    """网络类型枚举"""
    ResNet= "ResNet"               # 残差网络
    ResNet_prune = "ResNet_prune"
    Drsn = "Drsn"                   # 深度残差网路
    MobileNetV1 = "MobileNetV1"     # MobileNetV1网络
    MobileNetV2 = "MobileNetV2"     # MobileNetV2网络
    LightNet = "LightNet"           # MobileNetV1改进网络
    SCSKNet = "SCSKNet"             # 卷积小波小卷积网络
    SCSKNet_prune = "SCSKNet_prune"
    DenseNet = "DenseNet"
    DenseNet_prune = "DenseNet_prune"
    ShuffleNet = "ShuffleNet"
    ShuffleNet_prune = "ShuffleNet_prune"
    GoogleNet = "GoogleNet"
    GoogleNet_prune = "GoogleNet_prune"


# TripletNet类，用于创建三元组网络
import torch
import torch.nn as nn

from utils.TripletDataset import TripletLoss


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


# 网络工厂类
class Network:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, network_name: str):
        """装饰器：注册网络创建函数"""

        def decorator(func: Callable):
            if network_name in cls._registry:
                print(f"警告: 网络类型 '{network_name}' 已被注册，将被覆盖")
            # 注册创建函数
            cls._registry[network_name] = func
            return func

        return decorator

    @classmethod
    def create(cls, network_type: NetworkType, in_channels, **kwargs) -> nn.Module:
        """创建网络实例"""
        network_name = network_type.value

        if network_name not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"未知的网络类型: {network_name}\n"
                f"可用的网络类型: {available_types}"
            )

        creator_func = cls._registry[network_name]
        return creator_func(in_channels, **kwargs)

    @classmethod
    def get_available_types(cls) -> list:
        """获取所有已注册的网络类型"""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, network_type: str) -> bool:
        """检查网络类型是否已注册"""
        network_name = str(network_type)
        return network_name in cls._registry


# ==================== 网络创建函数注册 ====================

@Network.register(NetworkType.ResNet.value)
def create_resnet(in_channels: int, **kwargs):
    """创建 ResNet 网络"""
    from net.net_ResNet import ResNet
    return ResNet(in_channels=in_channels)


@Network.register(NetworkType.ResNet_prune.value)
def create_resnet_prune(in_channels: int, **kwargs):
    """剪枝版 ResNet 网络"""
    from net.net_ResNet import ResNet_prune
    return ResNet_prune(in_channels=in_channels)


@Network.register(NetworkType.Drsn.value)
def create_drsn(in_channels: int, **kwargs):
    """创建 DRSN 网络"""
    from net.net_DRSN import drsnet18
    return drsnet18(in_channels=in_channels)


@Network.register(NetworkType.MobileNetV1.value)
def create_mobilenet_v1(in_channels: int, width_multiplier: float = 1 / 16, **kwargs):
    """创建 MobileNetV1 网络"""
    from net.net_MobileNet import MobileNet
    return MobileNet(version='v1', in_channels=in_channels, width_multiplier=width_multiplier)


@Network.register(NetworkType.MobileNetV2.value)
def create_mobilenet_v2(in_channels: int, width_multiplier: float = 1 / 16, **kwargs):
    """创建 MobileNetV2 网络"""
    from net.net_MobileNet import MobileNet
    return MobileNet(version='v2', in_channels=in_channels, width_multiplier=width_multiplier)


@Network.register(NetworkType.LightNet.value)
def create_lightnet(in_channels: int, width_multiplier: float = 1 / 16, **kwargs):
    """创建 LightNet 网络"""
    from net.net_MobileNet import MobileNet
    return MobileNet(version='light', in_channels=in_channels, width_multiplier=width_multiplier)


# 注意：NetworkType 枚举中未定义 LightNet_prune 和 LightNet_opt，此处保留字符串注册或需补充枚举
@Network.register("LightNet_prune")
def create_lightnet_prune(in_channels: int, width_multiplier: float = 1 / 16, **kwargs):
    """创建剪枝版 LightNet 网络"""
    from net.net_MobileNet import MobileNet
    return MobileNet(version='light_prune', in_channels=in_channels, width_multiplier=width_multiplier)


@Network.register("LightNet_opt")
def create_lightnet_opt(in_channels: int, width_multiplier: float = 1 / 16, **kwargs):
    """创建优化版 LightNet 网络"""
    from net.net_MobileNet import MobileNet
    return MobileNet(version='light_opt', in_channels=in_channels, width_multiplier=width_multiplier)


@Network.register(NetworkType.SCSKNet.value)
def create_scsknet(in_channels: int, **kwargs):
    """创建 SCSKNet 网络"""
    from net.net_SCSKNet import SCSKNet
    return SCSKNet(in_channels=in_channels)


@Network.register(NetworkType.SCSKNet_prune.value)
def create_scsknet_prune(in_channels: int, **kwargs):
    """创建剪枝版 SCSKNet 网络"""
    from net.net_SCSKNet import SCSKNet_prune
    return SCSKNet_prune(in_channels=in_channels)


@Network.register(NetworkType.DenseNet.value)
def create_densenet(in_channels: int, **kwargs):
    """创建 DenseNet 网络"""
    from net.net_DenseNet import DenseNet
    return DenseNet(in_channels=in_channels)


@Network.register(NetworkType.DenseNet_prune.value)
def create_densenet_prune(in_channels: int, **kwargs):
    """创建剪枝版 DenseNet 网络"""
    from net.net_DenseNet import DenseNet_prune
    return DenseNet_prune(in_channels=in_channels)


@Network.register(NetworkType.ShuffleNet.value)
def create_shufflenet(in_channels: int, **kwargs):
    """创建 ShuffleNet 网络"""
    from net.net_ShuffleNet import ShuffleNetV2
    return ShuffleNetV2(in_channels=in_channels)


@Network.register(NetworkType.ShuffleNet_prune.value)
def create_shufflenet_prune(in_channels: int, **kwargs):
    """创建剪枝版 ShuffleNet 网络"""
    from net.net_ShuffleNet import ShuffleNetV2_prune
    return ShuffleNetV2_prune(in_channels=in_channels)

@Network.register(NetworkType.GoogleNet.value)
def create_googlenet(in_channels: int, **kwargs):
    """创建 GoogleNet 网络"""
    from net.net_GoogleNet import GoogleNet
    return GoogleNet(in_channels=in_channels)


@Network.register(NetworkType.GoogleNet_prune.value)
def create_googlenet_prune(in_channels: int, **kwargs):
    """创建剪枝版 GoogleNet 网络"""
    from net.net_GoogleNet import GoogleNet_prune
    return GoogleNet_prune(in_channels=in_channels)