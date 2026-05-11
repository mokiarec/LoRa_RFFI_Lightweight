# 模型加载函数
import torch

from net import TripletNet


def load_model(file_path, net_type, generate_type, weights_only=True):
    """从指定路径加载模型"""
    model = TripletNet(net_type=net_type, in_channels=generate_type.in_channels)
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    model.eval()
    # print(f"Model loaded from {file_path}")
    return model