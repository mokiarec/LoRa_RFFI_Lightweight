# 数据预处理函数
import os
import time

import h5py
import numpy as np
import torch

from core.config import PreprocessType
from net.TripletNet import TripletNet
from training_utils.data_loader import LoadDataset
from utils.signal_trans import TimeFrequencyTransformer, awgn


def generate_spectrogram(data, generate_type, wst_j=6, wst_q=6):
    """
    数据预处理方式选择

    - PREPROCESS_TYPE : 0 信道独立的STFT
    - PREPROCESS_TYPE : 1 小波散射变换
    """
    # TF_Transformer = TimeFrequencyTransformer()
    if generate_type == PreprocessType.STFT:
        data = TimeFrequencyTransformer.generate_stft_channel(data)
    if generate_type == PreprocessType.WST:
        data = TimeFrequencyTransformer.generate_WST(data, J=wst_j, Q=wst_q)
    return data


def load_data(file_path, dev_range, pkt_range) -> tuple[np.ndarray, np.ndarray]:
    """根据参数加载指定数据"""
    # LoadDatasetObj = LoadDataset()
    data, label = LoadDataset.load_iq_samples(file_path, dev_range, pkt_range)

    return data, label


# 数据预处理
def load_generate(file_path, dev_range, pkt_range, generate_type, snr_range=None):
    """加载 & 预处理"""
    data, label = load_data(file_path, dev_range, pkt_range)
    # 数据加入人工白噪声
    if snr_range is not None:
        data = awgn(data, snr_range)
    data = generate_spectrogram(data, generate_type)

    # 数据三元组化并转换为张量类型
    data = torch.tensor(data).float()

    return label, data


# 数据预处理
def load_generate_triplet(file_path, dev_range, pkt_range, generate_type, snr_range=None):
    """加载 & 预处理 & data三元组化"""
    data, label = load_data(file_path, dev_range, pkt_range)
    # 数据加入人工白噪声
    if snr_range is not None:
        data = awgn(data, snr_range)
    data = generate_spectrogram(data, generate_type)

    # 数据三元组化并转换为张量类型
    triplet_data = [data, data, data]
    triplet_data = [torch.tensor(x).float() for x in triplet_data]

    return label, triplet_data

def prepare_train_data(
        new_file_flag,
        filename_train_prepared_data,
        path_train_data,
        dev_range=None,
        pkt_range=None,
        snr_range=None,
        generate_type=None,
        WST_J=None,
        WST_Q=None,
):
    """
    准备训练集

    根据选择的信号处理类型生成对应的训练集
    """
    time_prepare_start = time.time()
    # 数据预处理
    if not os.path.exists(filename_train_prepared_data) or new_file_flag == True:
        # 需要新处理数据
        print("Data Converting...")

        data, labels = load_data(path_train_data, dev_range, pkt_range)
        if generate_type != PreprocessType.IQ and snr_range is not None:
            data = awgn(data, snr_range)

        if generate_type != PreprocessType.IQ:
            data = generate_spectrogram(data, generate_type, WST_J, WST_Q)

        with h5py.File(filename_train_prepared_data, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)
        timeCost = time.time() - time_prepare_start
        print(f"Convert Time Cost: {timeCost:.3f}s")
    else:
        # 处理数据已存在
        print("Data exist, loading...")

        with h5py.File(filename_train_prepared_data, "r") as f:
            data = f["data"][:]
            labels = f["labels"][:]

        timeCost = time.time() - time_prepare_start
        print(f"Load Time Cost: {timeCost:.3f}s")
    return data, labels

def load_model(file_path, net_type, generate_type=None, weights_only=True):
    """从指定路径加载模型"""
    model = TripletNet(net_type=net_type, in_channels=generate_type.in_channels)
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model