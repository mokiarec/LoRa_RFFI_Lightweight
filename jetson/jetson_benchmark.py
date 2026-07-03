#!/usr/bin/env python3
"""
Jetson 端到端推理基准测试（完全自包含，无需项目依赖）
========================================================
单文件脚本，拷贝到 Jetson 即可运行，测试 8 个 Teacher/Student 模型。

用法:
    python jetson_benchmark.py \
        --enrol-data ./data/enrol.h5 \
        --clf-data ./data/test.h5 \
        --enrol-dev 0 39 --enrol-pkt 0 200 \
        --clf-dev 0 39 --clf-pkt 200 400 \
        --num-runs 30 --warmup 5 \
        --device cuda

依赖（仅 pip 包）:
    pip install torch numpy scipy h5py scikit-learn tqdm

目录结构:
    ./
    ├── jetson_benchmark.py    # 本文件
    ├── test_models/         # 8 个 .pth 权重
    │   ├── ResNet_teacher.pth
    │   ├── ResNet_student.pth
    │   ├── SCSKNet_teacher.pth
    │   ├── SCSKNet_student.pth
    │   ├── DenseNet_teacher.pth
    │   ├── DenseNet_student.pth
    │   ├── ShuffleNet_teacher.pth
    │   └── ShuffleNet_student.pth
    └── data/
        ├── enrol.h5
        └── test.h5
"""

import argparse
import os
import sys
import time
from collections import Counter
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# ============================================================================
# Section 1: 枚举定义 (from core/__init__.py)
# ============================================================================

class PreprocessType(Enum):
    """预处理类型枚举"""
    IQ = ("IQ", 1)
    STFT = ("STFT", 1)
    WST = ("WST", 2)

    def __init__(self, name, in_channels):
        self._value_ = name
        self.in_channels = in_channels


class NetworkType(str, Enum):
    """网络类型枚举"""
    ResNet = "ResNet"
    ResNet_prune = "ResNet_prune"
    SCSKNet = "SCSKNet"
    SCSKNet_prune = "SCSKNet_prune"
    DenseNet = "DenseNet"
    DenseNet_prune = "DenseNet_prune"
    ShuffleNet = "ShuffleNet"
    ShuffleNet_prune = "ShuffleNet_prune"


# ============================================================================
# Section 2: 数据加载 (from utils/data_loader.py)
# ============================================================================

class LoadDataset:
    dataset_name = "data"
    labelset_name = "label"

    @classmethod
    def _convert_to_complex(cls, data):
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = data[:, :round(num_col / 2)] + 1j * data[:, round(num_col / 2):]
        return data_complex

    @classmethod
    def load_iq_samples(cls, file_path, dev_range, pkt_range=None):
        with h5py.File(file_path, "r") as f:
            label = f[cls.labelset_name][:]
            label = label.astype(int).T - 1
            devices = np.unique(label) if dev_range is None else np.asarray(dev_range)

            sample_index_list = []
            for dev in devices:
                device_indices = np.where(label == dev)[0]
                if pkt_range is not None:
                    valid_mask = pkt_range < len(device_indices)
                    valid_pkt = pkt_range[valid_mask]
                    selected_indices = device_indices[valid_pkt]
                else:
                    selected_indices = device_indices
                sample_index_list.extend(selected_indices.tolist())

            data = f[cls.dataset_name][sample_index_list]
            label = label[sample_index_list]
            data = cls._convert_to_complex(data)

            has_dev_range = np.unique(label)
            num_pkt = data.shape[0] // len(has_dev_range)
            print(f"  Dataset: devs={len(has_dev_range)}, pkts/dev={num_pkt}, total={data.shape[0]}")
        return data, label


# ============================================================================
# Section 3: 信号处理 (from utils/signal_trans.py, 仅 STFT)
# ============================================================================

def awgn(data, snr_range):
    """加性高斯白噪声"""
    pkt_num = data.shape[0]
    SNRdB = np.random.uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = np.sum(np.abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = np.sqrt(N0 / 2) * (np.random.standard_normal(len(s)) + 1j * np.random.standard_normal(len(s)))
        data[pktIdx] = s + n
    return data


class TimeFrequencyTransformer:

    @staticmethod
    def generate_stft_channel(data, generate_type, win_len=256, overlap=128):
        def _normalization(data):
            s_norm = np.zeros(data.shape, dtype=complex)
            for i in range(data.shape[0]):
                sig_amplitude = np.abs(data[i])
                rms = np.sqrt(np.mean(sig_amplitude ** 2))
                s_norm[i] = data[i] / rms
            return s_norm

        def _spec_crop(x):
            num_row = x.shape[0]
            x_cropped = x[round(num_row * 0.3): round(num_row * 0.7)]
            return x_cropped

        def _gen_single_channel_ind_spectrogram(sig, win_len=256, overlap=128):
            f, t, spec = signal.stft(
                sig, window="boxcar", nperseg=win_len, noverlap=overlap,
                nfft=win_len, return_onesided=False, padded=False, boundary=None,
            )
            spec = np.fft.fftshift(spec, axes=0)

            if generate_type == PreprocessType.STFT:
                chan_ind_spec = spec[:, 1:] / spec[:, :-1]
                chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)
            elif generate_type == PreprocessType.IQ:
                return spec
            return chan_ind_spec_amp

        data = _normalization(data)
        num_sample = data.shape[0]

        if generate_type == PreprocessType.STFT:
            num_row = int(256 * 0.4)
            num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
            data_channel_ind_spec = np.zeros([num_sample, 1, num_row, num_column])

            for i in tqdm(range(num_sample), desc="  STFT"):
                chan_ind_spec_amp = _gen_single_channel_ind_spectrogram(data[i], win_len, overlap)
                chan_ind_spec_amp = _spec_crop(chan_ind_spec_amp)
                data_channel_ind_spec[i, 0, :, :] = chan_ind_spec_amp

            return data_channel_ind_spec

        elif generate_type == PreprocessType.IQ:
            sample_output = _gen_single_channel_ind_spectrogram(data[0], win_len, overlap)
            sample_output = _spec_crop(sample_output)
            actual_row, actual_col = sample_output.shape
            data_channel_ind_spec = np.zeros([num_sample, 1, actual_row, actual_col], dtype=np.complex64)
            for i in tqdm(range(num_sample), desc="  STFT(IQ)"):
                spec_result = _gen_single_channel_ind_spectrogram(data[i], win_len, overlap)
                spec_result = _spec_crop(spec_result)
                data_channel_ind_spec[i, 0, :, :] = spec_result
            return data_channel_ind_spec


# ============================================================================
# Section 4: TripletLoss (from utils/TripletDataset.py)
# ============================================================================

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)


# ============================================================================
# Section 5: 网络架构 (from net/net_*.py)
# ============================================================================

# --- ResNet ---
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


class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(32, 32)
        self.layer2 = ResBlock(32, 32)
        self.layer3 = ResBlock(32, 64, first_layer=True)
        self.layer4 = ResBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)
        return x


class ResNet_prune(nn.Module):
    def __init__(self, in_channels):
        super(ResNet_prune, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(8, 4, first_layer=True)
        self.layer2 = ResBlock(4, 8, first_layer=True)
        self.layer3 = ResBlock(8, 12, first_layer=True)
        self.layer4 = ResBlock(12, 16, first_layer=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)
        return x


# --- SCSKNet ---
class SCSKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, first_layer=False):
        super(SCSKBlock, self).__init__()
        self.first_layer = first_layer
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2 if in_channels > 1 else 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2 if in_channels > 1 else 1, in_channels, 1),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if first_layer else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = x * self.ca(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += res
        return self.relu(out)


class SCSKNet(nn.Module):
    def __init__(self, in_channels):
        super(SCSKNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.layer1 = SCSKBlock(32, 32)
        self.layer2 = SCSKBlock(32, 32)
        self.layer3 = SCSKBlock(32, 64, first_layer=True)
        self.layer4 = SCSKBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)


class SCSKNet_prune(nn.Module):
    def __init__(self, in_channels):
        super(SCSKNet_prune, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=7, stride=2, padding=3)
        self.layer1 = SCSKBlock(4, 8, first_layer=True)
        self.layer2 = SCSKBlock(8, 8)
        self.layer3 = SCSKBlock(8, 8)
        self.layer4 = SCSKBlock(8, 8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.normalize(self.fc(x), p=2, dim=1)


# --- DenseNet ---
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


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
        num_features = 64
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i, num_layers in enumerate(block_config):
            for _ in range(num_layers):
                layers.append(DenseLayer(num_features, growth_rate))
                num_features += growth_rate
            if i != len(block_config) - 1:
                out_features = num_features // 2
                layers.append(Transition(num_features, out_features))
                num_features = out_features

        self.features = nn.Sequential(*layers)
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
        num_features = 8
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i, num_layers in enumerate(block_config):
            for _ in range(num_layers):
                layers.append(DenseLayer(num_features, growth_rate))
                num_features += growth_rate
            if i != len(block_config) - 1:
                out_features = num_features // 2
                layers.append(Transition(num_features, out_features))
                num_features = out_features

        self.features = nn.Sequential(*layers)
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


# --- ShuffleNet ---
def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=self.stride,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_features, branch_features,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride,
                      padding=1, groups=branch_features, bias=False),
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


class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels, stages_repeats=(4, 8, 4),
                 stages_out_channels=(24, 116, 232, 464)):
        super(ShuffleNetV2, self).__init__()
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
            layers = [ShuffleBlock(input_channels, output_channels, stride=2)]
            for _ in range(num_repeat - 1):
                layers.append(ShuffleBlock(output_channels, output_channels, stride=1))
            self.stages.append(nn.Sequential(*layers))
            input_channels = output_channels

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
    def __init__(self, in_channels, stages_repeats=(4, 8, 4),
                 stages_out_channels=(8, 24, 48, 96)):
        super(ShuffleNetV2_prune, self).__init__()
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
            layers = [ShuffleBlock(input_channels, output_channels, stride=2)]
            for _ in range(num_repeat - 1):
                layers.append(ShuffleBlock(output_channels, output_channels, stride=1))
            self.stages.append(nn.Sequential(*layers))
            input_channels = output_channels

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 8)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        for stage in self.stages:
            x = stage(x)
        x = self.conv5(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ============================================================================
# Section 6: TripletNet + 网络工厂 (from net/__init__.py, 精简)
# ============================================================================

_NET_REGISTRY = {
    "ResNet": ResNet,
    "ResNet_prune": ResNet_prune,
    "SCSKNet": SCSKNet,
    "SCSKNet_prune": SCSKNet_prune,
    "DenseNet": DenseNet,
    "DenseNet_prune": DenseNet_prune,
    "ShuffleNet": ShuffleNetV2,
    "ShuffleNet_prune": ShuffleNetV2_prune,
}


class TripletNet(nn.Module):
    """嵌入网络包装器 — 仅用于推理（predict），不用于训练"""

    def __init__(self, net_type, in_channels, margin=0.1):
        super(TripletNet, self).__init__()
        self.margin = margin
        net_cls = _NET_REGISTRY.get(net_type.value if isinstance(net_type, NetworkType) else net_type)
        if net_cls is None:
            raise ValueError(f"Unknown network: {net_type}")
        self.embedding_net = net_cls(in_channels=in_channels)

    def predict(self, x):
        """推理：输入 (N,C,H,W) tensor → 输出 (N,D) 嵌入向量"""
        with torch.no_grad():
            return self.embedding_net(x)


# ============================================================================
# Section 7: 辅助函数
# ============================================================================

def load_and_preprocess(file_path, dev_range, pkt_range):
    """加载 IQ → STFT → Tensor"""
    t0 = time.perf_counter()
    data, label = LoadDataset.load_iq_samples(file_path, dev_range, pkt_range)
    data = TimeFrequencyTransformer.generate_stft_channel(data, PreprocessType.STFT)
    data = torch.tensor(data).float()
    print(f"  [preprocess] done in {time.perf_counter() - t0:.2f}s, shape={data.shape}")
    return label, data


def load_model(model_path, net_type_str):
    """加载 TripletNet 权重"""
    net_type = NetworkType(net_type_str)
    model = TripletNet(net_type=net_type, in_channels=PreprocessType.STFT.in_channels)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def apply_voting(labels, vote_size):
    """滑动窗口多数投票"""
    voted = []
    for i in range(len(labels)):
        start = max(0, i - vote_size // 2)
        end = min(len(labels), i + vote_size // 2 + 1)
        window = labels[start:end]
        most_common = Counter(window).most_common(1)[0][0]
        voted.append(most_common)
    return voted


# ============================================================================
# Section 8: 8 个模型定义
# ============================================================================

MODEL_DEFS = [
    ("ResNet Teacher",     "ResNet_teacher.pth",     "ResNet",         "Teacher"),
    ("ResNet Student",     "ResNet_student.pth",     "ResNet_prune",   "Student"),
    ("SCSKNet Teacher",    "SCSKNet_teacher.pth",    "SCSKNet",        "Teacher"),
    ("SCSKNet Student",    "SCSKNet_student.pth",    "SCSKNet_prune",  "Student"),
    ("DenseNet Teacher",   "DenseNet_teacher.pth",   "DenseNet",       "Teacher"),
    ("DenseNet Student",   "DenseNet_student.pth",   "DenseNet_prune", "Student"),
    ("ShuffleNet Teacher", "ShuffleNet_teacher.pth", "ShuffleNet",     "Teacher"),
    ("ShuffleNet Student", "ShuffleNet_student.pth", "ShuffleNet_prune","Student"),
]


# ============================================================================
# Section 9: 单模型基准测试
# ============================================================================

def benchmark_one_model(model, device, data_enrol, label_enrol, data_clf, label_clf,
                        pca_dim, vote_size, num_runs, warmup):
    model.to(device)
    model.eval()

    enrol_tensor = data_enrol.to(device)
    clf_tensor = data_clf.to(device)
    num_enrol = data_enrol.shape[0]
    num_clf = data_clf.shape[0]

    # 预热
    if warmup > 0:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.predict(enrol_tensor)
                _ = model.predict(clf_tensor)

    fe_enrol_times, fe_clf_times, clf_times, total_times, accuracies = [], [], [], [], []

    for _ in range(num_runs):
        # Enrol 特征提取
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fe_enrol = model.predict(enrol_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fe_enrol = time.perf_counter() - t0

        # Clf 特征提取
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fe_clf = model.predict(clf_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fe_clf = time.perf_counter() - t0

        # 分类 (PCA + KNN W/V)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        fe_enrol_np = fe_enrol.cpu().numpy()
        fe_clf_np = fe_clf.cpu().numpy()

        if pca_dim > 0 and fe_enrol_np.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim)
            pca.fit(fe_enrol_np)
            train_data = pca.transform(fe_enrol_np)
            test_data = pca.transform(fe_clf_np)
        else:
            train_data = fe_enrol_np
            test_data = fe_clf_np

        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(train_data, label_enrol.ravel())
        pred = knn.predict(test_data)

        if vote_size > 1:
            pred = apply_voting(pred, vote_size)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_clf = time.perf_counter() - t0

        fe_enrol_times.append(t_fe_enrol)
        fe_clf_times.append(t_fe_clf)
        clf_times.append(t_clf)
        # Total = FE Clf + Clf（enrol 是离线操作）
        total_times.append(t_fe_clf + t_clf)
        accuracies.append(accuracy_score(label_clf.ravel(), pred))

    def stat(arr):
        return np.mean(arr), np.std(arr)

    return {
        "num_enrol": num_enrol,
        "num_clf": num_clf,
        "fe_enrol_mean": stat(fe_enrol_times)[0],
        "fe_enrol_std":  stat(fe_enrol_times)[1],
        "fe_clf_mean":  stat(fe_clf_times)[0],
        "fe_clf_std":   stat(fe_clf_times)[1],
        "clf_mean":     stat(clf_times)[0],
        "clf_std":      stat(clf_times)[1],
        "total_mean":   stat(total_times)[0],
        "total_std":    stat(total_times)[1],
        "accuracy":     stat(accuracies)[0],
        "accuracy_std": stat(accuracies)[1],
    }


# ============================================================================
# Section 10: 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Jetson RFFI Batch Benchmark (self-contained)")

    parser.add_argument("--model-dir", type=str, default="./jetson/test_models")
    parser.add_argument("--enrol-data", type=str, required=True)
    parser.add_argument("--clf-data", type=str, required=True)
    parser.add_argument("--enrol-dev", type=int, nargs=2, default=[30, 34])
    parser.add_argument("--enrol-pkt", type=int, nargs=2, default=[0, 10])
    parser.add_argument("--clf-dev", type=int, nargs=2, default=[30, 34])
    parser.add_argument("--clf-pkt", type=int, nargs=2, default=[10, 20])
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--vote-size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"Runs: {args.num_runs}  |  Warmup: {args.warmup}")
    print(f"PCA dim: {args.pca_dim}  |  Vote size: {args.vote_size}")
    print(f"{'='*60}")

    # ---- 加载数据（只做一次） ----
    enrol_dev = np.arange(args.enrol_dev[0], args.enrol_dev[1] + 1)
    enrol_pkt = np.arange(args.enrol_pkt[0], args.enrol_pkt[1])
    clf_dev = np.arange(args.clf_dev[0], args.clf_dev[1] + 1)
    clf_pkt = np.arange(args.clf_pkt[0], args.clf_pkt[1])

    print("\n[Data] Loading enrollment set...")
    label_enrol, data_enrol = load_and_preprocess(args.enrol_data, enrol_dev, enrol_pkt)

    print("[Data] Loading classification set...")
    label_clf, data_clf = load_and_preprocess(args.clf_data, clf_dev, clf_pkt)

    # ---- 逐个测试模型 ----
    all_results = []

    for display_name, weight_file, net_type_str, role in MODEL_DEFS:
        model_path = os.path.join(args.model_dir, weight_file)
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"[{display_name}]  net={net_type_str}  role={role}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        model = load_model(model_path, net_type_str)
        print(f"  Model loaded in {time.perf_counter() - t0:.2f}s")

        result = benchmark_one_model(
            model, device, data_enrol, label_enrol, data_clf, label_clf,
            pca_dim=args.pca_dim, vote_size=args.vote_size,
            num_runs=args.num_runs, warmup=args.warmup,
        )
        result["name"] = display_name
        all_results.append(result)

        # 单模型输出（per-packet）
        pp_enrol = result['fe_enrol_mean'] / result['num_enrol'] * 1000
        pp_fe   = result['fe_clf_mean'] / result['num_clf'] * 1000
        pp_cls  = result['clf_mean'] / result['num_clf'] * 1000
        pp_total = result['total_mean'] / result['num_clf'] * 1000
        print(f"  FE Enrol:  {result['fe_enrol_mean']*1000:.2f} ms  ({pp_enrol:.4f} ms/pkt)")
        print(f"  FE Clf:    {result['fe_clf_mean']*1000:.2f} ms  ({pp_fe:.4f} ms/pkt)")
        print(f"  Clf:       {result['clf_mean']*1000:.2f} ms  ({pp_cls:.4f} ms/pkt)")
        print(f"  TOTAL:     {result['total_mean']*1000:.2f} ms  ({pp_total:.4f} ms/pkt)")
        print(f"  Accuracy:  {result['accuracy']*100:.2f}%")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 汇总表 ----
    if not all_results:
        print("\nNo models tested.")
        return

    print(f"\n\n{'='*95}")
    print("BATCH BENCHMARK SUMMARY  (ms/pkt)")
    print(f"{'='*95}")
    print(f"{'Model':<22} {'FE Clf':>8} {'Clf':>8} {'Total':>8} {'Acc':>7} {'Size':>8}")
    print("-" * 95)

    for r in all_results:
        for m in MODEL_DEFS:
            if m[0] == r['name']:
                fpath = os.path.join(args.model_dir, m[1])
                size_kb = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
                break

        pp_fe   = r['fe_clf_mean'] / r['num_clf'] * 1000
        pp_cls  = r['clf_mean'] / r['num_clf'] * 1000
        pp_total = r['total_mean'] / r['num_clf'] * 1000

        print(f"{r['name']:<22} {pp_fe:7.4f} {pp_cls:7.4f} {pp_total:7.4f} "
              f"{r['accuracy']*100:6.2f}% {size_kb:7.0f}K")

    print("-" * 95)

    # ---- Teacher vs Student ----
    print(f"\n{'='*95}")
    print("TEACHER vs STUDENT COMPARISON  (Total = FE Clf + Clf, ms/pkt)")
    print(f"{'='*95}")
    print(f"{'Pair':<22} {'T_Total':>8} {'S_Total':>8} {'Speedup':>8} "
          f"{'T_Acc':>7} {'S_Acc':>7} {'ΔAcc':>7} {'T_Size':>7} {'S_Size':>7} {'Compress':>9}")
    print("-" * 95)

    pairs = [
        ("ResNet",    "ResNet Teacher",    "ResNet Student"),
        ("SCSKNet",   "SCSKNet Teacher",   "SCSKNet Student"),
        ("DenseNet",  "DenseNet Teacher",  "DenseNet Student"),
        ("ShuffleNet","ShuffleNet Teacher","ShuffleNet Student"),
    ]
    results_by_name = {r['name']: r for r in all_results}
    name_to_file = dict((m[0], m[1]) for m in MODEL_DEFS)

    for pair_name, t_name, s_name in pairs:
        t = results_by_name.get(t_name)
        s = results_by_name.get(s_name)
        if not t or not s:
            continue

        t_pp = t['total_mean'] / t['num_clf'] * 1000
        s_pp = s['total_mean'] / s['num_clf'] * 1000
        speedup = t_pp / s_pp if s_pp > 0 else 0
        delta_acc = s['accuracy'] - t['accuracy']

        t_size = os.path.getsize(os.path.join(args.model_dir, name_to_file[t_name])) / 1024
        s_size = os.path.getsize(os.path.join(args.model_dir, name_to_file[s_name])) / 1024
        compress = t_size / s_size if s_size > 0 else 0

        print(f"{pair_name:<22} {t_pp:7.4f} {s_pp:7.4f} {speedup:7.2f}x "
              f"{t['accuracy']*100:6.2f}% {s['accuracy']*100:6.2f}% "
              f"{delta_acc*100:+6.2f}% {t_size:6.0f}K {s_size:6.0f}K {compress:8.1f}x")

    print("-" * 95)
    print()


if __name__ == "__main__":
    main()
