"""逐层 Conv2d PCA 分析 —— 诊断每个卷积层的通道冗余度。

对训练好的大网络，hook 每一个 Conv2d 层，收集输出特征图，
在通道维度上运行 PCA，报告 95% 方差对应的有效维度。
"""

import sys
from pathlib import Path

# 确保父项目可导入（独立运行时也能找到 core/net 等模块）
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from core.config import DEVICE


def _hook_fn(name, storage, max_samples):
    """返回一个 forward hook，用于捕获卷积层输出特征"""

    def fn(module, _input, output):
        B, C, H, W = output.shape
        # 将 (B,C,H,W) 重整为 (B*H*W, C)：每个空间位置作为一个样本
        flat = output.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()
        if max_samples and flat.shape[0] > max_samples:
            idx = np.random.choice(flat.shape[0], max_samples, replace=False)
            flat = flat[idx]
        storage[name].append(flat)

    return fn


def analyze_conv_redundancy(model, dataloader, device=None, threshold=0.95,
                            max_spatial_samples=10000, max_batches=None):
    """对每个 Conv2d 层的输出运行 PCA，找到有效通道维度。

    Args:
        model: 训练好的 TripletNet（或任何 nn.Module）
        dataloader: 返回 (anchor, positive, negative) 三元组的 DataLoader
        device: torch 设备（默认使用 config 中的 DEVICE）
        threshold: 累积解释方差阈值，用于确定有效维度
        max_spatial_samples: 每层最多采样的空间位置数（防止内存溢出）
        max_batches: 最多处理的 batch 数（None = 全部）

    Returns:
        dict: 层名 -> {
            'original_dim': 原始通道数,
            'effective_dim': 达到 threshold 方差所需的主成分数,
            'cumsum': 累积方差比例,
            'explained_variance_ratio': 各主成分的方差比例,
        }
    """
    if device is None:
        device = DEVICE

    # 注册 hook
    layer_outputs = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_outputs[name] = []
            hooks.append(module.register_forward_hook(
                _hook_fn(name, layer_outputs, max_spatial_samples)))

    # 收集特征
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            anchor = anchor.to(device)
            _ = model.embedding_net(anchor)

    # 移除 hook
    for h in hooks:
        h.remove()

    # 逐层 PCA
    results = {}
    for name, outputs in layer_outputs.items():
        if not outputs:
            continue
        all_out = np.concatenate(outputs, axis=0)
        original_dim = all_out.shape[1]

        pca = PCA()
        pca.fit(all_out)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = int(np.searchsorted(cumsum, threshold) + 1)
        effective_dim = max(1, min(effective_dim, original_dim))

        results[name] = {
            'original_dim': original_dim,
            'effective_dim': effective_dim,
            'cumsum': cumsum,
            'explained_variance_ratio': pca.explained_variance_ratio_,
        }

    return results


def print_analysis(results):
    """格式化打印逐层 PCA 分析结果"""
    print("\n" + "=" * 72)
    print(f"{'层名称':<35} {'原始通道':>8} {'有效维度(95%)':>14} {'比例':>8}")
    print("-" * 72)
    total_orig = 0
    total_eff = 0
    for name, r in results.items():
        total_orig += r['original_dim']
        total_eff += r['effective_dim']
        ratio = r['effective_dim'] / r['original_dim'] * 100
        print(f"{name:<35} {r['original_dim']:>8} {r['effective_dim']:>14} {ratio:>7.1f}%")
    print("-" * 72)
    print(f"{'合计':<35} {total_orig:>8} {total_eff:>14} {total_eff / total_orig * 100:>7.1f}%")
    print("=" * 72 + "\n")
