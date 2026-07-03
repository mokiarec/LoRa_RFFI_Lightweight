# paper/plot_pca_comparison.py
"""多网络 PCA 有效维度对比分析 — 2×2 排版统一图例"""
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PreprocessType, DEVICE
from net import NetworkType
from utils.data_preprocessor import load_generate_triplet, load_model
from dataset import *
from paths import PathManager

path_manager = PathManager()

# ==================== 实验配置 ====================
EXPERIMENTS = [
    {
        "name": "ResNet18",
        "exp_dir": "EXP_02_ResNet_Base",
        "net_type": NetworkType.ResNet,
        "label": "(a) ResNet",
    },
    {
        "name": "SCSKNet",
        "exp_dir": "EXP_20_SCSKNet_Base",
        "net_type": NetworkType.SCSKNet,
        "label": "(b) SCSKNet",
    },
    {
        "name": "ShuffleNetV2",
        "exp_dir": "EXP_24_ShuffleNet_Base",
        "net_type": NetworkType.ShuffleNet,
        "label": "(c) ShuffleNetV2",
    },
    {
        "name": "DenseNet",
        "exp_dir": "EXP_22_DenseNet_Base",
        "net_type": NetworkType.DenseNet,
        "label": "(d) DenseNet",
    },
]


def extract_conv_layers(model):
    """提取模型中所有 Conv2d 层"""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layers.append((name, module))
    return layers


def analyze_layer_pca(layer, input_data, max_components=64):
    """分析单个卷积层的 PCA 有效维度"""
    with torch.no_grad():
        output = layer(input_data)
        # [B, C, H, W] -> [B, C]  全局平均池化
        features = output.mean(dim=[2, 3]).cpu().numpy()

    n_samples, n_features = features.shape
    n_components = min(max_components, n_features, n_samples)
    pca = PCA(n_components=n_components)
    pca.fit(features)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    effective_dims_95 = np.argmax(cumulative_variance >= 0.95) + 1
    effective_dims_99 = np.argmax(cumulative_variance >= 0.99) + 1

    return {
        "output_dim": n_features,
        "effective_dims_95": effective_dims_95,
        "effective_dims_99": effective_dims_99,
        "cumulative_variance": cumulative_variance,
    }


def analyze_network(model_path, net_type, preprocess_type, test_data):
    """分析单个网络的所有卷积层"""
    print(f"  Loading model: {model_path}")
    model = load_model(str(model_path), net_type, preprocess_type)
    model = model.to(DEVICE)
    test_data = test_data.to(DEVICE)

    embedding_net = model.embedding_net
    embedding_net.eval()

    all_layers = extract_conv_layers(embedding_net)
    print(f"  Found {len(all_layers)} Conv2d layers")

    # 注册 forward hook 捕获每层输入
    layer_inputs = {}
    hooks = []

    def create_hook(name):
        def hook(module, input_, output_):
            layer_inputs[name] = input_[0].detach()

        return hook

    for name, layer in all_layers:
        hook = layer.register_forward_hook(create_hook(name))
        hooks.append(hook)

    with torch.no_grad():
        _ = embedding_net(test_data)

    for hook in hooks:
        hook.remove()

    results = []
    for name, layer in all_layers:
        if name in layer_inputs:
            input_data = layer_inputs[name]
            result = analyze_layer_pca(layer, input_data)
            result["layer_name"] = name
            results.append(result)

    return results


def simplify_layer_name(name: str) -> str:
    """简化层名用于显示"""
    name = name.replace("features.", "")
    name = name.replace("stages.", "S")
    # 截断过长名称
    if len(name) > 20:
        parts = name.split(".")
        short = ".".join(p[:2] if len(p) > 2 else p for p in parts)
        name = short
    return name


def main():
    # ==================== 加载数据 ====================
    file_path_enrol = str(DATASET["Test"]["seen"].path)
    dev_range_enrol = np.arange(0, 40)
    pkt_range_enrol = np.arange(0, 10)
    preprocess_type = PreprocessType.STFT

    print("Loading data...")
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        preprocess_type, snr_range=None,
    )
    anchor_data = triplet_data_enrol[0]

    # ==================== 逐网络分析 ====================
    all_results = {}
    for exp in EXPERIMENTS:
        print(f"\nAnalyzing {exp['name']} ...")
        model_path = (
            path_manager.checkpoints_dir
            / exp["exp_dir"]
            / "weights"
            / "Extractor_best.pth"
        )
        results = analyze_network(model_path, exp["net_type"], preprocess_type, anchor_data)
        all_results[exp["name"]] = {"results": results, "label": exp["label"]}

    # ==================== 打印摘要 ====================
    for net_name, data in all_results.items():
        results = data["results"]
        print(f"\n{'='*60}")
        print(f"SUMMARY: {net_name}")
        print(f"{'='*60}")
        for r in results:
            print(
                f"{r['layer_name']:40s} | Out: {r['output_dim']:4d} | "
                f"95%: {r['effective_dims_95']:4d} | 99%: {r['effective_dims_99']:4d}"
            )

    # ==================== 绘图 ====================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(14, 8.5))

    # 统一色彩方案
    C_NATIVE = "#888888"  # 灰色
    C_TAU99 = "#1f77b4"  # 蓝色
    C_TAU95 = "#d62728"  # 红色

    # 2×2 子图
    subplot_order = [
        (1, "ResNet18"),      # a
        (2, "SCSKNet"),        # b
        (3, "ShuffleNetV2"),   # c
        (4, "DenseNet"),       # d
    ]

    for pos, net_name in subplot_order:
        data = all_results[net_name]
        results = data["results"]
        label = data["label"]

        layer_names = [r["layer_name"] for r in results]
        output_dims = [r["output_dim"] for r in results]
        eff_dims_99 = [r["effective_dims_99"] for r in results]
        eff_dims_95 = [r["effective_dims_95"] for r in results]

        x = np.arange(len(layer_names))
        ax = fig.add_subplot(2, 2, pos)

        # 判断是否显示具体层名
        show_names = net_name in ("ResNet18", "SCSKNet")

        # --- 三条曲线 ---
        ax.plot(
            x, output_dims, "--", color=C_NATIVE, linewidth=1.3,
            marker="o", markersize=3, markerfacecolor=C_NATIVE, markeredgewidth=0.4,
        )
        ax.plot(
            x, eff_dims_99, "-", color=C_TAU99, linewidth=1.5,
            marker="s", markersize=3, markerfacecolor=C_TAU99, markeredgewidth=0.4,
        )
        ax.plot(
            x, eff_dims_95, "-", color=C_TAU95, linewidth=2.8,
            marker="^", markersize=3, markerfacecolor=C_TAU95, markeredgewidth=0.4,
        )

        ax.set_title(label, fontsize=12, pad=8)
        ax.set_ylabel("Dimensions", fontsize=11)

        if show_names:
            # 精简层名并稀疏显示
            simple = [simplify_layer_name(n) for n in layer_names]
            n = len(simple)
            # 最多显示 10 个标签
            step = max(1, n // 10)
            tick_labels = [s if i % step == 0 else "" for i, s in enumerate(simple)]
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=6)
        else:
            ax.set_xlabel("Layer Index", fontsize=11)
            ax.set_xticks(x)
            # 每隔适当步长显示数字，避免拥挤
            step = max(1, len(x) // 20)
            tick_labels = [str(i + 1) if i % step == 0 else "" for i in range(len(x))]
            ax.set_xticklabels(tick_labels, fontsize=5)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", which="major", direction="in")

    # ==================== 统一图例（图上方居中） ====================
    legend_handles = [
        plt.Line2D([0], [0], linestyle="--", color=C_NATIVE, linewidth=1.5,
                   marker="o", markersize=4),
        plt.Line2D([0], [0], linestyle="-", color=C_TAU99, linewidth=1.5,
                   marker="s", markersize=4),
        plt.Line2D([0], [0], linestyle="-", color=C_TAU95, linewidth=2.8,
                   marker="^", markersize=4),
    ]
    legend_labels = [
        "Native Channels",
        r"$\tau = 0.99$",
        r"$\tau = 0.95$",
    ]

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=3,
        frameon=True, fancybox=False, edgecolor="black",
        fontsize=11, handlelength=2.8,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = path_manager.paper_results / "pca_comparison_four_networks.pdf"
    plt.savefig(str(output_path), bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
