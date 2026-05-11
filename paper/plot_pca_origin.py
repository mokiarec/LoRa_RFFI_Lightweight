# paper/plot_pca_origin.py
import os
import sys

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PreprocessType
from net import NetworkType

from utils.data_preprocessor import load_generate_triplet, load_model
from dataset import *
from paths import PAPER_OUTPUT_FILES

# ================= 路径设置 =================
PCA_ORIGIN_PLOT_PATH = PAPER_OUTPUT_FILES['pca_origin_pca']

def plot_pca_comparison(feats, n_components=16):
    """
    绘制PCA降维前后的特征对比图 (IEEE单页适配版)
    """
    # 执行PCA
    pca = PCA(n_components=min(n_components, feats.shape[1]))
    pca.fit(feats)
    feats_compressed = pca.fit_transform(feats)
    feats_reconstructed = pca.inverse_transform(feats_compressed)   # 重构原始数据
    # 计算重构误差
    reconstruction_error = np.mean((feats - feats_reconstructed) ** 2)
    print(f"重构均方误差: {reconstruction_error}")

    # 原始特征方差
    feature_variances = np.asarray(feats).var(axis=0)

    # PCA方差贡献
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)  # 计算累计方差

    # 创建对比图 - 调整为单列宽度(约3.5英寸)或双列宽度(约7.5英寸)
    # IEEE单栏宽度约为3.5英寸，双栏宽度约为7.5英寸
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))  # 双栏宽度适配
    # 使用 Times New Roman 字体 (如果系统有安装，否则回退到 serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # --- 左图：原始特征方差 ---
    # 使用深蓝色，线条稍微细一点
    axes[0].plot(feature_variances, linewidth=0.8, color='#1f77b4')
    axes[0].set_xlabel('Feature Dimension Index', fontsize=9)
    axes[0].set_ylabel('Variance', fontsize=9)  # 建议：如果差异太大，可以用对数坐标

    axes[0].tick_params(axis='both', which='major', labelsize=8, direction='in')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].set_title('(a)', y=-0.35, fontsize=10)  # 标题放下面

    # --- 右图：PCA 主成分 ---
    # 1. 画柱状图 (单一贡献)
    x_indices = range(1, len(explained_variance_ratio) + 1)
    bars = axes[1].bar(x_indices, explained_variance_ratio,
                       width=0.7, color='#2ca02c', label='Individual Ratio', alpha=0.8)

    # 2. 画折线图 (累计贡献) - 双Y轴是 PCA 图的标配
    ax2 = axes[1].twinx()
    line = ax2.plot(x_indices, cumulative_variance,
                    color='#d62728', linewidth=1.2, marker='.', markersize=4, label='Cumulative Ratio')

    # 设置标签
    axes[1].set_xlabel('Principal Component Index', fontsize=9)
    axes[1].set_ylabel('Explained Variance Ratio', fontsize=9, color='#2ca02c')
    ax2.set_ylabel('Cumulative Variance', fontsize=9, color='#d62728')

    # 设置刻度颜色
    axes[1].tick_params(axis='y', labelcolor='#2ca02c', labelsize=8, direction='in')
    ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=8, direction='in')
    axes[1].tick_params(axis='x', labelsize=8, direction='in')

    # 图例 (合并双轴图例)
    # lines_1, labels_1 = axes[1].get_legend_handles_labels() # 柱状图没有 handle 容易报错，手动处理
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=8)

    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].set_title('(b)', y=-0.35, fontsize=10)

    # ================= 4. 布局调整 =================
    # 调整底部边距，给下方的标题留出空间
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    # 保存
    # plt.savefig(PCA_ORIGIN_PLOT_PATH, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    # 修正数据集访问路径
    file_path_enrol = str(DATASET['Test']['seen'].path)
    # dev_range_enrol = DATASET['Test']['seen'].dev_range
    # pkt_range_enrol = DATASET['Test']['seen'].pkt_range

    # file_path_enrol = "D:\ScienceProject\LoRa_RFFI/dataset/DATA_all_dev_1~11_300times_433m_1M_3gain.h5"
    dev_range_enrol = np.arange(0, 5)
    pkt_range_enrol = np.arange(0, 100)
    
    net_type = NetworkType.ResNet
    model_path = "D:\ScienceProject\LoRa_RFFI/checkpoints/EXP_02_ResNet_Base/weights/Extractor_best.pth"
    # model_path = "D:\ScienceProject\LoRa_RFFI/model/stft/Drsn/origin/Extractor_300.pth"
    preprocess_type = PreprocessType.STFT
    
    # 加载注册数据集(IQ样本和标签)
    print("\nData loading...")
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        preprocess_type, snr_range=None
    )

    model = load_model(str(model_path), net_type, preprocess_type)
    
    # 将模型移动到正确的设备
    from core.config import DEVICE
    model = model.to(DEVICE)
    
    # 将数据移动到正确的设备
    triplet_data_enrol = [data.to(DEVICE) for data in triplet_data_enrol]

    with torch.no_grad():
        # 模型返回三元组特征 (anchor, positive, negative)
        feature_anchor, feature_positive, feature_negative = model(*triplet_data_enrol)
        # 使用 anchor 特征进行 PCA 分析
        feature_enrol = feature_anchor.cpu().numpy()
    
    plot_pca_comparison(feats=feature_enrol, n_components=16)
