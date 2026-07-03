# paper/plot_pca_ablation.py
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import PAPER_OUTPUT_FILES

# ================= 路径设置 =================
PCA_ABLATION_PLOT_PATH = PAPER_OUTPUT_FILES['pca_ablation']

# ================= 数据 =================
# 场景标签
scenario_labels = [ 'SEEN',
                    'Loc A\n(LOS)', 'Loc B\n(LOS)', 'Loc C\n(LOS)',
                    'Loc D\n(NLOS)', 'Loc E\n(NLOS)', 'Loc F\n(NLOS)']
x_pos = np.arange(len(scenario_labels))

# robustness场景数据
robust_labels = ['B Walk', 'F Walk', 'Mov Office', 'Mov Meeting', 'B Antenna', 'F Antenna']
robust_x_pos = np.arange(len(robust_labels))

# PCA维度
pca_dims = ['1', '2', '4', '8', '16']
width = 0.10

# 每个场景在不同PCA维度下的准确率数据
# static_data = [
#     [99.50, 100.0, 100.0, 100.0, 100.0],  # Loc A
#     [99.20, 100.0, 100.0, 100.0, 100.0],  # Loc B
#     [97.50, 99.50, 99.00, 100.0, 100.0],  # Loc C
#     [99.00, 99.00, 98.00, 98.00, 98.00],  # Loc D
#     [100.00, 100.00, 100.00, 100.00, 100.00], # Loc E
#     [99.50, 100.00, 100.00, 100.00, 100.00]   # Loc F
# ]
#
# robust_data = [
#     [100.00, 100.00, 100.00, 100.00, 100.00],  # B Walk
#     [100.00, 100.00, 100.00, 100.00, 100.00],  # F Walk
#     [86.00, 92.00, 89.00, 90.00, 93.50],       # Mov Office
#     [87.50, 93.50, 93.50, 91.00, 94.00],       # Mov Meeting
#     [99.00, 100.00, 100.00, 99.00, 100.0],     # B Antenna
#     [80.50, 90.00, 88.00, 86.00, 87.00]        # F Antenna
# ]

static_data = [
    [24.33, 61.00, 94.00, 99.33, 99.00],  # SEEN
    [43.50, 83.50, 99.50, 100.0, 100.0],  # Loc A
    [47.00, 83.50, 97.50, 99.00, 99.50],  # Loc B
    [37.50, 65.00, 94.00, 100.0, 100.0],  # Loc C
    [40.00, 77.50, 95.00, 98.00, 98.00],  # Loc D
    [42.50, 76.00, 90.00, 90.50, 92.00], # Loc E
    [55.00, 90.00, 99.50, 100.00, 100.00]   # Loc F
]

robust_data = [
    [33.50, 80.50, 99.50, 100.00, 100.00],  # B Walk
    [56.50, 88.50, 99.00, 100.00, 100.00],  # F Walk
    [41.50, 77.50, 86.00, 93.00, 94.50],       # Mov Office
    [40.00, 72.50, 84.50, 95.50, 96.00],       # Mov Meeting
    [51.50, 96.00, 97.50, 100.00, 99.50],     # B Antenna
    [47.50, 80.50, 94.00, 91.00, 92.50]        # F Antenna
]

if __name__ == '__main__':
    # ================= IEEE 风格设置 =================
    # 使用 Times New Roman 字体 (如果系统有安装，否则回退到 serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # ================= 绘图设置 =================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

    # 原来的 (i-2)*width 改为更大的间隔系数
    bar_positions = [-3*width, -1.5*width, 0, 1.5*width, 3*width]

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
    # ----------------- 上图：Static Environments -----------------
    # 先绘制网格线，设置较低的zorder值，自定义虚线样式
    ax1.grid(axis='y', alpha=0.5, linestyle='--', color='gray', linewidth=0.8, zorder=0)
    for i, dim in enumerate(pca_dims):
        values = [static_data[j][i] for j in range(len(scenario_labels))]
        # 绘制条形图时设置较高的zorder值
        ax1.bar(x_pos + bar_positions[i], values, width,
                label=f'k={dim}', color=colors[i], edgecolor='black', linewidth=0.5, zorder=3)

    ax1.set_ylim(20, 105)
    ax1.set_yticks(np.arange(20, 101, 20))  # 增加纵坐标刻度密度
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.set_title('(a) Static Environments', fontsize=11, loc='left', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenario_labels, fontsize=9)
    ax1.legend(ncol=5, loc='lower center', fontsize=8, framealpha=0.9, edgecolor='gray')

    # ----------------- 下图：Robustness Scenarios -----------------
    # 先绘制网格线，设置较低的zorder值，自定义虚线样式
    ax2.grid(axis='y', alpha=0.5, linestyle='--', color='gray', linewidth=0.8, zorder=0)
    for i, dim in enumerate(pca_dims):
        values = [robust_data[j][i] for j in range(len(robust_labels))]
        # 绘制条形图时设置较高的zorder值
        ax2.bar(robust_x_pos + bar_positions[i], values, width,
                label=f'k={dim}', color=colors[i], edgecolor='black', linewidth=0.5, zorder=3)

    ax2.set_ylim(20, 105)
    ax2.set_yticks(np.arange(20, 101, 20))  # 增加纵坐标刻度密度
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    # ax2.set_xlabel('Scenarios', fontsize=10)
    ax2.set_title('(b) Robustness Scenarios', fontsize=11, loc='left', fontweight='bold')
    ax2.set_xticks(robust_x_pos)
    ax2.set_xticklabels(robust_labels, fontsize=9, rotation=15)
    ax2.legend(ncol=5, loc='lower center', fontsize=8, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(PCA_ABLATION_PLOT_PATH, bbox_inches='tight')
    plt.show()
