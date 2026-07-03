# paper/plot_accuracy.py
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import PAPER_OUTPUT_FILES

# 设置保存路径变量
CHANNEL_ROBUSTNESS_PLOT_PATH = PAPER_OUTPUT_FILES['channel_robustness']
MOBILITY_ROBUSTNESS_PLOT_PATH = PAPER_OUTPUT_FILES['mobility_robustness']
ACCURACY_HEATMAP_PLOT_PATH = PAPER_OUTPUT_FILES['accuracy_heatmap']
DETAILED_ACCURACY_HEATMAP_PLOT_PATH = PAPER_OUTPUT_FILES['detailed_accuracy_heatmap']

# 联合滑动投票测试结果
# experiments = {
#     "ResNet18 (Teacher) + w/o PCA": {
#         "Seen": 99.33,
#         "LOS": [95.00, 95.00, 90.00],       # A, B, C
#         "NLOS": [90.50, 80.50, 90.50],      # D, E, F
#         "Walk": [96.50, 91.50],             # B_walk, F_walk
#         "Mobile": [96.0, 97.5],             # Office, Meeting
#         "Antenna": [89.50, 82.00],          # B, F
#     },
#
#     "MobileNetV2 (No KD) + w/o PCA": {
#         "Seen": 71.00,
#         "LOS": [86.00, 81.50, 78.00],
#         "NLOS": [84.00, 67.50, 90.00],
#         "Walk": [83.50, 89.50],
#         "Mobile": [77.00, 81.00],
#         "Antenna": [86.50, 85.50]
#     },
#
#     "MobileNetV1 (No KD) + w/o PCA": {
#         "Seen": 61.00,
#         "LOS": [90.50, 73.00, 79.00],
#         "NLOS": [88.00, 69.50, 86.00],
#         "Walk": [75.50, 77.00],
#         "Mobile": [71.50, 85.50],
#         "Antenna": [88.00, 94.50],
#     },
#
#     "MobileNetV1 (No KD) + w/  PCA": {
#         "Seen": 72.33,
#         "LOS": [88.00, 76.00, 84.00],
#         "NLOS": [89.00, 68.50, 88.50],
#         "Walk": [78.50, 84.00],
#         "Mobile": [75.50, 87.00],
#         "Antenna": [87.50, 95.00],
#     },
#
#     "MobileNetV1 (PCA KD) + w/o PCA": {
#         "Seen": 97.67,
#         "LOS": [100.00, 97.50, 91.00],
#         "NLOS": [97.00, 91.00, 99.50],
#         "Walk": [99.00, 98.00],
#         "Mobile": [77.50, 81.00],
#         "Antenna": [91.00, 85.00],
#     },
#
#     "MobileNetV1 (PCA KD) + w/  PCA": {
#         "Seen": 98.67,
#         "LOS": [99.50, 99.00, 90.00],
#         "NLOS": [94.50, 88.00, 100.00],
#         "Walk": [97.50, 97.50],
#         "Mobile": [75.00, 81.50],
#         "Antenna": [91.00, 84.50],
#     },
#
#     "LightNet (No KD) + w/o PCA": {
#         "Seen": 97.67,
#         "LOS": [99.50, 90.50, 90.00],
#         "NLOS": [98.00, 68.00, 98.50],
#         "Walk": [98.00, 94.50],
#         "Mobile": [97.50, 99.00],
#         "Antenna": [99.50, 92.50],
#     },
#
#     "LightNet (No KD) + w/  PCA": {
#         "Seen": 98.50,
#         "LOS": [99.50, 90.00, 90.00],
#         "NLOS": [98.00, 66.00, 99.00],
#         "Walk": [97.50, 93.50],
#         "Mobile": [97.50, 99.00],
#         "Antenna": [100.0, 93.00],
#     },
#
#     "LightNet (PCA KD) + w/o PCA": {
#         "Seen": 99.00,
#         "LOS": [100.0, 99.50, 100.0],
#         "NLOS": [98.0, 90.50, 100.0],
#         "Walk": [100.0, 100.0],
#         "Mobile": [91.50, 94.50],
#         "Antenna": [98.50, 91.00],
#     },
#
#     "LightNet (PCA KD) + w/  PCA": {
#         "Seen": 99.00,
#         "LOS": [100.0, 99.50, 100.0],
#         "NLOS": [98.0, 92.00, 100.0],
#         "Walk": [100.0, 100.0],
#         "Mobile": [94.50, 96.00],
#         "Antenna": [99.50, 92.50],
#     },
# }

# KNN w/o 测试结果
experiments = {
    "ResNet18 (Teacher) + w/o PCA": {
        "Seen": 99.33,
        "LOS": [95.00, 95.00, 90.00],       # A, B, C
        "NLOS": [90.50, 80.50, 90.50],      # D, E, F
        "Walk": [96.50, 91.50],             # B_walk, F_walk
        "Mobile": [96.0, 97.5],             # Office, Meeting
        "Antenna": [89.50, 82.00],          # B, F
    },

    "MobileNetV1 (No KD) + w/o PCA": {
        "Seen": 61.00,
        "LOS": [90.50, 73.00, 79.00],
        "NLOS": [88.00, 69.50, 86.00],
        "Walk": [75.50, 77.00],
        "Mobile": [71.50, 85.50],
        "Antenna": [88.00, 94.50],
    },

    "MobileNetV1 (No KD) + w/  PCA": {
        "Seen": 72.33,
        "LOS": [88.00, 76.00, 84.00],
        "NLOS": [89.00, 68.50, 88.50],
        "Walk": [78.50, 84.00],
        "Mobile": [75.50, 87.00],
        "Antenna": [87.50, 95.00],
    },

    "MobileNetV1 (PCA KD) + w/o PCA": {
        "Seen": 97.67,
        "LOS": [100.00, 97.50, 91.00],
        "NLOS": [97.00, 91.00, 99.50],
        "Walk": [99.00, 98.00],
        "Mobile": [77.50, 81.00],
        "Antenna": [91.00, 85.00],
    },

    "MobileNetV1 (PCA KD) + w/  PCA": {
        "Seen": 98.67,
        "LOS": [99.50, 99.00, 90.00],
        "NLOS": [94.50, 88.00, 100.00],
        "Walk": [97.50, 97.50],
        "Mobile": [75.00, 81.50],
        "Antenna": [91.00, 84.50],
    },

    "LightNet (No KD) + w/o PCA": {
        "Seen": 97.67,
        "LOS": [99.50, 90.50, 90.00],
        "NLOS": [98.00, 68.00, 98.50],
        "Walk": [98.00, 94.50],
        "Mobile": [97.50, 99.00],
        "Antenna": [99.50, 92.50],
    },

    "LightNet (No KD) + w/  PCA": {
        "Seen": 98.50,
        "LOS": [99.50, 90.00, 90.00],
        "NLOS": [98.00, 66.00, 99.00],
        "Walk": [97.50, 93.50],
        "Mobile": [97.50, 99.00],
        "Antenna": [100.0, 93.00],
    },

    "LightNet (PCA KD) + w/o PCA": {
        "Seen": 99.00,
        "LOS": [100.0, 99.50, 100.0],
        "NLOS": [98.0, 90.50, 100.0],
        "Walk": [100.0, 100.0],
        "Mobile": [91.50, 94.50],
        "Antenna": [98.50, 91.00],
    },

    "LightNet (PCA KD) + w/  PCA": {
        "Seen": 99.00,
        "LOS": [100.0, 99.50, 100.0],
        "NLOS": [98.0, 92.00, 100.0],
        "Walk": [100.0, 100.0],
        "Mobile": [94.50, 96.00],
        "Antenna": [99.50, 92.50],
    },
}


if __name__ == '__main__':

    ##############################################
    # 图一：通道鲁棒性比较柱状图（LOS vs NLOS）
    ##############################################

    labels = []
    los_avg = []
    nlos_avg = []

    for name, v in experiments.items():
        labels.append(name)
        los_avg.append(np.mean(v["LOS"]))
        nlos_avg.append(np.mean(v["NLOS"]))

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, los_avg, width, label="LOS Avg")
    plt.bar(x + width/2, nlos_avg, width, label="NLOS Avg")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Channel Robustness Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(CHANNEL_ROBUSTNESS_PLOT_PATH, format="pdf", bbox_inches='tight')
    plt.show()

    ##############################################
    # 图二：移动性鲁棒性折线图（Walk vs Mobile）
    ##############################################

    walk_avg = []
    mobile_avg = []

    for v in experiments.values():
        walk_avg.append(np.mean(v["Walk"]))
        mobile_avg.append(np.mean(v["Mobile"]))

    plt.figure(figsize=(10, 4))
    plt.plot(labels, walk_avg, marker="o", label="Walk")
    plt.plot(labels, mobile_avg, marker="s", label="Mobile")

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Mobility Robustness")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(MOBILITY_ROBUSTNESS_PLOT_PATH, format="pdf", bbox_inches='tight')
    plt.show()

    ##############################################
    # 图三：条件概览热力图（每类平均表现）
    ##############################################

    # 1. 数据处理
    rows = []

    for name in experiments.keys():
        if name not in experiments: continue
        v = experiments[name]
        rows.append([
            name,
            v["Seen"],
            np.mean(v["LOS"]),
            np.mean(v["NLOS"]),
            np.mean(v["Walk"]),
            np.mean(v["Mobile"]),
            np.mean(v["Antenna"]),
        ])

    df = pd.DataFrame(
        rows,
        columns=["Strategy", "Seen", "LOS", "NLOS", "Walk", "Mobile", "Antenna"]
    )

    # 设置 Strategy 为索引
    df.set_index("Strategy", inplace=True)

    # 2. 绘图设置
    fig, ax = plt.subplots(figsize=(9, 4.5))
    norm = PowerNorm(gamma=4, vmin=60, vmax=100)
    custom_ticks = [60, 90, 95, 98, 100]

    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        norm=norm,
        linewidths=1,
        linecolor='white',
        cbar_kws={
            'label': 'Accuracy (%)',
            'ticks': custom_ticks,
            'format': '%.0f',
        },
        ax = ax,
    )

    # # 为前三个模型策略添加统一红色边框（带间距）
    # padding = 0
    # rect = plt.Rectangle(
    #     (-padding, -padding),
    #     len(df.columns) + 2 * padding,
    #     2 + 2 * padding,
    #     fill=False,
    #     edgecolor='red',
    #     lw=2
    # )
    # ax.add_patch(rect)

    # 3. 美化标签
    plt.title("Ablation Study: Accuracy Comparison Across Scenarios", fontsize=12, pad=15, fontweight='bold')
    plt.ylabel("")
    plt.xlabel("")

    # 旋转X轴标签，防止重叠
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(ACCURACY_HEATMAP_PLOT_PATH, format="pdf", bbox_inches='tight')
    plt.show()

    ##############################################
    # 图四：详细热力图（所有实验点）
    ##############################################

    # 构建详细数据帧
    detailed_rows = []
    for model_name, data in experiments.items():
        # 添加Seen数据
        detailed_rows.append([model_name, 'Seen', data["Seen"]])

        # 添加LOS数据 (A, B, C)
        los_keys = ['A', 'B', 'C']
        for i, value in enumerate(data["LOS"]):
            detailed_rows.append([model_name, f'LOS-{los_keys[i]}', value])

        # 添加NLOS数据 (D, E, F)
        nlos_keys = ['D', 'E', 'F']
        for i, value in enumerate(data["NLOS"]):
            detailed_rows.append([model_name, f'NLOS-{nlos_keys[i]}', value])

        # 添加Walk数据
        walk_keys = ['B_walk', 'F_walk']
        for i, value in enumerate(data["Walk"]):
            detailed_rows.append([model_name, f'{walk_keys[i]}', value])

        # 添加Mobile数据
        mobile_keys = ['Office', 'Meeting']
        for i, value in enumerate(data["Mobile"]):
            detailed_rows.append([model_name, f'Mobile-{mobile_keys[i]}', value])

        # 添加Antenna数据
        antenna_keys = ['B_antenna', 'F_antenna']
        for i, value in enumerate(data["Antenna"]):
            detailed_rows.append([model_name, f'{antenna_keys[i]}', value])

    # 创建DataFrame
    detailed_df = pd.DataFrame(detailed_rows, columns=["Strategy", "Condition", "Accuracy"])

    # 转换为透视表格式
    pivot_df = detailed_df.pivot(index="Strategy", columns="Condition", values="Accuracy")

    # 在绘制详细热力图之前添加列排序
    column_order = ["Seen", "LOS-A", "LOS-B", "LOS-C",
                    "NLOS-D", "NLOS-E", "NLOS-F",
                    "B_walk", "F_walk",
                    "Mobile-Office", "Mobile-Meeting",
                    "B_antenna", "F_antenna"]

    # 按照指定顺序重新排列列
    pivot_df = pivot_df[column_order]

    # 保持行（模型）顺序与原始 experiments 一致
    model_order = list(experiments.keys())
    pivot_df = pivot_df.reindex(model_order)

    # 绘制详细热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        norm=norm
    )
    plt.title("Detailed Accuracy Heatmap (All Data Points)")
    plt.tight_layout()
    plt.savefig(DETAILED_ACCURACY_HEATMAP_PLOT_PATH, format="pdf", bbox_inches='tight')
    plt.show()
