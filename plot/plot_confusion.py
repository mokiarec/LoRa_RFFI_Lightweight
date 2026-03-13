import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrices(wwo_cms, wwo_accs, epoch, net_type, preprocess_type, vote_size, save_dir):
    """
    绘制不同分类器在有/无投票情况下的混淆矩阵

    :param wwo_cms: 包含混淆矩阵的列表 [无投票矩阵, 有投票矩阵]
    :param wwo_accs: 包含准确率的列表 [无投票准确率, 有投票准确率]
    :param epoch: 当前训练轮数
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型名称
    :param vote_size: 投票窗口大小
    :param save_dir: 图片保存目录
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    types = ["KNN", "SVM", "Combined"]
    wwo = ["w/o", "w/"]

    for i in range(2):
        for j in range(2 if i == 0 else 3):
            # for j in range(2 if i == 0 else 2):
            sns.heatmap(
                wwo_cms[i][j],
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                ax=axs[i][j],
            )
            axs[i][j].set_title(
                f"{types[j]} {wwo[i]} Vote (Accuracy = {wwo_accs[i][j] * 100:.2f}%)"
            )
            axs[i][j].set_xlabel("Predicted label")
            axs[i][j].set_ylabel("True label")

    # 删除第一行第三个子图
    fig.delaxes(axs[0, 2])
    fig.suptitle(
        f"Heatmap Comparison After {epoch} Epochs "
        f"net type: {net_type}, pps: {preprocess_type}, Vote Size: {vote_size}",
        fontsize=16,
    )
    pic_save_path = os.path.join(save_dir, f"cft_{epoch}.png")
    plt.savefig(pic_save_path)
    print(f"Png save path: {pic_save_path}")
    # plt.show()
