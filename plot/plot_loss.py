from matplotlib import pyplot as plt


def plot_loss_curve(loss_per_epoch, num_epochs, net_type, preprocess_type, pic_save_path):
    """
    绘制并保存训练损失曲线

    :param loss_per_epoch: 每个epoch的损失值列表
    :param num_epochs: 总训练轮数
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param pic_save_path: 图片保存路径
    """
    fig, ax1 = plt.subplots()
    ax1.plot(
        range(len(loss_per_epoch)),
        loss_per_epoch,
        label="Loss",
        color="red",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # 添加标题和图例
    net_name = net_type.value
    plt.title(
        f"Loss of {num_epochs} Epoch, Net: {net_name}, Convert Type: {preprocess_type.value}"
    )
    fig.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        bbox_transform=ax1.transAxes,
    )

    # 显示图表
    plt.grid(True)
    plt.savefig(pic_save_path)
    # plt.show()
    plt.close()  # 关闭图形以释放内存
