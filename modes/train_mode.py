"""训练模式相关函数"""
import math
import os
import time

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从配置模块导入设备
from core.config import DEVICE, Mode
from net.TripletNet import TripletNet
from plot.plot_loss import plot_loss_curve
from training_utils.TripletDataset import TripletDataset, TripletLoss


def train(mode, data, labels, batch_size=16, num_epochs=200, learning_rate=1e-3,
          net_type=None, preprocess_type=None, test_list=None, model_dir=None):
    """
    准备数据并训练三元组网络模型。

    过程:
    1. 将数据集划分为训练集和验证集(尽管此函数只使用了训练集)。
    2. 创建三元组数据集(TripletDataset)和数据加载器(DataLoader)。
    3. 初始化三元组网络模型(TripletNet)、优化器(如Adam)和损失函数(如TripletLoss)。
    4. 将模型移动到指定的设备(如GPU)上, 并设置模型为训练模式。
    5. 在每个epoch中, 遍历数据加载器, 进行前向传播、计算损失、反向传播和优化步骤。
    6. 记录每个epoch的损失, 并在指定的轮次(test_list)保存模型状态字典。
    7. 在训练的最后几个轮次(test_list[-3:]), 绘制损失随epoch变化的图表并保存。

    :param mode: 模式，用于确定子目录名称及模型保存路径。
    :param data: 输入数据, 通常为图像特征向量。
    :param labels: 输入数据的标签。
    :param batch_size: 批处理大小, 每次迭代训练的网络输入数量。默认为32。
    :param num_epochs: 训练的轮数(遍历整个数据集的次数)。默认为200。
    :param learning_rate: 学习率, 控制优化器更新权重的步长。默认为1e-3。
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试点列表
    :param model_dir: 模型保存路径
    """

    # 子目录路径
    if mode == Mode.TRAIN:
        mode = 'origin'
    model_dir = os.path.join(model_dir, mode)

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 初始化模型和优化器
    model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels)
    model.load_state_dict(torch.load(os.path.join(model_dir, "Extractor_200.pth")), strict=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin=0.1)

    # 训练模型
    model.to(DEVICE)
    model.train()

    print(
        "\n---------------------\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "---------------------\n".format(num_epochs, batch_size, batch_num)
    )
    loss_per_epoch = []

    # 总进度条
    with tqdm(total=num_epochs, desc="Total Progress") as total_bar:
        for epoch in range(num_epochs):
            start_time_ep = time.time()
            total_loss = 0.0
            # 每一轮训练进度条
            with tqdm(total=batch_num, desc=f"Epoch {epoch}", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
                    anchor, positive, negative = (
                        anchor.to(DEVICE),
                        positive.to(DEVICE),
                        negative.to(DEVICE),
                    )

                    # 前向传播
                    embedded_anchor, embedded_positive, embedded_negative = model(
                        anchor, positive, negative
                    )
                    loss = loss_fn(
                        embedded_anchor, embedded_positive, embedded_negative
                    )

                    # 反向传播与优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    pbar.update(1)

            end_time_ep = time.time()

            loss_ep = total_loss / len(train_loader) * 10

            text = (
                f"Epoch [{epoch+1}/{num_epochs}], "
                + f"time: {end_time_ep-start_time_ep:.2f}s, "
                + f"Loss: {loss_ep:.6f}"
            )

            tqdm.write(text)
            loss_per_epoch.append(loss_ep)

            # 保存训练好的模型
            if test_list and (epoch + 1) in test_list:
                # 创建文件夹&文件
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                file_name = f"Extractor_{epoch + 1}.pth"

                # 保存模型到指定路径
                file_path = os.path.join(model_dir, file_name)
                torch.save(model.state_dict(), file_path)
                tqdm.write(f"Model saved to {file_path}")

                # 绘制loss折线图
                if test_list and (epoch + 1) in test_list[-3:]:
                    pic_save_path = os.path.join(model_dir, f"loss_{epoch+1}.png")
                    plot_loss_curve(loss_per_epoch, num_epochs, net_type, preprocess_type, pic_save_path)

            # 更新总进度条
            total_bar.update(1)

    return model