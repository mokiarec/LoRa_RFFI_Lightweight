"""训练模式相关函数"""
import math
import os
import time

import swanlab
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从配置模块导入设备
from core.config import DEVICE
from net.TripletNet import TripletNet
from plot.plot_loss import plot_loss_curve
from training_utils.TripletDataset import TripletDataset, TripletLoss


def train(config, data, labels, batch_size=16, num_epochs=200, learning_rate=1e-3):
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

    :param data: 输入数据, 通常为图像特征向量。
    :param labels: 输入数据的标签。
    :param batch_size: 批处理大小, 每次迭代训练的网络输入数量。默认为32。
    :param num_epochs: 训练的轮数(遍历整个数据集的次数)。默认为200。
    :param learning_rate: 学习率, 控制优化器更新权重的步长。默认为1e-3。
    """

    try:
        import swanlab
        SWANLAB_AVAILABLE = True
    except ImportError:
        SWANLAB_AVAILABLE = False

    # 创建权重目录
    weights_dir = os.path.join(config.MODEL_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 准备验证集数据 (转换为 tensor)
    valid_data_tensor = torch.tensor(data_valid, dtype=torch.float32).to(DEVICE)
    valid_labels_tensor = torch.tensor(labels_valid, dtype=torch.long).to(DEVICE)

    # 初始化模型和优化器
    model = TripletNet(net_type=config.NET_TYPE, in_channels=config.PREPROCESS_TYPE.in_channels)

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

    # 追踪最佳模型
    best_accuracy = 0.0
    best_epoch = 0

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

            # 验证集评估 (最近邻分类)
            model.eval()
            with torch.no_grad():
                # 特征提取：获取验证集数据的 embedding
                valid_embeddings = model.embedding_net(valid_data_tensor)

                # 计算距离矩阵 (样本间欧氏距离)
                distance_matrix = torch.cdist(valid_embeddings, valid_embeddings, p=2)

                # 最近邻投票
                n_neighbors = 5  # 可以调整为其他值，如 1, 3, 5 等

                # 对于每个样本，找到距离最近的 N 个邻居 (不包括自己)
                # 获取距离矩阵的排序索引 (按距离从小到大)
                _, sorted_indices = torch.sort(distance_matrix, dim=1)

                # 取前 N+1 个 (包含自己)，然后去掉自己，取 N 个邻居
                k_indices = sorted_indices[:, 1:n_neighbors + 1]

                # 获取邻居的标签
                neighbor_labels = valid_labels_tensor[k_indices]

                # 投票：统计每个标签的出现次数，选择最高频的标签
                predicted_labels = torch.mode(neighbor_labels, dim=1)[0]

                # 计算准确率
                accuracy = (predicted_labels == valid_labels_tensor).float().mean().item() * 100

            model.train()

            # 输出验证结果
            text += f", Val Acc@{n_neighbors}NN: {accuracy:.2f}%"

            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                text += f" ⭐ Best"

                # 保存最佳模型到指定路径
                best_file_path = os.path.join(weights_dir, "Extractor_best.pth")
                torch.save(model.state_dict(), best_file_path)
                tqdm.write(f"Best model saved to {best_file_path} (Acc: {accuracy:.2f}%)")

            # 更新总进度条
            total_bar.update(1)

            tqdm.write(text)
            loss_per_epoch.append(loss_ep)

            # 记录到 SwanLab
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "train/loss_epoch": loss_ep,
                    "train/val_accuracy": accuracy,
                    "train/epoch_time": end_time_ep - start_time_ep,
                }, step=epoch + 1)

            # 保存固定checkpoint的模型
            if config.TEST_LIST and (epoch + 1) in config.TEST_LIST:
                # 保存模型到指定路径
                file_name = f"Extractor_{epoch + 1}.pth"
                file_path = os.path.join(weights_dir, file_name)
                torch.save(model.state_dict(), file_path)
                tqdm.write(f"Model saved to {file_path}")

                # 绘制loss折线图
                if config.TEST_LIST and (epoch + 1) in config.TEST_LIST[-3:]:
                    pic_save_path = os.path.join(config.MODEL_DIR, f"loss_{epoch+1}.png")
                    plot_loss_curve(loss_per_epoch, num_epochs, config.NET_TYPE, config.PREPROCESS_TYPE, pic_save_path)


    # 打印最佳模型信息
    print(f"\n{'=' * 50}")
    print(f"训练完成！最佳模型信息:")
    print(f"  - Epoch: {best_epoch}")
    print(f"  - 验证集准确率：{best_accuracy:.2f}%")
    print(f"  - 保存路径：{os.path.join(config.MODEL_DIR, 'Extractor_best.pth')}")
    print(f"{'=' * 50}\n")

    return model