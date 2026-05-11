# \modes\distillation_mode.py
"""蒸馏模式相关函数"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config, PreprocessType
from core.config import DEVICE
from net import TripletNet
from plot.plot_loss import plot_loss_curve
from utils.TripletDataset import TripletDataset, TripletLoss
from utils.data_preprocessor import generate_spectrogram


def distillation(
    config: Config,
    data,
    labels,
    teacher_model_path,
    batch_size=32,
    num_epochs=200,
    learning_rate=1e-3,
    temperature=3.0,
    alpha=0.7,
    is_pca=True,
    pca_file_path=None,
):
    """
    使用知识蒸馏训练轻量级模型

    :param mode: 模式，用于确定子目录名称及模型保存路径。
    :param data: 输入数据
    :param labels: 输入数据的标签
    :param teacher_model_path: 教师模型路径
    :param batch_size: 批处理大小
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param temperature: 蒸馏温度
    :param alpha: 蒸馏损失权重
    :param is_pca: 是否使用PCA降维处理特征（默认为True）
    :param pca_file_path: PCA文件路径
    """

    try:
        import swanlab
        SWANLAB_AVAILABLE = not config.disable_swanlab
    except ImportError:
        SWANLAB_AVAILABLE = False

    if is_pca:
        # 加载 PCA
        pca = np.load(pca_file_path)
        W = torch.tensor(pca['components'].T, dtype=torch.float32).to(DEVICE)  # D_t x d
        mean = torch.tensor(pca['mean'], dtype=torch.float32).to(DEVICE)
        d = W.shape[1]
        print(f"PCA dim: {d}")
        print("PCA data loaded!!!")

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train, PreprocessType.STFT)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 初始化教师模型
    teacher_model = TripletNet(net_type=config.base_net_type, in_channels=config.preprocess_type.in_channels)  # RESNET
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    teacher_model.to(DEVICE)
    teacher_model.eval()

    # 准备验证集数据 (转换为 tensor)
    valid_data_tensor = torch.tensor(data_valid, dtype=torch.float32).to(DEVICE)
    valid_labels_tensor = torch.tensor(labels_valid, dtype=torch.long).to(DEVICE)

    # 初始化学生模型 (MobileNet)
    student_model = TripletNet(net_type=config.net_type, in_channels=config.preprocess_type.in_channels)  # MobileNet
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    triplet_loss_fn = TripletLoss(margin=0.1)

    # 训练模型
    student_model.to(DEVICE)
    student_model.train()

    print(
        "\n---------------------\n"
        "Distillation Training\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "Temperature: {}\n"
        "Alpha: {}\n"
        "PCA: {}\n"
        "---------------------\n".format(num_epochs, batch_size, batch_num, temperature, alpha, is_pca)
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

                    # 对特征进行PCA降维和归一化处理
                    def process_pca_features(feature):
                        f_centered = feature - mean.unsqueeze(0)
                        f_pca = f_centered @ W  # B x d
                        return F.normalize(f_pca, p=2, dim=1)

                    # 教师模型前向传播（不计算梯度）
                    with torch.no_grad():
                        teacher_anchor, teacher_positive, teacher_negative = teacher_model(
                            anchor, positive, negative
                        )

                        if is_pca:
                            # 对教师特征进行PCA降维和归一化处理
                            teacher_anchor = process_pca_features(teacher_anchor)
                            teacher_positive = process_pca_features(teacher_positive)
                            teacher_negative = process_pca_features(teacher_negative)

                    # 学生模型前向传播
                    student_anchor, student_positive, student_negative = student_model(
                        anchor, positive, negative
                    )

                    # 三元组损失
                    triplet_loss = triplet_loss_fn(
                        student_anchor, student_positive, student_negative
                    )

                    # if is_pca:
                    #     # 对学生特征进行PCA降维和归一化处理
                    #     student_anchor = process_pca_features(student_anchor)
                    #     student_positive = process_pca_features(student_positive)
                    #     student_negative = process_pca_features(student_negative)

                    # 蒸馏损失（使用KL散度）
                    distill_loss = (    # 教师为目标分布, 学生为源分布
                       F.kl_div(
                           F.log_softmax(student_anchor / temperature, dim=1),
                           F.softmax(teacher_anchor / temperature, dim=1),
                           reduction='batchmean'    # 把每个样本的 KL 散度求完后, 再对 batch 取平均
                       ) +
                       F.kl_div(
                           F.log_softmax(student_positive / temperature, dim=1),
                           F.softmax(teacher_positive / temperature, dim=1),
                           reduction='batchmean'
                       ) +
                       F.kl_div(
                           F.log_softmax(student_negative / temperature, dim=1),
                           F.softmax(teacher_negative / temperature, dim=1),
                           reduction='batchmean'
                       )
                    ) * (temperature ** 2)

                    # 总损失
                    loss = (1 - alpha) * triplet_loss + alpha * distill_loss

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

            # 验证集评估 (最近邻分类)
            student_model.eval()
            with torch.no_grad():
                # 特征提取：获取验证集数据的 embedding
                valid_embeddings = student_model.embedding_net(valid_data_tensor)

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

            student_model.train()

            # 输出验证结果
            text += f", Val Acc@{n_neighbors}NN: {accuracy:.2f}%"

            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                text += f" ⭐ Best"

                # 保存最佳模型到指定路径
                best_file_path = os.path.join(config.MODEL_WEIGHTS_DIR, "Extractor_best.pth")
                torch.save(student_model.state_dict(), best_file_path)
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

            # 保存训练好的模型
            if config.test_list and (epoch + 1) in config.test_list:
                # 保存模型到指定路径
                file_name = f"Extractor_{epoch + 1}.pth"
                file_path = os.path.join(config.MODEL_WEIGHTS_DIR, file_name)
                torch.save(student_model.state_dict(), file_path)
                tqdm.write(f"Distilled model saved to {file_path}")

                # 绘制loss折线图
                if config.test_list and (epoch + 1) in config.test_list[-3:]:
                    pic_save_path = os.path.join(config.MODEL_WEIGHTS_DIR, f"loss_{epoch+1}.png")
                    plot_loss_curve(loss_per_epoch, num_epochs, config.net_type, config.preprocess_type, pic_save_path)

    # 打印最佳模型信息
    print(f"\n{'=' * 50}")
    print(f"蒸馏训练完成！最佳模型信息:")
    print(f"  - Epoch: {best_epoch}")
    print(f"  - 验证集准确率：{best_accuracy:.2f}%")
    print(f"  - 保存路径：{os.path.join(config.MODEL_WEIGHTS_DIR, 'Extractor_best.pth')}")
    print(f"{'=' * 50}\n")

    return student_model


def finetune_with_awgn(
    data,
    labels,
    pretrained_model_path,
    snr_range,  # 噪声范围，例如 [0, 30] 表示SNR在0到30dB之间
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,  # 微调时通常使用更小的学习率
    net_type=None,
    preprocess_type=None,
    test_list=None,
    model_dir_path=None
):
    """
    使用加了AWGN的IQ数据对预训练模型进行微调

    :param data: 输入IQ数据
    :param labels: 数据标签
    :param pretrained_model_path: 预训练模型路径
    :param snr_range: SNR范围 [min_snr, max_snr]
    :param batch_size: 批处理大小
    :param num_epochs: 微调轮数
    :param learning_rate: 学习率
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试点列表
    :param model_dir_path: 模型保存路径
    """

    from utils.signal_trans import awgn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    import math

    # 添加噪声到数据
    noisy_data = awgn(data, snr_range)
    noisy_data = generate_spectrogram(noisy_data, preprocess_type)

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        noisy_data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 加载预训练模型
    model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(DEVICE)
    model.train()

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    triplet_loss_fn = TripletLoss(margin=0.1)

    print(
        "\n---------------------\n"
        "Fine-tuning with AWGN\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "SNR Range: {}~{} dB\n"
        "---------------------\n".format(num_epochs, batch_size, snr_range[0], snr_range[1])
    )

    loss_per_epoch = []

    # 训练过程
    with tqdm(total=num_epochs, desc="Total Progress") as total_bar:
        for epoch in range(num_epochs):
            start_time_ep = time.time()
            total_loss = 0.0

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

                    # 计算损失
                    loss = triplet_loss_fn(embedded_anchor, embedded_positive, embedded_negative)

                    # 反向传播与优化
                    optimizer.zero_grad()
                    loss.backward()
                    # 梯度裁剪，防止大的更新
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            # 保存模型
            if test_list and (epoch + 1) in test_list:
                if not os.path.exists(model_dir_path):
                    os.makedirs(model_dir_path)
                file_name = f"Finetuned_Extractor_SNR{snr_range[0]}-{snr_range[1]}dB_{epoch + 1}.pth"
                file_path = os.path.join(model_dir_path, file_name)
                torch.save(model.state_dict(), file_path)
                tqdm.write(f"Fine-tuned model saved to {file_path}")

            total_bar.update(1)
