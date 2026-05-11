# utils/PCA.py
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from core.config import Config, DEVICE, Mode, PreprocessType
from net import NetworkType, TripletNet
from utils.TripletDataset import TripletDataset
from utils.data_preprocessor import prepare_train_data


def pca_extract_features(
        data,
        labels,
        batch_size,
        model_path,
        output_path,
        teacher_net_type,
        preprocess_type,
):
    """
    使用预训练的教师模型提取数据特征并保存
    
    :param data: 输入数据
    :param labels: 数据标签
    :param batch_size: 批处理大小
    :param model_path: 预训练模型路径
    :param output_path: 特征输出保存路径
    :param teacher_net_type: 教师网络类型
    :param preprocess_type: 预处理类型配置对象，需包含in_channels属性
    :return: None
    """

    # 生成数据加载器
    dataset = TripletDataset(data, labels, preprocess_type)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    teacher_model = TripletNet(net_type=teacher_net_type, in_channels=preprocess_type.in_channels)  # MobileNet
    teacher_model.load_state_dict(torch.load(model_path))
    teacher_model.to(DEVICE)
    teacher_model.eval()

    feats = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(loader):
            anchor, positive, negative = (
                anchor.to(DEVICE),
                positive.to(DEVICE),
                negative.to(DEVICE),
            )
            # teacher 输出高维特征
            teacher_anchor, teacher_positive, teacher_negative = teacher_model(
                anchor, positive, negative
            )  # shape (B, D_t)

            # 保存对应 label（anchor label 就可以）
            feats.append(teacher_anchor.cpu().numpy())
            labels_list.append(labels[batch_idx * batch_size: (batch_idx + 1) * batch_size])  # 或者 anchor labels

    feats = np.concatenate(feats, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    np.savez(output_path, feats=feats, labels=labels_arr)
    print("Teacher features saved, shape:", feats.shape)


def pca_perform(input_file="teacher_feats.npz", output_file="pca_16.npz", n_components=16):
    """
    对输入特征进行PCA降维处理
    
    :param input_file: 输入的特征文件路径
    :param output_file: 输出的PCA参数文件路径
    :param n_components: 降维后的维度数
    :return: components, mean
    """
    
    # 加载数据
    data = np.load(input_file)
    feats = data['feats']
    labels = data['labels']
    
    # 执行PCA
    pca = PCA(n_components=n_components)
    pca.fit(feats)
    components = pca.components_  # d x D_t
    mean = pca.mean_
    
    # 保存结果
    np.savez(output_file, components=components, mean=mean)
    print("PCA done, shape:", components.shape)
    
    return components, mean


def plot_pca_scree(input_file="teacher_feats.npz", max_components=None, save_path=None, feats=None):
    """
    绘制PCA碎石图

    :param input_file: 输入的特征文件路径
    :param max_components: 最大显示的主成分数量，默认为特征维度数
    :param save_path: 图片保存路径，如果提供则保存图片
    :param feats: 特征数据（可选，如果不提供则从input_file加载）
    :return: None
    """

    # 加载数据
    if feats is None:
        data = np.load(input_file)
        feats = data['feats']

    # 如果未指定最大组件数，则使用特征数
    if max_components is None:
        max_components = min(feats.shape[0], feats.shape[1])

    # 执行PCA
    pca = PCA(n_components=max_components)
    pca.fit(feats)

    # 获取方差解释比例
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # 绘制碎石图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1: 各主成分的方差解释比例
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot - Explained Variance by Component')
    ax1.grid(True, alpha=0.3)

    # 子图2: 累积方差解释比例
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    ax2.legend()

    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA scree plot saved to {save_path}")
    
    plt.show()

    # 打印前20个主成分的方差解释比例
    print("Top 20 Principal Components:")
    for i in range(min(20, len(explained_variance_ratio))):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} "
              f"({cumulative_variance_ratio[i]:.4f} cumulative)")

if __name__ == '__main__':
    config = Config(Mode.TEST)

    # 准备训练数据
    data, labels = prepare_train_data(
        config.new_file_flag,
        config.filename_train_prepared_data,
        path_train_data="../dataset/Train/dataset_training_no_aug.h5",
        dev_range=np.arange(0, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
        snr_range=np.arange(20, 80),
        generate_type=config.preprocess_type,
        WST_J=config.WST_J,
        WST_Q=config.WST_Q,
    )

    # 提取特征
    pca_extract_features(
        data, labels,
        batch_size=128,
        model_path="../model/stft/ResNet/origin/Extractor_200.pth",
        output_path="../model/stft/ResNet_no_aug/distilled/pca_results/teacher_feats_origintrain.npz",
        teacher_net_type=NetworkType.ResNet,
        preprocess_type=PreprocessType.STFT
    )

    # 执行PCA
    # pca_perform()

    # 绘制碎石图
    plot_pca_scree(input_file="../model/stft/ResNet_no_aug/distilled/pca_results/teacher_feats_origintrain.npz",
                   max_components=16,
                   save_path="../model/stft/ResNet_no_aug/distilled/pca_results/pca_scree_origintrain.png"
                   )