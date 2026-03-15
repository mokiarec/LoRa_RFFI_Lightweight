"""恶意设备检测模式相关函数"""
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from core.config import Config, Mode
from plot.plot_roc import evaluate_and_plot_roc
from training_utils.data_preprocessor import load_generate_triplet, load_data, generate_spectrogram, load_model


def test_rogue_device_detection(
    config: Config,
    mode: str = None,
    file_path_enrol: str=None,
    dev_range_enrol: np.array=None,
    pkt_range_enrol: np.array=None,
    file_path_legitimate: str=None,
    dev_range_legitimate: np.array=None,
    pkt_range_legitimate: np.array=None,
    file_path_rogue: str=None,
    dev_range_rogue: np.array=None,
    pkt_range_rogue: np.array=None,
    net_type=None,
    preprocess_type=None,
    test_list=None,
    model_dir=None,
    is_pac=False,
    wst_j=None,
    wst_q=None
):
    """
    该函数使用特征提取模型对恶意设备进行检测, 并返回相关的检测结果和性能指标。

    :param mode: 模式，用于确定子目录名称及模型保存路径。
    :param file_path_enrol (str): 注册数据集的路径。
    :param dev_range_enrol (tuple): 注册数据集中设备的范围。
    :param pkt_range_enrol (tuple): 注册数据集中数据包的范围。
    :param file_path_legitimate (str): 合法设备数据集的路径。
    :param dev_range_legitimate (tuple): 合法设备数据集中设备的范围。
    :param pkt_range_legitimate (tuple): 合法设备数据集中数据包的范围。
    :param file_path_rogue (str): 恶意设备数据集的路径。
    :param dev_range_rogue (tuple): 恶意设备数据集中设备的范围。
    :param pkt_range_rogue (tuple): 恶意设备数据集中数据包的范围。
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试列表
    :param model_dir: 模型目录路径
    :param wst_j: WST J参数
    :param wst_q: WST Q参数

    :return fpr (ndarray): 假阳性率。
    :return tpr (ndarray): 真阳性率。
    :return roc_auc (float): ROC 曲线下面积。
    :return eer (float): 等错误率。
    """

    # 子目录路径
    if mode == Mode.ROGUE_DEVICE_DETECTION:
        mode = 'origin'
    model_dir = os.path.join(model_dir, mode)
    print(f"PCA used!!" if is_pac else "PCA not used!!")

    # 投票参数
    vote_size = 10

    # 加载和处理数据
    """
    加载注册设备数据
    """
    print("\nDevice enrolling...")
    # 加载注册数据集(IQ样本和标签)
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol, preprocess_type
    )

    """
    加载合法设备和恶意设备数据
    """

    print("\nData loading...")
    # 加载合法设备和恶意设备数据
    data_legitimate, _ = load_data(
        file_path_legitimate, dev_range_legitimate, pkt_range_legitimate
    )
    data_rogue, _ = load_data(
        file_path_rogue, dev_range_rogue, pkt_range_rogue
    )

    # 合并合法设备和恶意设备数据
    data_test = np.concatenate([data_legitimate, data_rogue])
    label_test = np.concatenate(
        [np.ones(len(data_legitimate)), np.zeros(len(data_rogue))]
    )

    # 提取特征
    data_test = generate_spectrogram(data_test, preprocess_type, wst_j, wst_q)

    # 准备三元组数据
    triplet_data_test = [
        data_test,
        data_test,
        data_test,
    ]
    triplet_label_test = [label_test, label_test, label_test]

    # 将三元组输入转换为张量
    triplet_data_test = [
        torch.tensor(x).float() for x in triplet_data_test
    ]

    for epoch in test_list or []:
        print()
        model_path = os.path.join(model_dir, f"Extractor_{epoch}.pth")

        if not os.path.exists(model_path):
            print(f"{model_path} isn't exist")
        else:
            model = load_model(model_path, net_type, preprocess_type)
            print("Model loaded!!!")

            """
            设备注册
            """

            # 提取特征
            with torch.no_grad():
                feature_enrol = model(*triplet_data_enrol)

            # 构建 K-NN 分类器
            knnclf = KNeighborsClassifier(n_neighbors=15, metric="euclidean")

            if is_pac:
                pca = PCA(n_components=config.PCA_DIM_TEST)
                pca.fit(feature_enrol[0])  # 只用 enrollment 特征
                feature_enrol_pca = pca.transform(feature_enrol[0])  # 投影到低维
                knnclf.fit(feature_enrol_pca, label_enrol.ravel())
            else:
                knnclf.fit(feature_enrol[0], label_enrol.ravel())

            """
            测试恶意设备检测能力
            """

            # 提取特征
            with torch.no_grad():
                feature_test = model(*triplet_data_test)

            print("Device predicting...")
            # 使用 K-NN 分类器进行预测
            if is_pac:
                feature_clf_pca = pca.transform(feature_test[0])
                distances, _ = knnclf.kneighbors(feature_clf_pca)
            else:
                distances, _ = knnclf.kneighbors(feature_test[0])
            detection_score = distances.mean(axis=1)

            # 应用投票机制
            def apply_voting(scores, vote_size):
                """应用滑动窗口投票机制"""
                voted_scores = []
                for i in range(len(scores)):
                    window_start = max(0, i - vote_size // 2)
                    window_end = min(len(scores), i + vote_size // 2 + 1)
                    window = scores[window_start:window_end]
                    voted_score = np.mean(window)
                    voted_scores.append(voted_score)
                return voted_scores

            # 应用投票机制到检测分数
            detection_score_voted = apply_voting(detection_score, vote_size)

            # 调用评估和绘图函数
            fpr, tpr, roc_auc, eer, eer_threshold = evaluate_and_plot_roc(
                label_test, detection_score_voted, epoch
            )

    return