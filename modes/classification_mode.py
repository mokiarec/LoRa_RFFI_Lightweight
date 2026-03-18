# \modes\classfication_mode.py
"""分类模式相关函数"""
import os
import time
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from core.config import Config  # Config枚举型
from plot.plot_confusion import plot_confusion_matrices
from training_utils.data_preprocessor import load_generate, load_model
# 工具包
from utils.FLOPs import calculate_flops_and_params
from utils.PCA import plot_pca_scree
from utils.better_print import TextAnimator


def test_classification(
        config: Config,
        dataset_name,
        file_path_enrol: str = None,
        file_path_clf: str = None,
        dev_range_enrol: np.ndarray = None,
        dev_range_clf: np.ndarray = None,
        pkt_range_enrol: np.ndarray = None,
        pkt_range_clf: np.ndarray = None,
        snr_range=None,
        is_pac=False,
        enable_plots=True,
):
    """
    * 使用给定的特征提取模型(从指定路径加载)对注册数据集和分类数据集进行分类测试。

    :param file_path_enrol (str): 注册数据集的文件路径。
    :param file_path_clf (str): 分类数据集的文件路径。
    :param dev_range_enrol: 注册数据集中设备的范围。
    :param dev_range_clf: 分类数据集中设备的范围。
    :param pkt_range_enrol: 注册数据集中数据包的范围。
    :param pkt_range_clf: 分类数据集中数据包的范围。
    :param is_pac: 是否使用PAC降维
    :param enable_plots: 控制是否绘制混淆矩阵（默认为True）
    """

    try:
        import swanlab
        SWANLAB_AVAILABLE = True
    except ImportError:
        SWANLAB_AVAILABLE = False

    # 加载数据

    vote_size = 10
    weight_knn = 0.5
    weight_svm = 1 - weight_knn

    """
    提取设备特征
    """

    # 加载注册数据集(IQ样本和标签)
    print("\nData loading...")
    label_enrol, data_enrol = load_generate(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        config.PREPROCESS_TYPE, snr_range=snr_range
    )

    # 加载分类数据集(IQ样本和标签)
    label_clf, data_clf = load_generate(
        file_path_clf, dev_range_clf, pkt_range_clf,
        config.PREPROCESS_TYPE, snr_range=snr_range
    )
    print("\nData loaded!!!")

    print(f"PCA used!!" if is_pac else "PCA not used!!")

    # 存储所有 epoch 的结果用于汇总
    all_epoch_results = []

    # 构建待测试的模型列表：包含 config.TEST_LIST 中的数字编号以及 'best_model'
    test_epochs = (config.TEST_LIST or []) + ['best_model']

    for epoch in test_epochs:
        print()
        print("=============================")

        # 根据 epoch 类型构建模型路径
        if epoch == 'best_model':
            model_path = os.path.join(config.MODEL_WEIGHTS_DIR, "Extractor_best.pth")
        else:
            model_path = os.path.join(config.MODEL_WEIGHTS_DIR, f"Extractor_{epoch}.pth")

        if not os.path.exists(model_path):
            print(f"{model_path} isn't exist")
        else:
            model = load_model(model_path, config.NET_TYPE, config.PREPROCESS_TYPE)
            print("Model loaded!!!")

            # 计算FLOPs和参数量
            calculate_flops_and_params(model, data_clf)

            # 提取特征
            # print("Feature extracting...")
            try:
                text = TextAnimator("Feature extracting", "Feature extracted")
                text.start()
                with torch.no_grad():
                    start_time = time.time()
                    feature_enrol = model.predict(data_enrol)
                    enrol_feature_extraction_time = time.time() - start_time

                # 碎石图绘制逻辑
                plot_scree = False
                scree_max_components = 64
                if plot_scree:
                    plot_pca_scree(feats=feature_enrol, max_components=scree_max_components)

                if is_pac:
                    pca = PCA(n_components=config.PCA_DIM_TEST)
                    pca.fit(feature_enrol)  # 只用 enrollment 特征
                    feature_enrol_pca = pca.transform(feature_enrol)  # 投影到低维

                # 使用 K-NN 分类器进行训练
                knnclf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
                # 使用 SVM 分类器进行训练
                svmclf = SVC(kernel="rbf", C=1.0)  # 可以根据需要调整参数
                if is_pac:
                    svmclf.fit(feature_enrol_pca, label_enrol.ravel())
                    knnclf.fit(feature_enrol_pca, label_enrol.ravel())
                else:
                    svmclf.fit(feature_enrol, label_enrol.ravel())
                    knnclf.fit(feature_enrol, label_enrol.ravel())
            finally:
                text.stop()

            """
            进行预测
            """

            # print("Device predicting...")
            try:
                text = TextAnimator("Device predicting", "Device prediction finish")
                text.start()

                # 提取分类数据集的特征
                with torch.no_grad():
                    start_time = time.time()
                    feature_clf = model.predict(data_clf)
                    clf_feature_extraction_time = time.time() - start_time

                start_time = time.time()

                # K-NN和SVM的初步预测
                if is_pac:
                    # SVM和KNN 投影到 PCA
                    feature_clf_pca = pca.transform(feature_clf)
                    pred_label_svm_wo = svmclf.predict(feature_clf_pca)
                    pred_label_knn_wo = knnclf.predict(feature_clf_pca)
                else:
                    pred_label_svm_wo = svmclf.predict(feature_clf)
                    pred_label_knn_wo = knnclf.predict(feature_clf)

                def apply_voting(labels, vote_size):
                    """应用滑动窗口投票机制"""
                    voted_labels = []
                    for i in range(len(labels)):
                        window_start = max(0, i - vote_size // 2)
                        window_end = min(len(labels), i + vote_size // 2 + 1)
                        window = labels[window_start:window_end]
                        most_common_label = Counter(window).most_common(1)[0][0]
                        voted_labels.append(most_common_label)
                    return voted_labels

                # 应用投票机制
                pred_label_knn_w_v = apply_voting(pred_label_knn_wo, vote_size)
                pred_label_svm_w_v = apply_voting(pred_label_svm_wo, vote_size)

                # 综合投票机制
                combined_label = []
                for i in range(0, len(pred_label_knn_w_v), vote_size):
                    window_end = min(i + vote_size, len(pred_label_knn_w_v))

                    knn_votes = Counter()
                    svm_votes = Counter()

                    for j in range(i, window_end):
                        knn_votes[pred_label_knn_w_v[j]] += weight_knn
                        svm_votes[pred_label_svm_w_v[j]] += weight_svm
                    combined_votes = knn_votes + svm_votes
                    final_label = combined_votes.most_common(1)[0][0]

                    # 保持与原样本相同的长度
                    combined_label.extend([final_label] * (window_end - i))

                # 计算各分类器的准确率
                wo_acc_knn = accuracy_score(label_clf, pred_label_knn_wo)
                wo_acc_svm = accuracy_score(label_clf, pred_label_svm_wo)
                w_acc_knn = accuracy_score(label_clf, pred_label_knn_w_v)
                w_acc_svm = accuracy_score(label_clf, pred_label_svm_w_v)
                acc_combined = accuracy_score(label_clf, combined_label)
                wo_accs = [wo_acc_knn, wo_acc_svm]
                w_accs = [w_acc_knn, w_acc_svm, acc_combined]
                wwo_accs = [wo_accs, w_accs]

            finally:
                prediction_time = time.time() - start_time
                text.stop()

            print("-----------------------------")
            print(f"Extractor ID: {epoch}")
            print(f"Enroll Feature extraction time: {enrol_feature_extraction_time:.4f}s")
            print(f"Classification feature extraction time: {clf_feature_extraction_time:.4f}s")
            print(f"Prediction time: {prediction_time:.4f}s")
            print(f"Vote Size: {vote_size}")
            print(
                f"KNN accuracy\t\tw/o\tvoting = {wo_acc_knn * 100:.2f}%\n"
                f"SVM accuracy\t\tw/o\tvoting = {wo_acc_svm * 100:.2f}%\n"
                f"KNN accuracy\t\tw/\tvoting = {w_acc_knn * 100:.2f}%\n"
                f"SVM accuracy\t\tw/\tvoting = {w_acc_svm * 100:.2f}%\n"
                f"Combined accuracy\tw/\tweighted voting = {acc_combined * 100:.2f}%",
            )
            print("-----------------------------")
            print()

            # 绘制混淆矩阵
            conf_mat_knn_wo = confusion_matrix(label_clf, pred_label_knn_wo)
            conf_mat_svm_wo = confusion_matrix(label_clf, pred_label_svm_wo)
            conf_mat_knn_w = confusion_matrix(label_clf, pred_label_knn_w_v)
            conf_mat_svm_w = confusion_matrix(label_clf, pred_label_svm_w_v)
            conf_mat_combined = confusion_matrix(label_clf, combined_label)
            wo_cms = [conf_mat_knn_wo, conf_mat_svm_wo]
            w_cms = [conf_mat_knn_w, conf_mat_svm_w, conf_mat_combined]
            wwo_cms = [wo_cms, w_cms]

            suffix = "_pca" if is_pac else ""
            if enable_plots:
                # 确保保存目录存在
                save_path = os.path.join(config.MODEL_EVAL_DIR, f"{dataset_name}{suffix}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plot_confusion_matrices(wwo_cms, wwo_accs, epoch, config.NET_TYPE, config.PREPROCESS_TYPE, vote_size,
                                        save_path)

            # 记录到 SwanLab
            if SWANLAB_AVAILABLE:
                # 记录准确率指标
                swanlab.log({
                    f"knn_wo_accuracy{suffix}/{dataset_name}": wo_acc_knn * 100,
                    # f"svm_wo_accuracy/{dataset_name}": wo_acc_svm * 100,
                    f"knn_w_accuracy{suffix}/{dataset_name}": w_acc_knn * 100,
                    # f"svm_w_accuracy/{dataset_name}": w_acc_svm * 100,
                    # f"combined_accuracy/{dataset_name}": acc_combined * 100,
                }, step=epoch)

            # T-SNE 3D绘图
            # tsne_3d_plot(feature_clf,labels=label_clf)
        print("=============================")

    return {
        "knn_wo": wo_acc_knn,
        "svm_wo": wo_acc_svm,
        "knn_w": w_acc_knn,
        "svm_w": w_acc_svm,
        "combined": acc_combined,
    }
