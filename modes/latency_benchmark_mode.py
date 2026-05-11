# \modes\latency_benchmark_mode.py
"""延迟基准测试模式相关函数"""
import os
import time
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from core.config import Config, DEVICE
from utils.data_preprocessor import load_generate_triplet, load_model
from utils.FLOPs import calculate_flops_and_params


def test_latency_benchmark(
    config: Config,
    file_path_enrol: str = None,
    file_path_clf: str = None,
    dev_range_enrol: np.array = None,
    dev_range_clf: np.array = None,
    pkt_range_enrol: np.array = None,
    pkt_range_clf: np.array = None,
    snr_range=None,
    is_pac=True,
    num_runs=10,
    enable_warmup=True,
):
    """
    多次运行分类测试以统计平均延迟

    :param config: 配置对象
    :param file_path_enrol: 注册数据集的文件路径
    :param file_path_clf: 分类数据集的文件路径
    :param dev_range_enrol: 注册数据集中设备的范围
    :param dev_range_clf: 分类数据集中设备的范围
    :param pkt_range_enrol: 注册数据集中数据包的范围
    :param pkt_range_clf: 分类数据集中数据包的范围
    :param snr_range: 信噪比范围
    :param is_pac: 是否使用 PCA 降维
    :param num_runs: 运行次数
    :param enable_warmup: 是否启用预热
    """

    print(f"\n{'='*60}")
    print(f"延迟基准测试 - {num_runs} 次运行")
    print(f"{'='*60}\n")

    # 加载数据 (只加载一次)
    print("Data loading...")
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        config.preprocess_type, snr_range=snr_range
    )

    label_clf, triplet_data_clf = load_generate_triplet(
        file_path_clf, dev_range_clf, pkt_range_clf,
        config.preprocess_type, snr_range=snr_range
    )
    print("Data loaded!!!\n")

    # 移动数据到设备
    triplet_data_enrol = tuple(tensor.to(DEVICE) for tensor in triplet_data_enrol)
    triplet_data_clf = tuple(tensor.to(DEVICE) for tensor in triplet_data_clf)

    # 存储所有运行的结果
    all_run_results = []

    # 构建待测试的模型列表
    test_epochs = (config.TEST_LIST or []) + ['best_model']

    for epoch in test_epochs:
        print(f"\n{'='*60}")
        print(f"测试模型：Extractor_{epoch}")
        print(f"{'='*60}\n")

        # 构建模型路径
        if isinstance(epoch, int) or (isinstance(epoch, str) and epoch.isdigit()):
            model_path = os.path.join(config.MODEL_WEIGHTS_DIR, f"Extractor_{epoch}.pth")
        elif epoch == 'best_model':
            model_path = os.path.join(config.MODEL_WEIGHTS_DIR, "Extractor_best.pth")

        if not os.path.exists(model_path):
            print(f"{model_path} 不存在，跳过")
            continue

        # 加载模型
        model = load_model(model_path, config.net_type, config.preprocess_type)
        model.to(DEVICE)
        model.eval()
        print("Model loaded!!!\n")

        # 计算 FLOPs 和参数量
        calculate_flops_and_params(model, triplet_data_clf)

        # 预热
        if enable_warmup:
            print("Warming up...")
            with torch.no_grad():
                _ = model(*triplet_data_enrol)
                _ = model(*triplet_data_clf)
            print("Warmup finished.\n")

        # 多次运行收集数据
        enrol_times = []
        clf_feat_times = []
        predict_times = []

        for run_idx in range(num_runs):
            print(f"Run {run_idx + 1}/{num_runs}...", end='\r')

            # 注册集特征提取时间
            with torch.no_grad():
                start_time = time.time()
                feature_enrol = model(*triplet_data_enrol)
                enrol_time = time.time() - start_time

            # 分类集特征提取时间
            with torch.no_grad():
                start_time = time.time()
                feature_clf = model(*triplet_data_clf)
                clf_feat_time = time.time() - start_time

            # 预测时间 (包括 PCA 变换、分类器训练和预测)
            start_time = time.time()

            # PCA 处理
            if is_pac:
                pca = PCA(n_components=config.PCA_DIM_TEST)
                pca.fit(feature_enrol[0].cpu().numpy())
                feature_enrol_pca = pca.transform(feature_enrol[0].cpu().numpy())
                feature_clf_pca = pca.transform(feature_clf[0].cpu().numpy())
                train_data = feature_enrol_pca
                test_data = feature_clf_pca
            else:
                train_data = feature_enrol[0].cpu().numpy()
                test_data = feature_clf[0].cpu().numpy()

            # 训练分类器
            knnclf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
            svmclf = SVC(kernel="rbf", C=1.0)
            knnclf.fit(train_data, label_enrol.ravel())
            svmclf.fit(train_data, label_enrol.ravel())

            # 预测
            pred_label_svm = svmclf.predict(test_data)
            pred_label_knn = knnclf.predict(test_data)

            # 投票机制
            vote_size = 10
            def apply_voting(labels, vote_size):
                voted_labels = []
                for i in range(len(labels)):
                    window_start = max(0, i - vote_size // 2)
                    window_end = min(len(labels), i + vote_size // 2 + 1)
                    window = labels[window_start:window_end]
                    most_common_label = Counter(window).most_common(1)[0][0]
                    voted_labels.append(most_common_label)
                return voted_labels

            pred_label_knn_voted = apply_voting(pred_label_knn, vote_size)
            pred_label_svm_voted = apply_voting(pred_label_svm, vote_size)

            # 综合投票
            weight_knn = 0.5
            weight_svm = 1 - weight_knn
            combined_label = []
            for i in range(0, len(pred_label_knn_voted), vote_size):
                window_end = min(i + vote_size, len(pred_label_knn_voted))
                knn_votes = Counter()
                svm_votes = Counter()
                for j in range(i, window_end):
                    knn_votes[pred_label_knn_voted[j]] += weight_knn
                    svm_votes[pred_label_svm_voted[j]] += weight_svm
                combined_votes = knn_votes + svm_votes
                final_label = combined_votes.most_common(1)[0][0]
                combined_label.extend([final_label] * (window_end - i))

            predict_time = time.time() - start_time

            enrol_times.append(enrol_time)
            clf_feat_times.append(clf_feat_time)
            predict_times.append(predict_time)

        print(f"Run {num_runs}/{num_runs} - 完成!\n")

        # 计算统计数据
        avg_enrol = np.mean(enrol_times)
        std_enrol = np.std(enrol_times)

        avg_clf_feat = np.mean(clf_feat_times)
        std_clf_feat = np.std(clf_feat_times)

        avg_predict = np.mean(predict_times)
        std_predict = np.std(predict_times)

        total_time = avg_enrol + avg_clf_feat + avg_predict
        std_total = np.sqrt(std_enrol**2 + std_clf_feat**2 + std_predict**2)

        # 打印结果
        print(f"\n{'-'*60}")
        print(f"模型：Extractor_{epoch}")
        print(f"{'-'*20}")
        print(f"Enrollment Time:")
        print(f"  Mean: {avg_enrol:.4f} s ± {std_enrol:.4f} s")
        print(f"Classification Feature Extraction Time:")
        print(f"  Mean: {avg_clf_feat:.4f} s ± {std_clf_feat:.4f} s")
        print(f"Prediction Time (PCA + Classification + Voting):")
        print(f"  Mean: {avg_predict:.4f} s ± {std_predict:.4f} s")
        print(f"{'-'*20}")
        print(f"Total Time:")
        print(f"  Mean: {total_time:.4f} s ± {std_total:.4f} s")
        print(f"{'-'*60}\n")

        # 存储结果
        epoch_result = {
            "epoch": epoch,
            "enrol_time_mean": avg_enrol,
            "enrol_time_std": std_enrol,
            "clf_feat_time_mean": avg_clf_feat,
            "clf_feat_time_std": std_clf_feat,
            "predict_time_mean": avg_predict,
            "predict_time_std": std_predict,
            "total_time_mean": total_time,
            "total_time_std": std_total,
        }
        all_run_results.append(epoch_result)

    # 打印汇总统计
    if all_run_results:
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        best_result = min(all_run_results, key=lambda x: x["total_time_mean"])
        print(f"最快模型：Extractor_{best_result['epoch']}")
        print(f"  Total Time: {best_result['total_time_mean']:.4f} s ± {best_result['total_time_std']:.4f} s")
        print(f"{'='*60}\n")

    return all_run_results
