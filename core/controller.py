import os

import numpy as np

# 从配置模块导入配置、设备和模式枚举
from core.config import Config, Mode
from dataset import DATASET

from modes import (
    distillation,
    test_classification,
    test_latency_benchmark,
    test_multi_clf,
    test_rogue_device_detection,
    train,
)

from training_utils.data_preprocessor import prepare_train_data
from utils.PCA import pca_perform, pca_extract_features
from utils.better_print import print_colored_text


def main(config: Config):
    """主函数"""

    # 打印网络类型
    print(f"Running mode: {config.mode}")
    if hasattr(config, 'BASE_MODEL_DIR'):
        print(f"Base Net TYPE: {config.BASE_NET_TYPE}")
    if hasattr(config, 'EXP_NET_TYPE'):
        print(f"Exp Net TYPE: {config.EXP_NET_TYPE}")

    # 使用字典映射替代 if-elif-else 结构
    mode_functions = {
        Mode.TRAIN: run_train_mode,
        Mode.CLASSIFICATION: run_classification_mode,
        Mode.MULTI_CLASSIFICATION: run_multi_clf_mode,
        Mode.ROGUE_DEVICE_DETECTION: run_rogue_device_detection_mode,
        Mode.DISTILLATION: run_distillation_mode,
        Mode.LATENCY_BENCHMARK: run_latency_benchmark_mode,
    }

    # 执行对应模式的函数
    if config.mode in mode_functions:
        mode_functions[config.mode](config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def run_train_mode(config):
    """训练模式"""
    print_colored_text("训练模式", "32")
    print(f"Convert Type: {config.PREPROCESS_TYPE.value}")

    data, labels = prepare_train_data(
        config.new_file_flag,
        config.filename_train_prepared_data,
        path_train_data=DATASET['Train']['no_aug'].path,
        dev_range=np.arange(0, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
        # snr_range=np.arange(20, 80),
        generate_type=config.PREPROCESS_TYPE,
        WST_J=config.WST_J,
        WST_Q=config.WST_Q,
    )

    # 训练特征提取模型
    train(config, data, labels,
          batch_size=config.HP['batch_size'],
          num_epochs=config.HP['num_epochs'],
          learning_rate=config.HP['learning_rate'],
          )


def run_classification_mode(config):
    """分类模式"""
    print_colored_text("分类模式", "32")

    # 执行分类任务
    test_classification(
        config,
        dataset_name=DATASET['Train']['no_aug'].name,
        file_path_enrol=DATASET['Train']['no_aug'].path,
        file_path_clf=DATASET['Test']['seen'].path,
        dev_range_enrol=np.arange(0, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 400, dtype=int),
        dev_range_clf=np.arange(0, 40, dtype=int),
        pkt_range_clf=np.arange(0, 200, dtype=int),
        is_pac=config.IS_PCA_TEST
    )


def run_multi_clf_mode(config):
    """多数据集分类评估模式"""
    print_colored_text("多数据集分类评估", "32")

    # 执行多数据集评估
    test_multi_clf(config)


def run_rogue_device_detection_mode(config):
    """甄别恶意模式"""
    print_colored_text("甄别恶意模式", "32")

    # 执行恶意设备检测任务
    test_rogue_device_detection(
        config,
        file_path_enrol=DATASET['Train']['no_aug'].path,
        dev_range_enrol=np.arange(0, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 200, dtype=int),
        file_path_legitimate=DATASET['Test']['seen'].path,
        dev_range_legitimate=np.arange(0, 40, dtype=int),
        pkt_range_legitimate=np.arange(200, 400, dtype=int),
        file_path_rogue=DATASET['Test']['rogue'].path,
        dev_range_rogue=np.arange(40, 46, dtype=int),
        pkt_range_rogue=np.arange(0, 200, dtype=int),
        wst_j=config.WST_J,
        wst_q=config.WST_Q,
        is_pac=config.IS_PCA_TEST
    )


def run_distillation_mode(config):
    """蒸馏模式"""

    # 模型指定路径
    file_name = f"Extractor_best.pth"
    file_path = os.path.join(config.BASE_MODEL_DIR, "weights", file_name)

    print_colored_text("蒸馏训练模式", "32")

    if config.IS_PCA_TRAIN:
        data, labels = prepare_train_data(
            config.new_file_flag,
            config.filename_train_prepared_data,
            path_train_data=DATASET['Train']['no_aug'].path,
            dev_range=np.arange(0, 40, dtype=int),
            pkt_range=np.arange(0, 800, dtype=int),
            # snr_range=np.arange(20, 80),
            generate_type=config.PREPROCESS_TYPE,
            WST_J=config.WST_J,
            WST_Q=config.WST_Q,
        )

        # 提取教师模型特征
        pca_extract_features(data, labels, batch_size=128,
                             model_path=file_path,
                             output_path=config.PCA_FILE_INPUT,
                             teacher_net_type=config.BASE_NET_TYPE,
                             preprocess_type=config.PREPROCESS_TYPE
                             )
        # 执行 PCA
        pca_perform(
            input_file=config.PCA_FILE_INPUT,
            output_file=config.PCA_FILE_OUTPUT,
            n_components=config.PCA_DIM_TRAIN
        )

    # 准备训练数据
    data, labels = prepare_train_data(
        config.new_file_flag,
        config.filename_train_prepared_data,
        path_train_data=DATASET['Train']['no_aug'].path,
        dev_range=np.arange(0, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
        # snr_range=np.arange(20, 80),
        generate_type=config.PREPROCESS_TYPE,
        WST_J=config.WST_J,
        WST_Q=config.WST_Q,
    )

    # 执行蒸馏训练
    distillation(
        config,
        data,
        labels,
        teacher_model_path=file_path,
        batch_size=config.HP['batch_size'],
        num_epochs=config.HP['num_epochs'],
        learning_rate=config.HP['learning_rate'],
        temperature=config.HP['temperature'],
        alpha=config.HP['alpha'],
        is_pca=config.IS_PCA_TRAIN,
        pca_file_path=config.PCA_FILE_OUTPUT,
    )


def run_latency_benchmark_mode(config):
    """延迟基准测试模式"""
    print_colored_text("延迟基准测试", "32")

    # 执行延迟基准测试
    test_latency_benchmark(
        config,
        file_path_enrol=DATASET['Train']['no_aug'].path,
        file_path_clf=DATASET['Test']['seen'].path,
        dev_range_enrol=np.arange(0, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 200, dtype=int),
        dev_range_clf=np.arange(0, 40, dtype=int),
        pkt_range_clf=np.arange(0, 100, dtype=int),
        is_pac=config.IS_PCA_TEST,
        num_runs=config.HP.get('benchmark_runs', 10),
        enable_warmup=True,
    )
