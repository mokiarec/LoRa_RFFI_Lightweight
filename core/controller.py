import numpy as np

# 从配置模块导入配置、设备和模式枚举
from core.config import Config, Mode, DistillateMode
from modes.classification_mode import test_classification
from modes.distillation_mode import distillation
from modes.rogue_device_detection_mode import test_rogue_device_detection
from modes.train_mode import train
from paths import get_dataset_path
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
        Mode.ROGUE_DEVICE_DETECTION: run_rogue_device_detection_mode,
        Mode.DISTILLATION: run_distillation_mode,
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
        path_train_original_data=get_dataset_path('train_no_aug'),
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
        file_path_enrol=get_dataset_path('train_aug'),
        file_path_clf=get_dataset_path('test_seen'),
        dev_range_enrol=np.arange(0, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 400, dtype=int),
        dev_range_clf=np.arange(0, 40, dtype=int),
        pkt_range_clf=np.arange(0, 200, dtype=int),
        is_pac=config.IS_PCA_TEST
    )


def run_rogue_device_detection_mode(config):
    """甄别恶意模式"""
    print_colored_text("甄别恶意模式", "32")

    # 执行恶意设备检测任务
    test_rogue_device_detection(
        config.mode,
        file_path_enrol=get_dataset_path('train_no_aug'),
        dev_range_enrol=np.arange(0, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 200, dtype=int),
        file_path_legitimate=get_dataset_path('test_seen'),
        dev_range_legitimate=np.arange(0, 40, dtype=int),
        pkt_range_legitimate=np.arange(200, 400, dtype=int),
        file_path_rogue=get_dataset_path('test_rogue'),
        dev_range_rogue=np.arange(40, 46, dtype=int),
        pkt_range_rogue=np.arange(0, 200, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PREPROCESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir=config.ORIGIN_MODEL_DIR_PATH,
        wst_j=config.WST_J,
        wst_q=config.WST_Q,
        is_pac=config.IS_PCA_TEST
    )


def run_distillation_mode(config):
    """蒸馏模式"""

    if config.DISTILLATE_MODE in [DistillateMode.ALL, DistillateMode.ONLY_DISTILLATE]:
        print_colored_text("蒸馏训练模式", "32")

        if config.IS_PCA_TRAIN:
            data, labels = prepare_train_data(
                config.new_file_flag,
                config.filename_train_prepared_data,
                path_train_original_data=get_dataset_path('train_no_aug'),
                dev_range=np.arange(0, 40, dtype=int),
                pkt_range=np.arange(0, 800, dtype=int),
                snr_range=np.arange(20, 80),
                generate_type=config.PREPROCESS_TYPE,
                WST_J=config.WST_J,
                WST_Q=config.WST_Q,
            )

            # 提取教师模型特征
            pca_extract_features(data, labels, batch_size=128,
                             model_path=config.TEACHER_MODEL_PATH,
                             output_path=config.PCA_FILE_INPUT,
                             teacher_net_type=config.TEACHER_NET_TYPE,
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
            path_train_original_data=get_dataset_path('train_no_aug'),
            dev_range=np.arange(0, 40, dtype=int),
            pkt_range=np.arange(0, 800, dtype=int),
            snr_range=np.arange(20, 80),
            generate_type=config.PREPROCESS_TYPE,
            WST_J=config.WST_J,
            WST_Q=config.WST_Q,
        )

        # 执行蒸馏训练
        distillation(
            config.mode,
            data,
            labels,
            teacher_model_path=config.TEACHER_MODEL_PATH,
            num_epochs=max(config.TEST_LIST),
            temperature=3.0,
            alpha=0.7,
            teacher_net_type=config.TEACHER_NET_TYPE,
            student_net_type=config.STUDENT_NET_TYPE,
            preprocess_type=config.PREPROCESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir=config.STUDENT_MODEL_DIR,
            is_pca=config.IS_PCA_TRAIN,
            pca_file_path=config.PCA_FILE_OUTPUT,
        )

    if config.DISTILLATE_MODE in [DistillateMode.ALL, DistillateMode.ONLY_TEST]:
        print_colored_text("蒸馏后的分类模式", "32")

        # 执行分类测试
        test_classification(
            config.mode,
            file_path_enrol="dataset/DATA_all_dev_1~11_499times_433m_1M_3gain.h5",
            file_path_clf="dataset/DATA_all_dev_1~11_499times_433m_1M_3gain.h5",
            dev_range_enrol=np.arange(0, 40, dtype=int),
            pkt_range_enrol=np.arange(0, 300, dtype=int),
            dev_range_clf=np.arange(0, 40, dtype=int),
            pkt_range_clf=np.arange(300, 400, dtype=int),
            net_type=config.STUDENT_NET_TYPE,
            preprocess_type=config.PREPROCESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir=config.STUDENT_MODEL_DIR,
            is_pac=config.IS_PCA_TEST,
        )

    if config.DISTILLATE_MODE == [DistillateMode.ALL, DistillateMode.ONLY_ROGUE]:
        print_colored_text("蒸馏后的甄别恶意模式", "32")

        # 执行恶意设备检测任务
        test_rogue_device_detection(
            config.mode,
            file_path_enrol=get_dataset_path('train_aug'),
            dev_range_enrol=np.arange(0, 30, dtype=int),
            pkt_range_enrol=np.arange(0, 400, dtype=int),
            file_path_legitimate=get_dataset_path('test_seen'),
            dev_range_legitimate=np.arange(0, 30, dtype=int),
            pkt_range_legitimate=np.arange(0, 200, dtype=int),
            file_path_rogue=get_dataset_path('test_rogue'),
            dev_range_rogue=np.arange(40, 46, dtype=int),
            pkt_range_rogue=np.arange(0, 200, dtype=int),
            net_type=config.STUDENT_NET_TYPE,
            preprocess_type=config.PREPROCESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir=config.STUDENT_MODEL_DIR,
            wst_j=config.WST_J,
            wst_q=config.WST_Q,
            is_pac=config.IS_PCA_TEST
        )
