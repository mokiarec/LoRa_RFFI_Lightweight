"""配置管理模块"""
import os
import random

import numpy as np
import torch
from enum import Enum

from paths import get_model_path

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PCA相关的配置
PCA_DIM_TRAIN = 16  # 训练时PCA的维度
PCA_DIM_TEST = 16    # 测试时PCA的维度


# 定义运行模式的枚举
class Mode(str, Enum):
    """运行模式枚举"""
    TRAIN = "train"                    # 训练模式 - 用于训练基础模型
    CLASSIFICATION = "classification"  # 分类模式 - 用于设备指纹分类任务
    ROGUE_DEVICE_DETECTION = "rogue_device_detection"  # 恶意设备检测模式 - 用于检测非法设备
    DISTILLATION = "distillation"      # 蒸馏模式 - 用于知识蒸馏训练轻量级模型


class DistillateMode(Enum):
    """蒸馏训练模式枚举"""
    ALL = 0              # 执行所有步骤
    ONLY_DISTILLATE = 1  # 仅执行蒸馏
    ONLY_TEST = 2        # 仅执行测试
    ONLY_ROGUE = 3       # 仅执行恶意设备检测


# 定义网络类型枚举
class NetworkType(str, Enum):
    """网络类型枚举"""
    RESNET = "ResNet"               # 残差网络
    DRSN = "Drsn"                   # 深度残差网路
    MobileNetV1 = "MobileNetV1"     # MobileNetV1网络
    MobileNetV2 = "MobileNetV2"     # MobileNetV2网络
    LightNet = "LightNet"           # MobileNetV1改进网络


# 定义预处理类型枚举
class PreprocessType(Enum):
    """预处理类型枚举"""
    IQ = ("IQ", 2)      # IQ数据直接使用，2个通道（I和Q）
    STFT = ("STFT", 1)  # 短时傅里叶变换，1个通道（幅度）
    WST = ("WST", 2)    # 小波散射变换，2个通道（实部和虚部）

    def __init__(self, name, in_channels):
        self._value_ = name
        self.in_channels = in_channels


# 配置类，用于存储全局配置参数
class Config:
    def __init__(self, mode, **kwargs):
        # 设置模式
        self.mode = mode
        # 设置网络类型
        self.NET_TYPE = kwargs.get('net_type', NetworkType.LightNet)
        # 教师网络类型, 学生网络类型
        self.TEACHER_NET_TYPE = kwargs.get('teacher_net_type', NetworkType.RESNET)
        self.STUDENT_NET_TYPE = kwargs.get('student_net_type', NetworkType.LightNet)
        # 数据预处理类型
        self.PREPROCESS_TYPE = kwargs.get('preprocess_type', PreprocessType.STFT)
        # 蒸馏训练模式: [ALL, ONLY_DISTILLATE, ONLY_TEST, ONLY_ROGUE]
        self.DISTILLATE_MODE = kwargs.get('distillate_mode', DistillateMode.ONLY_TEST)
        # PCA设置
        self.IS_PCA_TRAIN = kwargs.get('is_pca_train', True)    # 训练时
        self.IS_PCA_TEST = kwargs.get('is_pca_test', True)      # 测试时
        # 我们需要一个新的训练文件吗？
        self.new_file_flag = kwargs.get('new_file_flag', True)

        # CHECKPOINT列表
        self.TEST_LIST = kwargs.get('test_list', [1, 5, 10, 20, 50, 100, 150, 200, 250, 300])
        # self.TEST_LIST = kwargs.get('test_list', [200])

        # WST参数
        if self.PREPROCESS_TYPE == PreprocessType.WST:
            self.WST_J = kwargs.get('wst_j', 6)
            self.WST_Q = kwargs.get('wst_q', 6)
        else:
            self.WST_J = None
            self.WST_Q = None


        # 生成模型路径
        self.ORIGIN_MODEL_DIR_PATH = get_model_path(self.PREPROCESS_TYPE, self.NET_TYPE, wst_j=self.WST_J, wst_q=self.WST_Q)
        self.TEACHER_MODEL_DIR = get_model_path(self.PREPROCESS_TYPE, self.TEACHER_NET_TYPE, wst_j=self.WST_J, wst_q=self.WST_Q)
        self.STUDENT_MODEL_DIR = get_model_path(self.PREPROCESS_TYPE, self.STUDENT_NET_TYPE, wst_j=self.WST_J, wst_q=self.WST_Q)

        # 新生成数据集文件路径
        if self.PREPROCESS_TYPE == PreprocessType.STFT:
            self.filename_train_prepared_data = f"train_data_{self.PREPROCESS_TYPE.value}.h5"
        elif self.PREPROCESS_TYPE == PreprocessType.WST:
            self.filename_train_prepared_data = f"train_data_{self.PREPROCESS_TYPE.value}_j{self.WST_J}q{self.WST_Q}.h5"

        # PCA相关的路径
        self.PCA_DATA_DIR = os.path.join(self.TEACHER_MODEL_DIR, "pca_results")
        self.PCA_FILE_INPUT = os.path.join(self.PCA_DATA_DIR, "teacher_feats.npz")
        self.PCA_FILE_OUTPUT = os.path.join(self.PCA_DATA_DIR, "pca_16.npz")

        # 确保目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保模型目录存在"""
        directories = [self.ORIGIN_MODEL_DIR_PATH, self.TEACHER_MODEL_DIR, self.STUDENT_MODEL_DIR, self.PCA_DATA_DIR]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)


def set_seed(seed=42):
    """设置随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
