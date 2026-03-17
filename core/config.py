"""配置管理模块"""
import os
import random
import json

import numpy as np
import torch
from enum import Enum

from paths import get_experiment_dir, generate_experiment_name

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义运行模式的枚举
class Mode(str, Enum):
    """运行模式枚举"""
    TRAIN = "train"                    # 训练模式 - 用于训练基础模型
    CLASSIFICATION = "clf"  # 分类模式 - 用于设备指纹分类任务
    MULTI_CLASSIFICATION = "multi_clf"  # 多数据集分类评估模式 - 用于跨场景泛化能力测试
    ROGUE_DEVICE_DETECTION = "rogue"  # 恶意设备检测模式 - 用于检测非法设备
    DISTILLATION = "distill"      # 蒸馏模式 - 用于知识蒸馏训练轻量级模型
    LATENCY_BENCHMARK = "benchmark"   # 延迟基准测试模式 - 用于评估模型推理速度


class DistillateMode(Enum):
    """蒸馏训练模式枚举"""
    ALL = 0              # 执行所有步骤
    ONLY_DISTILLATE = 1  # 仅执行蒸馏
    ONLY_TEST = 2        # 仅执行测试
    ONLY_ROGUE = 3       # 仅执行恶意设备检测


# 定义网络类型枚举
class NetworkType(str, Enum):
    """网络类型枚举"""
    ResNet= "ResNet"               # 残差网络
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
    def __init__(self,
        mode: Mode,
        net_type: NetworkType | None = None,
        **kwargs
    ):
        # 设置模式
        self.mode = mode
        # 网络类型
        self.NET_TYPE = net_type
        # 数据预处理类型
        self.PREPROCESS_TYPE = kwargs.get('preprocess_type', PreprocessType.STFT)
        # PCA设置
        self.IS_PCA_TRAIN = kwargs.get('is_pca_train', True)    # 训练时
        self.IS_PCA_TEST = kwargs.get('is_pca_test', True)      # 测试时
        # 我们需要一个新的训练文件吗？
        self.new_file_flag = kwargs.get('new_file_flag', True)

        # CHECKPOINT列表
        self.TEST_LIST = kwargs.get('test_list', [1, 5, 10, 20, 50, 100, 150, 200, 250, 300])

        # WST参数
        if self.PREPROCESS_TYPE == PreprocessType.WST:
            self.WST_J = kwargs.get('wst_j', 6)
            self.WST_Q = kwargs.get('wst_q', 6)
        else:
            self.WST_J = None
            self.WST_Q = None

        # PCA相关的配置
        self.PCA_DIM_TRAIN = 16  # 训练时PCA的维度
        self.PCA_DIM_TEST = 16  # 测试时PCA的维度

        # 超参数
        self.HP = {
            "batch_size": kwargs.get('batch_size', 16),
            "num_epochs": kwargs.get('num_epochs', max(self.TEST_LIST)),
            "learning_rate": kwargs.get('learning_rate', 1e-3),
            "temperature": kwargs.get('temperature', 3.0),  # 蒸馏温度参数
            "alpha": kwargs.get('alpha', 0.7),  # 蒸馏损失权重参数
            "gamma": kwargs.get('gamma', 0.1),  # 学习率衰减率
            "patience": kwargs.get('patience', 20),  # 早停耐心值
            "triplet_margin": kwargs.get('margin', 1.0),  # 三元组损失 Margin
            "benchmark_runs": kwargs.get('benchmark_runs', 10),  # 基准测试运行次数
        }

        # 实验描述
        self.EXP_DESCRIPTION = kwargs.get('exp_description', 'Base')  # 如 "Base", "Pruning", "FineTune"
        # 基础版本号 (可选，用于继承关系)
        self.BASE_VERSION = kwargs.get('base_version', None)  # 如 "01", "02"

        # 获取基础模型路径
        if 'base_model_dir' in kwargs:
            self.BASE_MODEL_DIR = kwargs['base_model_dir']
        elif self.BASE_VERSION:
            # 如果未直接提供 base_model 但指定了 BASE_VERSION，则自动查找对应目录
            exp_prefix = f"EXP_{self.BASE_VERSION}"
            parent_dir = get_experiment_dir(None)
            # 在父目录下查找以 EXP_{BASE_VERSION} 开头的目录
            matching_dirs = [d for d in os.listdir(parent_dir) if d.startswith(exp_prefix) and os.path.isdir(os.path.join(parent_dir, d))]
            if matching_dirs:
                # 取第一个匹配项（可按需调整排序逻辑，例如按时间排序）
                self.BASE_MODEL_DIR = os.path.join(parent_dir, matching_dirs[0])
            else:
                self.BASE_MODEL_DIR = None
        else:
            self.BASE_MODEL_DIR = None
        # 从 BASE_MODEL_DIR 解析基础网络类型 (例如从 ".../EXP_02_ResNet_Base" 中提取 "ResNet")
        if self.BASE_MODEL_DIR:
            dir_name = os.path.basename(self.BASE_MODEL_DIR)
            parts = dir_name.split('_')
            # 假设目录命名格式为 EXP_{version}_{NetType}_{Description}
            if len(parts) >= 3:
                net_type_str = parts[2]
                # 尝试匹配 NetworkType 枚举，防止大小写或命名不一致
                try:
                    self.BASE_NET_TYPE = NetworkType(net_type_str)
                except ValueError:
                    # 如果直接匹配失败，尝试不区分大小写或常见变体匹配（可选增强鲁棒性）
                    for member in NetworkType:
                        if member.value.lower() == net_type_str.lower():
                            self.BASE_NET_TYPE = member
                            break
                    else:
                        # 若仍未找到，默认使用 ResNet 或抛出警告/设为 None，此处设为 None 并打印警告
                        print(f"Warning: 无法识别的基础网络类型 '{net_type_str}'，源自目录 '{dir_name}'")
                        self.BASE_NET_TYPE = None
            else:
                print(f"Warning: 基础模型目录名称格式不符合预期: '{dir_name}'")
                self.BASE_NET_TYPE = None
        else:
            self.BASE_NET_TYPE = None
        # 生成模型路径
        # 仅当提供了参数时，才生成实验名称和目录
        if 'exp_name' in kwargs:
            self.EXP_NAME = kwargs['exp_name']
        else:
            self.EXP_NAME = generate_experiment_name(
                self.NET_TYPE,
                self.EXP_DESCRIPTION,
                base_version=self.BASE_VERSION
            )
        # 获取实验模型路径
        self.MODEL_DIR, self.MODEL_WEIGHTS_DIR, self.MODEL_EVAL_DIR = get_experiment_dir(self.EXP_NAME)

        # 新生成数据集文件路径
        if self.PREPROCESS_TYPE == PreprocessType.STFT:
            self.filename_train_prepared_data = f"train_data_{self.PREPROCESS_TYPE.value}.h5"
        elif self.PREPROCESS_TYPE == PreprocessType.WST:
            self.filename_train_prepared_data = f"train_data_{self.PREPROCESS_TYPE.value}_j{self.WST_J}q{self.WST_Q}.h5"

        # PCA 相关的路径 (仅在需要时使用)
        if self.IS_PCA_TRAIN:
            self.PCA_DATA_DIR = os.path.join(self.MODEL_WEIGHTS_DIR, "pca_results")
            self.PCA_FILE_INPUT = os.path.join(self.PCA_DATA_DIR, "teacher_feats.npz")
            self.PCA_FILE_OUTPUT = os.path.join(self.PCA_DATA_DIR, f"pca_{self.PCA_DIM_TEST}.npz")
        else:
            self.PCA_DATA_DIR = None
            self.PCA_FILE_INPUT = None
            self.PCA_FILE_OUTPUT = None

        # 确保目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保模型目录存在"""
        directories = []

        if hasattr(self, 'MODEL_DIR') and self.MODEL_DIR:
            directories.append(str(self.MODEL_DIR))
        if hasattr(self, 'MODEL_WEIGHTS_DIR') and self.MODEL_WEIGHTS_DIR:
            directories.append(str(self.MODEL_WEIGHTS_DIR))
        if hasattr(self, 'MODEL_EVAL_DIR') and self.MODEL_EVAL_DIR:
            directories.append(str(self.MODEL_EVAL_DIR))
        if hasattr(self, 'PCA_DATA_DIR') and self.PCA_DATA_DIR:
            directories.append(str(self.PCA_DATA_DIR))

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def to_dict(self):
        """将所有配置转换为可JSON序列化的字典"""
        data = {}
        for key, value in self.__dict__.items():
            # 处理枚举类型
            if isinstance(value, Enum):
                data[key] = value.name  # 存储枚举名称，如 "TRAIN"
            # 处理 Path 对象 (如果有的话)
            elif hasattr(value, '__fspath__'):
                data[key] = str(value)
            # 处理基本类型
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                data[key] = value
        return data

    def save_to_json(self):
        """保存配置到实验目录"""
        # 根据目录结构，存放在实验根目录
        save_path = os.path.join(self.MODEL_DIR, "config.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        print(f"--- config 已备份至: {save_path} ---")

    @classmethod
    def from_json(cls, mode: Mode, model_dir):
        """从已有的目录重新构造 Config 对象"""
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 准备解包参数
        kwargs = {}
        if 'PREPROCESS_TYPE' in data:
            kwargs['preprocess_type'] = PreprocessType[data['PREPROCESS_TYPE']]
        if 'DISTILLATE_MODE' in data:
            kwargs['distillate_mode'] = DistillateMode[data['DISTILLATE_MODE']]

        # 网络类型
        net_type = NetworkType[data['NET_TYPE']] if 'NET_TYPE' in data else None

        # 描述和版本，确保重新构造时路径一致
        kwargs['test_list'] = data.get('TEST_LIST', [1, 5, 10, 20, 50, 100, 150, 200, 250, 300])
        kwargs['model_dir'] = data.get('MODEL_DIR', model_dir)
        kwargs['exp_name'] = data.get('EXP_NAME', None)
        kwargs['exp_description'] = data.get('EXP_DESCRIPTION', 'Base')
        kwargs['base_version'] = data.get('BASE_VERSION', None)
        kwargs['is_pca_train'] = data.get('IS_PCA_TRAIN', True)
        kwargs['is_pca_test'] = data.get('IS_PCA_TEST', True)


        # 重新初始化对象
        return cls(mode, net_type=net_type, **kwargs)

def set_seed(seed=42):
    """设置随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
