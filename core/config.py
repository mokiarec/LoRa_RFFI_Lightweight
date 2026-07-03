"""配置管理模块"""
import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch

from core import Mode, PreprocessType
from net import NetworkType
from paths import PathManager, ExperimentInfo

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 是否禁用 SwanLab
DISABLE_SWANLAB = os.getenv("DISABLE_SWANLAB", "true").lower() in ("true", "1", "yes")


# 配置类，用于存储全局配置参数
class Config:
    def __init__(self, **kwargs):
        # SwanLab 配置
        self.disable_swanlab = DISABLE_SWANLAB

        # ===== 核心参数 =====
        self.mode: Mode = kwargs.get('mode')  # 设置模式
        self.net_type: Optional[NetworkType] = kwargs.get('net_type', None)  # 网络类型
        self.preprocess_type: PreprocessType = kwargs.get('preprocess_type', PreprocessType.STFT)  # 数据预处理类型
        # PCA设置
        self.is_pca_train: bool = kwargs.get('is_pca_train', True)  # 训练时
        self.is_pca_test: bool = kwargs.get('is_pca_test', True)  # 测试时
        # WST参数
        self.WST_J: Optional[int] = kwargs.get('wst_j', None)
        self.WST_Q: Optional[int] = kwargs.get('wst_q', None)
        # PCA相关的配置
        self.PCA_DIM_TRAIN: int = kwargs.get('pca_dim_train', 8)  # 训练时PCA的维度
        self.PCA_DIM_TEST: int = kwargs.get('pca_dim_test', 8)  # 测试时PCA的维度

        # ===== 实验配置 =====
        self.exp_description: str = kwargs.get('exp_description', 'Base')  # 实验描述
        self.base_version: Optional[str] = kwargs.get('base_version', None)  # 继承的实验编号
        # 我们需要一个新的训练文件吗？
        self.new_file_flag: bool = kwargs.get('new_file_flag', True)
        # CHECKPOINT列表
        self.test_list: List = kwargs.get('test_list', [1, 5, 10, 20, 35, 50, 60, 70, 85, 100, 150, 200, 250, 300])

        # 超参数
        self.HP = {
            "batch_size": kwargs.get('batch_size', 16),
            "num_epochs": kwargs.get('num_epochs', max(self.test_list)),
            "learning_rate": kwargs.get('learning_rate', 1e-3),
            "temperature": kwargs.get('temperature', 3.0),  # 蒸馏温度参数
            "alpha": kwargs.get('alpha', 1.0),  # 蒸馏损失权重参数
            "triplet_margin": kwargs.get('margin', 1.0),  # 三元组损失 Margin
            "benchmark_runs": kwargs.get('benchmark_runs', 10),  # 基准测试运行次数
            "snr": kwargs.get('snr', None)
        }

        # 路径管理类实例化
        self.path_manager = PathManager()
        
        # 检查是否提供了 exp_name，如果有则复用已有实验
        exp_name = kwargs.get('exp_name', None)
        if exp_name:
            # 从已有实验名称恢复 ExperimentInfo
            self.info = ExperimentInfo.from_name(exp_name)
        else:
            # 创建新实验
            self.info = self.path_manager.create_experiment(self.net_type, self.exp_description,
                                                            base_exp_num=self.base_version)
        # 处理父实验
        if self.info.is_extension:
            self.base_info = ExperimentInfo.from_name(self.path_manager.find_base_exp_path(self.info).name)
            self.base_net_type = NetworkType[self.base_info.net_type]

        # ===== 实验目录 =====
        path = self.path_manager.get_exp_paths(self.info)
        # 获取实验模型路径
        self.MODEL_DIR = path["root"]
        self.MODEL_WEIGHTS_DIR = path["weights"]
        self.MODEL_EVAL_DIR = path["eval"]

        # 新生成数据集文件路径
        if self.preprocess_type == PreprocessType.STFT:
            self.filename_train_prepared_data = f"train_data_{self.preprocess_type.value}.h5"
        elif self.preprocess_type == PreprocessType.WST:
            self.filename_train_prepared_data = f"train_data_{self.preprocess_type.value}_j{self.WST_J}q{self.WST_Q}.h5"

        # PCA 相关的路径 (仅在需要时使用)
        if self.is_pca_train:
            self.PCA_DATA_DIR = path["pca_dir"]
            self.PCA_FILE_INPUT = path["pca_input"]
            self.PCA_FILE_OUTPUT = path["pca_output"](self.PCA_DIM_TEST)
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
        """递归地将所有配置转换为可JSON序列化的字典"""

        def _serialize(obj):
            # 1. 处理 NumPy 类型 (数组和标量)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            # 2. 处理枚举类型
            if isinstance(obj, Enum):
                return obj.name
            # 3. 处理字典 (递归处理内容)
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            # 4. 处理列表/元组
            if isinstance(obj, (list, tuple)):
                return [_serialize(i) for i in obj]
            # 5. 处理 Path 或其他有路径属性的对象
            if hasattr(obj, '__fspath__'):
                return str(obj)
            # 6. 基本类型直接返回
            return obj

        # 需要排除的非序列化属性
        excluded_keys = {'path_manager', 'info', 'base_info'}

        # 遍历实例的所有属性并序列化
        return {k: _serialize(v) for k, v in self.__dict__.items() if k not in excluded_keys}

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
        if 'preprocess_type' in data:
            kwargs['preprocess_type'] = PreprocessType[data['preprocess_type']]

        # 网络类型
        if 'net_type' in data:
            net_type = NetworkType[data['net_type']]

        # 描述和版本，确保重新构造时路径一致
        kwargs['test_list'] = data.get('test_list', [1, 5, 10, 20, 50, 100, 150, 200, 250, 300])
        kwargs['exp_description'] = data.get('exp_description', 'Base')
        kwargs['base_version'] = data.get('base_version', None)
        kwargs['is_pca_train'] = data.get('is_pca_train', True)
        kwargs['is_pca_test'] = data.get('is_pca_test', True)
        kwargs['exp_name'] = Path(model_dir).name

        # 重新初始化对象
        return cls(mode=mode, net_type=net_type, **kwargs)


def set_seed(seed=42):
    """设置随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    cfg = Config(mode=Mode.TRAIN, NET_TYPE=NetworkType.ResNet)
